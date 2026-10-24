# Copyright (c) MyoSuite Authors. All rights reserved.
# ruff: noqa: E402
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Optional adapted renderer regression tests, extracted for the standalone backend.

Modified from the audited MyoSuite renderer tests to use myosuite_soma and skip
when the optional simulation package or renderer adapter is unavailable.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("myosuite", reason="Optional MyoSuite renderer integration")

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.viz import blender_render, muscle_tubes
from myosuite.viz.blender_render import RenderConfig

if "soma_config" not in {field.name for field in fields(RenderConfig)}:
    pytest.skip(
        "Optional MyoSuite SOMA renderer adapter is not installed",
        allow_module_level=True,
    )


class _Env(gym.Env):
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.002" gravity="0 0 0"/>
          <asset><mesh name="fixture" vertex="0 0 0 .2 0 0 0 .2 0 0 0 .2"/></asset>
          <worldbody><geom name="goal" type="mesh" mesh="fixture" pos="0 2 0"/>
          <geom name="fence" type="mesh" mesh="fixture" pos="0 -2 0"/>
          <geom name="room" type="mesh" mesh="fixture" pos="0 0 3" contype="0" conaffinity="0"/>
          <body name="carriage"><joint name="slide" type="slide" axis="1 0 0"/>
            <geom name="moving-box" type="box" size=".1 .1 .1"/>
            <geom name="bone" type="mesh" mesh="fixture" contype="0" conaffinity="0"/>
            <site name="anchor" pos="0 0 0"/>
          </body><site name="end" pos="0 1 0"/></worldbody>
          <tendon><spatial name="test-path"><site site="anchor"/><site site="end"/></spatial></tendon>
          <actuator><motor joint="slide"/><muscle name="flexor" tendon="test-path" lengthrange="0.9 1.3" force="10"/></actuator>
        </mujoco>""")
        self.data = mujoco.MjData(self.model)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.closed = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos.astype(np.float32), {}

    def step(self, action: np.ndarray) -> tuple:
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=5)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos.astype(np.float32), 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


def _config(output: Path, checkpoint: Path | None = None) -> RenderConfig:
    return RenderConfig(
        env="test",
        output=output,
        checkpoint=checkpoint,
        seconds=0.04,
        fps=50,
        resolution=(128, 128),
        preview=True,
        samples=16,
    )


def test_export_preserves_metres_timing_motion_and_valid_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 100 Hz episode must not inherit USD's centimetres/60 Hz defaults."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    import myosuite

    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    meta = blender_render.export_rollout(_config(tmp_path))
    stage = Usd.Stage.Open(meta["usd"])
    reference = np.load(tmp_path / "reference.npz")
    assert env.closed
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetTimeCodesPerSecond() == pytest.approx(100)
    assert stage.GetEndTimeCode() == 4
    assert meta["duration"] == pytest.approx(0.04)
    assert not np.allclose(reference["geom_xpos"][0], reference["geom_xpos"][-1])
    box_id = env.model.geom("moving-box").id
    prim = stage.GetPrimAtPath(f"/World/Mesh_Xform_moving_box_id{box_id}_geom")
    assert prim.IsValid()
    for index in [0, 2, 4]:
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(index)
        np.testing.assert_allclose(
            transform.ExtractTranslation(),
            reference["geom_xpos"][index, env.model.geom("moving-box").id],
            atol=1e-7,
        )
    for name in ["goal", "fence"]:
        geom_id = env.model.geom(name).id
        prim = stage.GetPrimAtPath(f"/World/Mesh_Xform_{name}_id{geom_id}_geom")
        assert prim.IsValid()
        assert UsdGeom.Imageable(prim).ComputeVisibility(0) == "inherited"
        assert f"{name}_id{geom_id}_geom" not in meta["scenery_objects"]
        assert f"{name}_id{geom_id}_geom" in meta["static_objects"]
    room = f"room_id{env.model.geom('room').id}_geom"
    assert meta["scenery_objects"] == [room]
    assert meta["tendon_objects"]
    assert all("test_path" in name for name in meta["tendon_objects"])
    bone_id = env.model.geom("bone").id
    assert meta["bone_objects"] == [f"bone_id{bone_id}_geom"]
    assert meta["muscle_objects"] == ["Muscle_flexor"]
    tube = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Muscles/Muscle_flexor"))
    assert tube.GetPointsAttr().GetNumTimeSamples() == 5
    assert np.load(tmp_path / "muscles.npz")["activation"].shape == (5, 1)


def test_path_mode_colours_mujoco_segments_by_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thin paths keep MuJoCo's segments and still record each muscle's activation."""
    pytest.importorskip("pxr")
    from dataclasses import replace

    from pxr import Usd

    import myosuite

    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    meta = blender_render.export_rollout(replace(_config(tmp_path), muscles="paths"))
    tendon = env.model.tendon("test-path").id
    assert meta["muscle_objects"] == [f"_id{tendon}_tendon"]
    assert meta["replaced_tendons"] == []
    assert np.load(tmp_path / "muscles.npz")["activation"].shape == (5, 1)
    assert not Usd.Stage.Open(meta["usd"]).GetPrimAtPath("/World/Muscles").IsValid()


def test_skin_is_exported_as_an_animated_mesh_that_follows_its_bone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .skn bound to the moving body becomes one animated, UV-mapped USD mesh."""
    pytest.importorskip("pxr")
    import struct
    from dataclasses import replace

    from pxr import Usd, UsdGeom

    import myosuite

    vert = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], np.float32)
    skn = tmp_path / "skin.skn"
    skn.write_bytes(
        struct.pack("4i", 3, 3, 1, 1)
        + vert.tobytes()
        + np.array([[0, 0], [1, 0.25], [0.5, 1]], np.float32).tobytes()
        + np.arange(3, dtype=np.int32).tobytes()
        + b"carriage".ljust(40, b"\0")
        + np.array([0, 0, 0, 1, 0, 0, 0], np.float32).tobytes()
        + struct.pack("i", 3)
        + np.arange(3, dtype=np.int32).tobytes()
        + np.ones(3, np.float32).tobytes()
    )
    texture = tmp_path / "Tone.PNG"
    texture.write_bytes(b"not decoded at export")
    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    out = tmp_path / "out"
    meta = blender_render.export_rollout(
        replace(_config(out), skin=skn, skin_style="opaque", skin_texture=texture)
    )
    stage = Usd.Stage.Open(meta["usd"])
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Skin/Skin"))
    assert meta["skin_objects"] == ["Skin"] and meta["skin_style"] == "opaque"
    points = mesh.GetPointsAttr()
    assert points.GetNumTimeSamples() == 5
    qpos = np.load(out / "reference.npz")["qpos"][:, 0]
    for frame in (0, 4):
        np.testing.assert_allclose(
            np.array(points.Get(frame)), vert + [qpos[frame], 0, 0], atol=1e-6
        )
    # UVs with V up (the .skn stores it down the image); the texture travels with the scene.
    np.testing.assert_allclose(
        np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()),
        [[0, 1], [1, 0.75], [0.5, 0]],
    )
    assert meta["skin_texture"] == "skin_texture.png"
    assert (out / "skin_texture.png").read_bytes() == texture.read_bytes()
    with pytest.raises(ValueError, match="needs a skin"):
        RenderConfig(env="x", output=Path("o"), skin_texture=texture)
    with pytest.raises(ValueError, match="not found"):
        RenderConfig(
            env="x", output=Path("o"), skin="fullbody", skin_texture=tmp_path / "no.png"
        )
    with pytest.raises(ValueError, match="skin style"):
        RenderConfig(env="x", output=Path("o"), skin_style="glass")
    with pytest.raises(ValueError, match="alpha"):
        RenderConfig(env="x", output=Path("o"), skin_alpha=0)


def test_muscle_colours_follow_the_mujoco_viewer_on_paths() -> None:
    """Paths blend near-black to red with activation ** 0.25; uniform is validated."""
    colour = blender_render._muscle_colour
    np.testing.assert_allclose(colour(0.0, paths=True), blender_render.PATH_RELAXED)
    np.testing.assert_allclose(colour(1.0, paths=True), blender_render.PATH_ACTIVE)
    halfway = (np.add(blender_render.PATH_RELAXED, blender_render.PATH_ACTIVE)) / 2
    np.testing.assert_allclose(colour(0.5**4, paths=True), halfway)
    np.testing.assert_allclose(colour(1.0, paths=False), blender_render.MUSCLE_ACTIVE)
    RenderConfig(env="x", output=Path("o"), muscle_color="uniform")
    with pytest.raises(ValueError, match="colour"):
        RenderConfig(env="x", output=Path("o"), muscle_color="rainbow")


def test_muscle_tube_keeps_its_volume_as_the_path_shortens() -> None:
    """Shortening a path thickens the belly; the enclosed volume is unchanged."""
    profile = muscle_tubes.radius_profile(0.5)
    assert profile.max() == pytest.approx(1.0, abs=1e-2)
    np.testing.assert_allclose(profile[[0, -1]], muscle_tubes.TENDON_RADIUS_RATIO)
    assert np.argmax(profile) < len(profile) // 2  # belly towards the origin
    lengths = np.array([0.3, 0.2])
    radii = muscle_tubes.belly_radius(4e-5, lengths, profile)
    assert radii[1] > radii[0]
    np.testing.assert_allclose(
        muscle_tubes.belly_volume(radii, lengths, profile), 4e-5, rtol=1e-9
    )
    for length, radius in zip(lengths, radii):
        centre = np.zeros((len(profile), 3))
        centre[:, 2] = np.linspace(0, length, len(profile))
        points, counts, indices = muscle_tubes.tube_mesh(centre, radius * profile)
        assert counts.sum() == len(indices)
        assert indices.max() == len(points) - 1
        ring = points[:-2].reshape(len(profile), muscle_tubes.RING_POINTS, 3)
        np.testing.assert_allclose(
            np.linalg.norm(ring[..., :2], axis=-1),
            np.broadcast_to(radius * profile[:, None], ring.shape[:2]),
            rtol=1e-6,
        )


def test_forearm_muscle_volumes_are_anatomical() -> None:
    """Forearm bellies stay near measured volumes and leave long distal tendons.

    References: Holzbaur et al. 2007, J Biomech 40:742 (adult upper-limb muscle
    volumes; FDS split evenly over its four compartments).
    """
    from myosuite import make_env

    env = make_env("myoHandPoseRandom-v0")
    try:
        env.reset(seed=0)
        model, data = env.unwrapped.model, env.unwrapped.data
        muscles = muscle_tubes.muscle_actuators(model)
        path = data.ten_length[model.actuator_trnid[muscles, 0]]
        peak, fraction = muscle_tubes.muscle_shape(model, path)
        volume = muscle_tubes.belly_volume(
            peak, path, muscle_tubes.radius_profile(fraction)
        )
    finally:
        env.close()
    names = [model.actuator(a).name for a in muscles]
    for name, measured in {"FCR_r": 17e-6, "FCU_r": 26e-6, "FDS3_r": 14e-6}.items():
        assert 0.5 < volume[names.index(name)] / measured < 2
    flexor = names.index("FDS3_r")
    belly_end = (1 - fraction[flexor]) / 4 + fraction[flexor]
    assert belly_end < 0.7  # the belly ends in the forearm, a tendon runs to the finger


def test_large_muscles_take_measured_volumes() -> None:
    """Hip, thigh, calf and shoulder muscles approach their reference volumes.

    Broad muscles on short paths (vasti, gluteals, adductor magnus) stay under
    the slenderness cap, so they reach at least 45% of the reference.
    """
    from myosuite import make_env

    env = make_env("myoMimicFullbody-v0")
    try:
        env.reset(seed=0)
        model, data = env.unwrapped.model, env.unwrapped.data
        muscles = muscle_tubes.muscle_actuators(model)
        path = data.ten_length[model.actuator_trnid[muscles, 0]]
        peak, fraction = muscle_tubes.muscle_shape(model, path)
        volume = muscle_tubes.belly_volume(
            peak, path, muscle_tubes.radius_profile(fraction)
        )
    finally:
        env.close()
    groups = [muscle_tubes._reference_group(model.actuator(a).name) for a in muscles]
    for name, reference in muscle_tubes.REFERENCE_VOLUMES.items():
        side = "_r" if name.islower() else ""
        total = volume[[g == (name, side) for g in groups]].sum()
        assert 0.45 < total / reference <= 1.0 + 1e-6, name


def test_resample_path_is_even_in_arc_length() -> None:
    """Wrap points of uneven spacing become evenly spaced centreline samples."""
    path = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.3, 0.0]])
    points, length = muscle_tubes.resample_path(path, 9)
    assert length == pytest.approx(0.4)
    np.testing.assert_allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), 0.05)


def test_checkpoint_symlink_keeps_format_and_rejects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HF-style blob symlinks retain .zip dispatch and never become random rollouts."""
    pytest.importorskip("pxr")
    import myosuite
    from myosuite.utils import checkpoint_utils

    blob = tmp_path / "blob_without_extension"
    blob.write_bytes(b"fixture")
    checkpoint = tmp_path / "policy.zip"
    try:
        checkpoint.symlink_to(blob)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or privilege")
        raise
    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(2,)), action_space=env.action_space
    )
    monkeypatch.setattr(checkpoint_utils, "load_sb3_model", lambda path: model)
    with pytest.raises(ValueError, match="observation/action shapes"):
        blender_render.export_rollout(_config(tmp_path / "output", checkpoint))
    assert env.closed
    assert not (tmp_path / "output" / "render.json").exists()


def test_muscle_mesh_option_checks_env_and_downloads_on_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atlas meshes come from the Hugging Face dataset; any env may ask for them."""
    import huggingface_hub

    RenderConfig(env="myoArmReachRandom-v0", output=tmp_path, muscle_mesh="atlas")
    with pytest.raises(ValueError, match="not found"):
        RenderConfig(env="test", output=tmp_path, muscle_mesh=str(tmp_path / "x.glb"))
    RenderConfig(env="myoMimicFullbody-v0", output=tmp_path, muscle_mesh="atlas-hd")
    calls = []

    def download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(tmp_path / "cached.glb")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert blender_render._resolve_muscle_mesh("atlas-hd") == tmp_path / "cached.glb"
    assert calls == [
        {
            "repo_id": "myohub/myosuite-assets",
            "filename": "muscles/fullbody_atlas.glb",
            "repo_type": "dataset",
        }
    ]
    assert blender_render._resolve_muscle_mesh("own.glb") == Path("own.glb")

    def offline(**kwargs: str) -> str:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(RuntimeError, match="myohub/myosuite-assets"):
        blender_render._resolve_muscle_mesh("atlas")


def test_export_records_bone_rest_poses_for_a_muscle_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muscle meshes bind at qpos0, whatever pose the episode starts in."""
    pytest.importorskip("pxr")
    import myosuite

    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _id: env)
    glb = tmp_path / "muscles.glb"
    glb.write_bytes(b"glTF")
    config = RenderConfig(
        **{**vars(_config(tmp_path / "out")), "muscle_mesh": str(glb)}
    )
    meta = blender_render.export_rollout(config)
    assert (tmp_path / "out" / meta["muscle_mesh"]).read_bytes() == b"glTF"
    data = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, data)
    bone = env.model.geom("bone").id
    (rig,) = meta["muscle_rigs"]
    rest = np.asarray(rig[f"bone_id{bone}_geom"]).reshape(4, 4)
    np.testing.assert_allclose(rest[:3, 3], data.geom_xpos[bone])
    np.testing.assert_allclose(rest[:3, :3], data.geom_xmat[bone].reshape(3, 3))


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf"])
def test_invalid_duration_rejected_before_export(seconds: str, tmp_path: Path) -> None:
    """Non-finite durations must not start an unbounded recording."""
    argv = [
        "--env",
        "test",
        "--random",
        "--seconds",
        seconds,
        "--output",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit) as exc:
        blender_render.main(argv)
    assert exc.value.code == 2


def test_skin_bones_bind_under_either_phalanx_naming() -> None:
    """``midph2_r`` (bundled skin) binds to ``2midph_r`` (musclemimic_models) and back; unknown bones raise."""
    from myosuite.viz.skin import Skin, SkinPose

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<body name="pelvis"><geom size="0.1"/>'
        '<body name="2midph_r"><geom size="0.01"/></body>'
        '<body name="distph3_l"><geom size="0.01"/></body>'
        "</body></worldbody></mujoco>"
    )

    def skin(bones: list[str]) -> Skin:
        n = len(bones)
        return Skin(
            vert=np.zeros((1, 3), np.float32),
            texcoord=np.zeros((0, 2), np.float32),
            face=np.zeros((0, 3), np.int32),
            bone_names=bones,
            bindpos=np.zeros((n, 3), np.float32),
            bindquat=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
            vertid=[np.zeros(1, np.int32)] * n,
            vertweight=[np.ones(1, np.float32)] * n,
        )

    pose = SkinPose.bind(skin(["pelvis", "midph2_r", "3distph_l"]), model)
    body = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # noqa: E731
    assert list(pose.body_ids) == [body("pelvis"), body("2midph_r"), body("distph3_l")]
    with pytest.raises(ValueError, match="midph9_r"):
        SkinPose.bind(skin(["pelvis", "midph9_r"]), model)


def test_cli_defaults_to_the_newest_checkpoint_and_never_to_random(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --checkpoint the CLI renders find_checkpoint's policy; with none found it stops (no random fallback)."""
    import myosuite.utils.checkpoint_utils as checkpoint_utils

    rendered: list[RenderConfig] = []
    monkeypatch.setattr(
        blender_render, "render_rollout", lambda config, **kw: rendered.append(config)
    )
    found = tmp_path / "logs" / "model_3000.pt"
    monkeypatch.setattr(checkpoint_utils, "find_checkpoint", lambda env_id: found)
    blender_render.main(
        ["--env", "myoLegWalk-v0", "--output", str(tmp_path / "a"), "--export-only"]
    )
    assert rendered[-1].checkpoint == found

    blender_render.main(
        ["--env", "myoLegWalk-v0", "--output", str(tmp_path / "b"), "--random"]
    )
    assert rendered[-1].checkpoint is None

    monkeypatch.setattr(checkpoint_utils, "find_checkpoint", lambda env_id: None)
    with pytest.raises(SystemExit):
        blender_render.main(["--env", "myoLegWalk-v0", "--output", str(tmp_path / "c")])
    assert len(rendered) == 2


@pytest.mark.parametrize("preview", [False, True])
def test_render_rollout_ends_with_the_result_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    preview: bool,
) -> None:
    """The last line printed is the absolute path of the video (or of the preview image)."""
    import myosuite.utils.video_io as video_io

    meta = {"samples": 3, "duration": 0.1, "usd": "x.usdc"}
    monkeypatch.setattr(blender_render, "export_rollout", lambda config: meta)
    monkeypatch.setattr(blender_render.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(video_io, "write_video", lambda *a, **kw: None)
    out = tmp_path / "render"
    blender_render.render_rollout(
        RenderConfig(env="myoLegWalk-v0", output=out, preview=preview)
    )
    last = capsys.readouterr().out.strip().splitlines()[-1]
    expected = out.resolve() / ("preview.png" if preview else "video.mp4")
    assert last == f"{'Preview' if preview else 'Video'}: {expected}"


def test_authored_binding_matches_bones_by_rest_position() -> None:
    # Another model build: other side naming and swapped phalanx IDs.
    joints = {
        "l_femur_id5_geom": [0.1, 0.0, 0.8],
        "_distph_r_id9_geom": [0.3, 0.0, 0.9],
        "_distph_r_id8_geom": [0.3, 0.02, 0.9],
    }
    bones = {
        "femur_l_id7_geom": [0.1, 0.001, 0.8],
        "_distph_r_id8_geom": [0.3, 0.0, 0.9],
        "_distph_r_id9_geom": [0.3, 0.02, 0.9],
    }
    assert blender_render._binding_targets(joints, bones) == {
        "l_femur_id5_geom": "femur_l_id7_geom",
        "_distph_r_id9_geom": "_distph_r_id8_geom",
        "_distph_r_id8_geom": "_distph_r_id9_geom",
    }


@pytest.mark.parametrize(
    "joints",
    [{"a": [0.0, 0.0, 0.1]}, {"a": [0.0, 0.0, 0.0], "b": [0.001, 0.0, 0.0]}],
)
def test_authored_binding_rejects_no_match_or_shared_bones(
    joints: dict[str, list[float]],
) -> None:
    with pytest.raises(ValueError, match="scene bone"):
        blender_render._binding_targets(joints, {"femur_r_id1_geom": [0.0, 0.0, 0.0]})


def test_authored_binding_leaves_out_joints_of_missing_bones() -> None:
    """An arm has no femur: its joint is left out, the others bind."""
    joints = {"humerus": [0.0, 0.0, 1.4], "femur": [0.0, 0.0, 0.8]}
    bones = {"humerus_id3_geom": [0.0, 0.0, 1.4]}
    assert blender_render._binding_targets(joints, bones) == {
        "humerus": "humerus_id3_geom"
    }


_BODY = """
<mujoco><asset><mesh name="{p}femur" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"/>
  <mesh name="{p}tibia" vertex="0 0 0 .2 0 0 0 .1 0 0 0 .1"/></asset>
  <worldbody><body name="{p}thigh" pos="{x} 0 1">
    <joint name="{p}hip" type="hinge" axis="0 1 0" ref="{ref}"/>
    <geom name="{p}femur" type="mesh" mesh="{p}femur"/>{shank}
  </body></worldbody></mujoco>"""
_SHANK = """<body name="{p}shank" pos="0 0 -.4">
      <geom name="{p}tibia" type="mesh" mesh="{p}tibia"/></body>"""


def _bones(model: mujoco.MjModel) -> dict[str, int]:
    return {
        model.geom(g).name: g for g in range(model.ngeom) if model.geom_dataid[g] >= 0
    }


def test_muscle_rigs_follow_the_reference_body_by_bone_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copies and partial bodies bind in the reference's rest pose, not their own."""
    body = lambda p, x, ref, shank=True: mujoco.MjModel.from_xml_string(  # noqa: E731
        _BODY.format(p=p, x=x, ref=ref, shank=_SHANK.format(p=p) if shank else "")
    )
    reference = body("", 0, 0)
    monkeypatch.setattr(blender_render, "_muscle_mesh_reference", lambda _: reference)
    expected = blender_render._rest_poses(reference, _bones(reference))
    config = RenderConfig(env="other", output=tmp_path, muscle_mesh="atlas")

    partial = body("", 3, 0.5, shank=False)  # elsewhere, another rest angle, no shank
    assert blender_render._muscle_rigs(config, partial, _bones(partial)) == [
        {"femur": expected["femur"]}
    ]

    spec = mujoco.MjSpec()
    for i in range(2):
        copy = mujoco.MjSpec.from_string(
            _BODY.format(p="", x=i, ref=0, shank=_SHANK.format(p=""))
        )
        spec.attach(copy, frame=spec.worldbody.add_frame(), prefix=f"w{i}_")
    two = spec.compile()
    assert blender_render._muscle_rigs(config, two, _bones(two)) == [
        {f"w{i}_femur": expected["femur"], f"w{i}_tibia": expected["tibia"]}
        for i in range(2)
    ]

    with pytest.raises(ValueError, match="No bone"):
        blender_render._muscle_rigs(config, _Env().model, {"bone": 4})


def test_muscle_rigs_of_a_file_are_the_rendered_models_own_rest_pose(
    tmp_path: Path,
) -> None:
    glb = tmp_path / "own.glb"
    glb.write_bytes(b"glTF")
    model = _Env().model
    bones = {"bone": model.geom("bone").id}
    config = RenderConfig(env="test", output=tmp_path, muscle_mesh=str(glb))
    assert blender_render._muscle_rigs(config, model, bones) == [
        blender_render._rest_poses(model, bones)
    ]
    assert blender_render._muscle_rigs(_config(tmp_path), model, bones) == []


def test_motion_clip_is_replayed_frame_by_frame_at_its_own_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named qpos clip drives the export instead of a policy, at the clip's rate."""
    pytest.importorskip("pxr")
    import myosuite

    clip = tmp_path / "clip.npz"
    np.savez(
        clip,
        qpos=np.linspace(0, 0.3, 6)[:, None],
        joint_names=np.array(["slide"]),
        frequency=50.0,
    )
    monkeypatch.setattr(myosuite, "make_env", lambda _id: _Env())
    config = RenderConfig(
        env="test",
        output=tmp_path / "out",
        motion=clip,
        seconds=1,
        fps=50,
        preview=True,
    )
    meta = blender_render.export_rollout(config)
    reference = np.load(tmp_path / "out" / "reference.npz")
    assert meta["motion_source"] == "motion"
    assert not np.load(tmp_path / "out" / "muscles.npz")["activation"].any()
    assert meta["dt"] == pytest.approx(0.02)
    np.testing.assert_allclose(reference["qpos"][:, 0], np.linspace(0, 0.3, 6))
    np.testing.assert_allclose(reference["qvel"], 3.0)
    assert meta["velocity_source"] == "finite_difference"
    assert np.max(np.abs(reference["actuator_velocity"])) > 0.1
    back = tmp_path / "back.npz"  # sliding back to 0 shortens the muscle path
    np.savez(
        back,
        qpos=np.linspace(0.3, 0, 6)[:, None],
        joint_names=np.array(["slide"]),
        frequency=50.0,
    )
    blender_render.export_rollout(
        RenderConfig(
            env="test",
            output=tmp_path / "proxy",
            motion=back,
            motion_activation=True,
            seconds=1,
            fps=50,
            preview=True,
        )
    )
    proxy = np.load(tmp_path / "proxy" / "muscles.npz")["activation"][:, 0]
    assert proxy[0] == 0 and proxy[1:].min() > 0
    with np.load(tmp_path / "proxy" / "reference.npz") as recorded:
        independent = _Env()
        for frame in range(len(recorded["time"])):
            independent.data.qpos[:] = recorded["qpos"][frame]
            independent.data.qvel[:] = recorded["qvel"][frame]
            independent.data.act[:] = recorded["act"][frame]
            independent.data.ctrl[:] = recorded["ctrl"][frame]
            mujoco.mj_forward(independent.model, independent.data)
            np.testing.assert_allclose(
                recorded["actuator_force"][frame], independent.data.actuator_force
            )
    with pytest.raises(ValueError, match="not both"):
        RenderConfig(env="test", output=tmp_path, motion=clip, checkpoint=clip)


def test_replay_velocities_handle_free_joint_quaternions_and_recorded_values() -> None:
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><freejoint/><geom size="0.1"/></body></worldbody></mujoco>
    """)
    qpos = np.tile(model.qpos0, (4, 1))
    time = np.arange(4) / 10
    qpos[:, 0] = time * 2
    qpos[:, 3] = np.cos(time * 1.5 / 2)
    qpos[:, 6] = np.sin(time * 1.5 / 2)
    clip = SimpleNamespace(qpos=qpos, qvel=None, frequency_hz=10)
    velocity = blender_render._motion_velocities(clip, model)
    np.testing.assert_allclose(
        velocity[:, [0, 5]], np.tile([2, 1.5], (4, 1)), atol=1e-12
    )
    np.testing.assert_allclose(velocity[:, [1, 2, 3, 4]], 0)
    clip.qvel = np.ones((4, 6))
    np.testing.assert_array_equal(
        blender_render._motion_velocities(clip, model), clip.qvel
    )
    clip.qvel = np.ones((3, 6))
    with pytest.raises(ValueError, match="velocities"):
        blender_render._motion_velocities(clip, model)


def test_motion_replay_never_unpickles_object_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pxr")
    import myosuite

    marker = tmp_path / "unpickled.txt"

    class ObjectValue:
        def __reduce__(self):
            return open, (str(marker), "w")

    clip = tmp_path / "unsafe.npz"
    np.savez(
        clip,
        qpos=np.array([[ObjectValue()], [ObjectValue()]], dtype=object),
        frequency=50.0,
    )
    env = _Env()
    monkeypatch.setattr(myosuite, "make_env", lambda _: env)
    with pytest.raises(ValueError, match="Object arrays"):
        blender_render.export_rollout(
            RenderConfig(env="test", output=tmp_path / "out", motion=clip)
        )
    assert not marker.exists()
    assert env.closed


@pytest.mark.parametrize("frequency", [0, -1, float("nan"), float("inf")])
def test_invalid_replay_frequency_is_rejected(frequency: float) -> None:
    env = _Env()
    with pytest.raises(ValueError, match="frequency"):
        blender_render._motion_velocities(
            SimpleNamespace(qpos=np.zeros((2, 1)), qvel=None, frequency_hz=frequency),
            env.model,
        )


@pytest.mark.parametrize(
    "values",
    [
        {"fps": 24.5},
        {"fps": float("inf")},
        {"samples": 1.5},
        {"resolution": (640, 640.5)},
        {"muscle_scale": float("inf")},
    ],
)
def test_render_rejects_fractional_timeline_and_nonfinite_scale(
    tmp_path: Path, values: dict
) -> None:
    with pytest.raises(ValueError):
        RenderConfig(env="test", output=tmp_path, **values)


def test_render_normalizes_numpy_integer_settings_for_metadata(tmp_path: Path) -> None:
    import json

    config = RenderConfig(
        env="test",
        output=tmp_path,
        fps=np.int64(24),
        samples=np.int32(16),
        resolution=(np.int64(640), np.int64(480)),
    )
    assert json.loads(json.dumps([config.fps, config.samples, config.resolution])) == [
        24,
        16,
        [640, 480],
    ]


def test_direct_export_protects_an_existing_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pxr")
    import myosuite

    (tmp_path / "render.json").write_text("{}")
    monkeypatch.setattr(
        myosuite, "make_env", lambda _: pytest.fail("overwriting a recorded run")
    )
    with pytest.raises(FileExistsError):
        blender_render.export_rollout(_config(tmp_path))


def _soma_fixture(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    time: np.ndarray,
    actuator_names: list[str],
    activation: np.ndarray,
):
    from myosuite_soma.soma import (
        AnimatedMesh,
        SomaAnimation,
        activation_digest,
        model_digest,
        qpos_digest,
    )

    triangle = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], np.float32)
    points = np.repeat(triangle[None], len(time), axis=0)
    points[:, :, 0] += qpos[:, :1]
    points[:, 2, 1] += np.arange(len(time)) * 0.01
    faces = np.array([[0, 1, 2]], np.int32)
    a = np.asarray(activation[:, 0], dtype=np.float32)
    return SomaAnimation(
        time=np.asarray(time),
        meshes=[
            AnimatedMesh("short_head", faces, points, activation=a),
            AnimatedMesh("long_head", faces, points + [0, 0, 0.1], activation=a),
            AnimatedMesh("surface", faces, points + [0, 0, 0.2], kind="skin"),
        ],
        metadata={
            "coordinate_system": "mujoco_world",
            "units": "m",
            "deformation_driver": "pose_only",
            "source_qpos_sha256": qpos_digest(qpos),
            "source_model_sha256": model_digest(model),
            "source_actuator_names": actuator_names,
            "source_activation_sha256": activation_digest(actuator_names, activation),
            "muscle_map": {
                "short_head": {actuator_names[0]: 1},
                "long_head": {actuator_names[0]: 1},
            },
        },
    )


def test_soma_export_uses_deforming_world_meshes_and_explicit_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SOMA vertices bypass bone skinning and anatomy colours need not match actuator order."""
    pytest.importorskip("pxr")
    from dataclasses import replace
    from pxr import Usd, UsdGeom, UsdSkel

    import myosuite
    from myosuite_soma import soma

    config_path = tmp_path / "soma.json"
    config_path.write_text("{}")
    envs = []

    def make_env(_: str) -> _Env:
        env = _Env()
        envs.append(env)
        return env

    generated = []

    def generate(config, model, qpos, time, actuator_names, activation, *, physiology):
        assert config == config_path
        assert actuator_names == ["flexor"]
        assert activation.shape == (5, 1)
        assert set(physiology) == {
            "activation",
            "actuator_length",
            "actuator_velocity",
            "actuator_force",
        }
        assert all(value.shape == activation.shape for value in physiology.values())
        animation = _soma_fixture(model, qpos, time, actuator_names, activation)
        generated.append(animation)
        return animation

    monkeypatch.setattr(myosuite, "make_env", make_env)
    monkeypatch.setattr(soma, "generate_animation", generate)
    baseline = tmp_path / "baseline"
    blender_render.export_rollout(_config(baseline))
    out = tmp_path / "soma"
    meta = blender_render.export_rollout(replace(_config(out), soma_config=config_path))
    stage = Usd.Stage.Open(meta["usd"])
    assert meta["soma"]["cache"] == "soma.npz"
    assert stage.GetTimeCodesPerSecond() == pytest.approx(100)
    assert not stage.GetPrimAtPath("/World/Muscles").IsValid()
    assert len(meta["muscle_objects"]) == 2
    assert len(meta["skin_objects"]) == 1
    assert meta["replaced_tendons"]
    assert meta["tendon_objects"] == []
    assert meta["muscle_mesh"] is None
    meshes = [
        UsdGeom.Mesh(p)
        for p in stage.Traverse()
        if p.GetTypeName() == "Mesh" and p.GetPath().HasPrefix("/World/SOMA")
    ]
    assert len(meshes) == 3
    for mesh, surface in zip(meshes, generated[0].meshes):
        assert mesh.GetPointsAttr().GetNumTimeSamples() == 5
        assert not mesh.GetPrim().HasAPI(UsdSkel.BindingAPI)
        assert not UsdGeom.Xformable(mesh.GetPrim().GetParent()).GetOrderedXformOps()
        np.testing.assert_allclose(mesh.GetPointsAttr().Get(4), surface.points[4])
    tracks = blender_render._activation_tracks(out, meta)
    np.testing.assert_array_equal(tracks[0][1], generated[0].meshes[0].activation)
    np.testing.assert_array_equal(tracks[1][1], generated[0].meshes[1].activation)
    assert (
        blender_render._initial_muscle_colour("Muscle_SOMA_unmapped", meta)
        == blender_render.MUSCLE_UNIFORM
    )
    assert (
        blender_render._initial_muscle_colour(meta["muscle_objects"][0], meta)
        == blender_render.MUSCLE_RELAXED
    )
    with (
        np.load(out / "reference.npz") as actual,
        np.load(baseline / "reference.npz") as before,
    ):
        for key in (
            "qpos",
            "qvel",
            "ctrl",
            "act",
            "actuator_length",
            "actuator_velocity",
            "actuator_force",
            "body_xpos",
            "body_xquat",
        ):
            np.testing.assert_array_equal(actual[key], before[key])
        assert actual["actuator_names"].tolist() == ["", "flexor"]
    assert all(env.closed for env in envs)
    cache = soma.load_animation(out / "soma.npz")
    np.testing.assert_array_equal(cache.meshes[0].points, generated[0].meshes[0].points)

    monkeypatch.setattr(
        soma,
        "generate_animation",
        lambda *args: pytest.fail("cached render ran inference"),
    )
    cached = blender_render.export_rollout(
        replace(_config(tmp_path / "cached"), soma_cache=out / "soma.npz")
    )
    assert cached["muscle_objects"] == meta["muscle_objects"]
    stale = tmp_path / "stale-colours.npz"
    soma.save_animation(
        stale,
        replace(
            cache,
            metadata={
                **cache.metadata,
                "source_activation_sha256": soma.activation_digest(
                    ["flexor"], np.zeros((5, 1), np.float32)
                ),
            },
        ),
    )
    with pytest.raises(ValueError, match="activation|physiology"):
        blender_render.export_rollout(
            replace(
                _config(tmp_path / "stale-colours"),
                soma_cache=stale,
            )
        )
    with pytest.raises(ValueError, match="time|sample"):
        blender_render.export_rollout(
            replace(
                _config(tmp_path / "wrong-time"),
                soma_cache=out / "soma.npz",
                seconds=0.03,
            )
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"soma_config": True},
        {"skin": "fullbody"},
        {"muscle_mesh": "atlas"},
        {"muscle_mesh_env": "other"},
        {"muscle_scale": 2},
        {"skin_inflate": 0.01},
    ],
)
def test_soma_conflicting_geometry_rejected(tmp_path: Path, extra: dict) -> None:
    source = tmp_path / "cache.npz"
    source.touch()
    if extra.get("soma_config"):
        extra = {"soma_config": source}
    with pytest.raises(ValueError, match="SOMA"):
        RenderConfig(env="test", output=tmp_path / "out", soma_cache=source, **extra)


@pytest.mark.parametrize("tolerance", [None, 0.1])
def test_soma_chunk_archive_roundtrip_through_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tolerance: float | None
) -> None:
    pytest.importorskip("pxr")
    from dataclasses import replace

    import myosuite
    from myosuite_soma import soma
    from myosuite_soma.soma_cache import AnimationArchive

    source = tmp_path / "config.json"
    source.write_text("{}")
    monkeypatch.setattr(myosuite, "make_env", lambda _id: _Env())

    def generate(config, model, qpos, time, actuator_names, activation, **kwargs):
        fixture = _soma_fixture(model, qpos, time, actuator_names, activation)
        return replace(
            fixture,
            meshes=tuple(
                replace(mesh, points=mesh.points.astype(np.float32))
                for mesh in fixture.meshes
            ),
        )

    monkeypatch.setattr(soma, "generate_animation", generate)
    output = tmp_path / "chunked"
    metadata = blender_render.export_rollout(
        replace(
            _config(output),
            soma_config=source,
            soma_cache_format="chunks",
            soma_cache_error_mm=tolerance,
        )
    )
    assert metadata["soma"]["cache"] == "soma.chunks"
    archive = AnimationArchive(output / "soma.chunks")
    assert archive.manifest["measured_max_error_m"] <= (tolerance or 0) / 1000
    assert len(archive.load().time) == 5
    monkeypatch.setattr(
        soma,
        "generate_animation",
        lambda *args, **kwargs: pytest.fail("Cached geometry must bypass inference"),
    )
    replay = blender_render.export_rollout(
        replace(_config(tmp_path / "replay"), soma_cache=output / "soma.chunks")
    )
    assert replay["soma"]["archive"]["format"] == "soma-chunks"
    np.testing.assert_array_equal(
        soma.load_animation(tmp_path / "replay" / "soma.npz").meshes[0].points,
        archive.load().meshes[0].points,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"soma_cache_format": "invalid"},
        {"soma_cache_error_mm": 0.1},
        {"soma_cache_format": "chunks", "soma_cache_error_mm": float("nan")},
        {"soma_cache_format": "chunks", "soma_cache_error_mm": -1},
    ],
)
def test_soma_cache_encoding_settings_are_validated(
    tmp_path: Path, settings: dict
) -> None:
    with pytest.raises(ValueError, match="soma_cache"):
        RenderConfig(env="test", output=tmp_path, **settings)


def test_soma_cli_selection_and_invalid_activation_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "cache.npz"
    source.touch()
    rendered = []
    monkeypatch.setattr(
        blender_render, "render_rollout", lambda config, **kw: rendered.append(config)
    )
    common = ["--env", "test", "--output", str(tmp_path / "out"), "--random"]
    blender_render.main(common + ["--soma-cache", str(source)])
    assert rendered[-1].soma_cache == source
    with pytest.raises(SystemExit):
        blender_render.main(
            common + ["--soma-cache", str(source), "--soma-config", str(source)]
        )
    np.savez(
        tmp_path / "muscles.npz",
        activation=np.zeros((2, 1)),
        visual_activation=np.zeros((2, 2)),
    )
    tracks = blender_render._activation_tracks(
        tmp_path,
        {
            "muscle_activation_columns": {"headB": 1, "headA": 0},
            "dt": 0.01,
        },
    )
    assert [name for name, _ in tracks] == ["headB", "headA"]
    with pytest.raises(ValueError, match="missing activation"):
        blender_render._activation_tracks(
            tmp_path, {"muscle_activation_columns": {"bad": 2}}
        )


def test_long_soma_names_keep_unique_blender_activation_bindings() -> None:
    pytest.importorskip("pxr")
    from pxr import Usd
    from myosuite_soma.soma import AnimatedMesh, SomaAnimation

    points = np.zeros((2, 3, 3), np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    meshes = tuple(
        AnimatedMesh("a" * 159 + suffix, faces, points, activation=np.array([0, 1]))
        for suffix in ("x", "y")
    )
    animation = SomaAnimation(np.array([0, 1]), meshes)
    stage = Usd.Stage.CreateInMemory()
    names, _, tracks, columns = blender_render._write_soma_animation(stage, animation)
    assert len(set(names)) == 2
    assert max(map(len, names)) <= 55
    assert set(columns) == set(names)
    np.testing.assert_array_equal(tracks, np.array([[0, 0], [1, 1]]))


def test_activation_keyframes_preserve_small_peaks_and_linear_interpolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    class Object:
        name = "Muscle_SOMA_0000_test"
        type = "MESH"

        def __init__(self):
            self.keyframes = {}
            self.point = SimpleNamespace(interpolation="BEZIER")
            self.animation_data = SimpleNamespace(
                action=SimpleNamespace(
                    fcurves=[SimpleNamespace(keyframe_points=[self.point])]
                )
            )

        def keyframe_insert(self, field, *, frame):
            self.keyframes[frame] = self.color

    obj = Object()
    monkeypatch.setitem(
        sys.modules, "bpy", SimpleNamespace(data=SimpleNamespace(objects=[obj]))
    )
    np.savez(
        tmp_path / "muscles.npz", visual_activation=np.array([[0.0], [0.025], [0.0]])
    )
    blender_render._animate_activation(
        tmp_path,
        {
            "dt": 0.02,
            "fps": 50,
            "soma": True,
            "muscle_activation_columns": {obj.name: 0},
        },
        SimpleNamespace(frame_start=0, frame_end=2),
    )
    assert set(obj.keyframes) == {0, 1, 2}
    np.testing.assert_allclose(
        obj.keyframes[1], blender_render._muscle_colour(0.025, False)
    )
    assert obj.point.interpolation == "LINEAR"
