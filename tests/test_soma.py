# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SOMA retargeting, explicit activation mapping and safe cache contracts."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite_soma.soma import (
    AnimatedMesh,
    BodyMapping,
    SomaAnimation,
    SomaConfig,
    activation_digest,
    generate_animation,
    inspect_config,
    load_animation,
    map_activations,
    model_digest,
    qpos_digest,
    retarget_rotations,
    save_animation,
    validate_animation,
)


_XML = """<mujoco><worldbody>
  <body name="upper" quat="0.9238795325 0.3826834324 0 0">
    <joint name="shoulder" axis="0 0 1"/><geom size="0.1"/>
    <body name="lower" pos="0 -0.3 0" quat="0.9659258263 0.2588190451 0 0">
      <joint name="elbow" axis="0 1 0"/><geom size="0.08"/>
    </body>
  </body>
</worldbody></mujoco>"""


def _animation() -> SomaAnimation:
    return SomaAnimation(
        np.array([0.0, 0.03]),
        (
            AnimatedMesh(
                "biceps_r",
                np.array([[0, 1, 2]], dtype=np.int32),
                np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]] * 2, dtype=np.float32),
                np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
                np.array([0.2, 0.6]),
            ),
        ),
        {"coordinate_system": "mujoco_world", "units": "m"},
    )


def test_animation_round_trip_is_numeric_and_keeps_fixed_topology(
    tmp_path: Path,
) -> None:
    animation = _animation()
    path = tmp_path / "animation.npz"
    save_animation(path, animation)
    with np.load(path, allow_pickle=False) as cache:
        assert all(cache[name].dtype.kind != "O" for name in cache.files)
    loaded = load_animation(path)
    assert loaded.metadata == animation.metadata
    assert loaded.meshes[0].name == "biceps_r"
    np.testing.assert_array_equal(loaded.meshes[0].points, animation.meshes[0].points)
    np.testing.assert_array_equal(loaded.meshes[0].faces, animation.meshes[0].faces)
    np.testing.assert_array_equal(
        loaded.meshes[0].texcoord, animation.meshes[0].texcoord
    )
    np.testing.assert_array_equal(
        loaded.meshes[0].activation, animation.meshes[0].activation
    )


@pytest.mark.parametrize(
    "mesh_change,match",
    [
        ({"name": "../../escape"}, "filename-safe"),
        ({"faces": np.array([[0, 1, 3]])}, "out-of-range"),
        ({"faces": np.array([[0.0, 1.0, 2.0]])}, "integer triangles"),
        ({"faces": np.array([[[0, 1, 2]], [[0, 1, 2]]])}, "integer triangles"),
        ({"points": np.zeros((1, 3, 3))}, "shape"),
        ({"points": np.full((2, 3, 3), np.nan)}, "finite"),
        ({"points": np.full((2, 3, 3), 1e300)}, "float32"),
        ({"texcoord": np.zeros((3, 3))}, "texcoord"),
        ({"texcoord": np.full((3, 2), 1e300)}, "float32"),
        ({"activation": np.array([0.5, 1.1])}, "activation"),
    ],
)
def test_cache_rejects_invalid_meshes(mesh_change: dict, match: str) -> None:
    animation = _animation()
    with pytest.raises(ValueError, match=match):
        validate_animation(
            replace(animation, meshes=(replace(animation.meshes[0], **mesh_change),))
        )


@pytest.mark.parametrize("time", [[0, 0], [1, 0], [0, np.inf], [[0, 1]]])
def test_cache_rejects_invalid_times(time: list) -> None:
    with pytest.raises(ValueError):
        validate_animation(replace(_animation(), time=np.array(time)))


def test_cache_rejects_pickle_and_extra_arrays(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(path, manifest=np.array({"version": 1}, dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        load_animation(path)
    save_animation(path, _animation())
    with np.load(path, allow_pickle=False) as cache:
        arrays = {name: cache[name] for name in cache.files}
    arrays["unexpected"] = np.array([1])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Unexpected"):
        load_animation(path)


def test_cache_reports_truncated_zip_as_invalid_cache(tmp_path: Path) -> None:
    path = tmp_path / "truncated.npz"
    save_animation(path, _animation())
    path.write_bytes(path.read_bytes()[:-22])
    with pytest.raises(ValueError, match="Invalid SOMA animation cache"):
        load_animation(path)


def test_provenance_checks_detect_stale_pose_model_and_time() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    qpos = np.zeros((2, model.nq))
    animation = _animation()
    animation.metadata.update(
        source_qpos_sha256=qpos_digest(qpos), source_model_sha256=model_digest(model)
    )
    validate_animation(animation, time=animation.time, qpos=qpos, model=model)
    with pytest.raises(ValueError, match="source qpos"):
        validate_animation(animation, qpos=qpos + 0.2)
    with pytest.raises(ValueError, match="times"):
        validate_animation(animation, time=animation.time + 0.1)
    model.jnt_axis[1] = [0, 0, 1]
    with pytest.raises(ValueError, match="source model"):
        validate_animation(animation, model=model)


def test_cache_detects_changed_physiology_at_the_same_pose() -> None:
    animation = _animation()
    names = ["long", "short"]
    activation = np.array([[0.1, 0.5], [0.6, 0.2]])
    mapping = {"biceps_r": {"long": 3, "short": 1}}
    animation = replace(
        animation,
        meshes=(replace(animation.meshes[0], activation=np.array([0.2, 0.5])),),
    )
    animation.metadata["muscle_map"] = mapping
    animation.metadata["source_activation_sha256"] = activation_digest(
        names, activation
    )
    validate_animation(animation, actuator_names=names, activation=activation)
    with pytest.raises(ValueError, match="source activation"):
        validate_animation(animation, actuator_names=names, activation=activation + 0.1)
    with pytest.raises(ValueError, match="source activation"):
        validate_animation(animation, actuator_names=names[::-1], activation=activation)
    geometry_only = replace(
        animation, meshes=(replace(animation.meshes[0], activation=None),)
    )
    validate_animation(geometry_only, actuator_names=names, activation=activation + 0.1)


def test_cache_rejects_forged_colors_even_with_correct_source_hash() -> None:
    animation = _animation()
    names = ["long", "short"]
    activation = np.array([[0.2, 0.8], [0.6, 0.4]])
    animation.metadata.update(
        source_activation_sha256=activation_digest(names, activation),
        muscle_map={"biceps_r": {"long": 1}},
    )
    validate_animation(animation, actuator_names=names, activation=activation)
    forged = replace(
        animation,
        meshes=(replace(animation.meshes[0], activation=np.array([0.3, 0.6])),),
    )
    with pytest.raises(ValueError, match="mapped activation"):
        validate_animation(forged, actuator_names=names, activation=activation)
    animation.metadata.pop("muscle_map")
    with pytest.raises(ValueError, match="muscle_map"):
        validate_animation(animation, actuator_names=names, activation=activation)
    validate_animation(animation)


def test_cache_checks_mapping_covers_exact_colored_muscle_meshes() -> None:
    animation = _animation()
    names, activation = ["long"], np.array([[0.2], [0.6]])
    animation.metadata.update(
        source_activation_sha256=activation_digest(names, activation),
        muscle_map={"absent": {"long": 1}},
    )
    with pytest.raises(ValueError, match="muscle_map"):
        validate_animation(animation, actuator_names=names, activation=activation)


def test_model_digest_includes_kinematic_joint_and_mocap_routing() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    for field in ("body_jntadr", "body_jntnum", "body_mocapid", "jnt_dofadr"):
        values = getattr(model, field)
        original = values.copy()
        before = model_digest(model)
        values[-1] += 1
        assert model_digest(model) != before, field
        values[:] = original


@pytest.mark.parametrize("joint", ["freejoint", 'joint type="ball"'])
def test_retarget_rejects_invalid_source_quaternions(
    tmp_path: Path, joint: str
) -> None:
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body name="root"><{joint}/><geom size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    config = SomaConfig(
        tmp_path, "S1", tmp_path / "checkpoint", body_map={"Hips": BodyMapping("root")}
    )
    valid = model.qpos0[None].copy()
    retarget_rotations(config, model, valid, ["Hips"], np.array([-1]))
    for bad in (np.zeros_like(valid), np.full_like(valid, 2.0)):
        with pytest.raises(ValueError, match="unit quaternion"):
            retarget_rotations(config, model, bad, ["Hips"], np.array([-1]))
    with pytest.raises(ValueError, match="unit quaternion"):
        retarget_rotations(
            replace(config, rest_qpos=np.zeros(model.nq)),
            model,
            valid,
            ["Hips"],
            np.array([-1]),
        )


def test_qpos_retarget_rejects_mocap_motion_but_allows_relative_child_motion(
    tmp_path: Path,
) -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="tracked" mocap="true"><geom size=".1"/>'
        '<body name="child"><joint name="hinge"/><geom size=".05"/>'
        "</body></body></worldbody></mujoco>"
    )
    config = SomaConfig(
        tmp_path, "S1", tmp_path / "checkpoint", body_map={"Hips": BodyMapping("child")}
    )
    with pytest.raises(ValueError, match="mocap"):
        retarget_rotations(config, model, np.array([[0.5]]), ["Hips"], np.array([-1]))
    relative = replace(config, body_map={"Hips": BodyMapping("child", "tracked")})
    result = retarget_rotations(
        relative, model, np.array([[0.5]]), ["Hips"], np.array([-1])
    )
    assert np.isclose(np.trace(result[0, 0]), 1 + 2 * np.cos(0.5))


def test_activation_weights_do_not_overflow() -> None:
    result = map_activations(
        {"biceps": {"long": 1e308, "short": 1e308}},
        ["long", "short"],
        np.array([[0.2, 0.8], [0.4, 0.6]]),
    )
    np.testing.assert_allclose(result["biceps"], [0.5, 0.5])


def test_config_resolves_paths_and_rejects_unknown_code_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    raw = {
        "data_root": "data",
        "subject": "S1",
        "checkpoint": "weights/model.pth",
        "body_map": {"RightForeArm": {"body": "lower", "parent_body": "upper"}},
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = SomaConfig.from_json(path)
    assert config.data_root == tmp_path / "data"
    assert config.checkpoint == tmp_path / "weights/model.pth"
    assert config.body_map["RightForeArm"].body == "lower"
    raw["python_module"] = "untrusted"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown SOMA config"):
        SomaConfig.from_json(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("subject", "../S1"),
        ("scale", "1"),
        ("scale", True),
        ("root_body", 17),
        ("root_offset", [1, 2]),
        ("root_offset", [0.1, 0, 0]),
        ("batch_size", 0),
    ],
)
def test_config_rejects_invalid_scalar_values(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "config.json"
    raw = {
        "data_root": "data",
        "subject": "S1",
        "checkpoint": "weights.pth",
        field: value,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        SomaConfig.from_json(path)


def test_local_retarget_removes_ancestor_motion_and_applies_basis(
    tmp_path: Path,
) -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    basis = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    config = SomaConfig(
        tmp_path,
        "S1",
        tmp_path / "checkpoint",
        body_map={"RightForeArm": BodyMapping("lower", "upper")},
        basis=basis,
    )
    qpos = np.array([[0, 0], [0.7, 0], [0.7, 0.8]])
    rotations = retarget_rotations(
        config, model, qpos, ("Hips", "RightForeArm"), np.array([-1, 0])
    )
    np.testing.assert_allclose(
        rotations[:2], np.broadcast_to(np.eye(3), rotations[:2].shape), atol=1e-12
    )
    reference, moved = mujoco.MjData(model), mujoco.MjData(model)
    moved.qpos[:] = qpos[2]
    mujoco.mj_forward(model, reference)
    mujoco.mj_forward(model, moved)
    rest = reference.xmat[1].reshape(3, 3).T @ reference.xmat[2].reshape(3, 3)
    relative = moved.xmat[1].reshape(3, 3).T @ moved.xmat[2].reshape(3, 3)
    np.testing.assert_allclose(
        rotations[2, 1], basis @ relative @ rest.T @ basis.T, atol=1e-12
    )
    assert not np.allclose(rotations[2, 1], np.eye(3))


def test_right_elbow_preset_uses_mujoco_fk_without_mutating_source(
    tmp_path: Path,
) -> None:
    from ._fixtures import create_elbow_model

    model = create_elbow_model()
    data = mujoco.MjData(model)
    data.qpos[:] = 0.4
    data.qvel[:] = 0.2
    data.act[:] = np.linspace(0.1, 0.6, model.na)
    data.ctrl[:] = np.linspace(0.2, 0.7, model.nu)
    mujoco.mj_forward(model, data)
    snapshot = {
        name: np.array(getattr(data, name), copy=True)
        for name in ("qpos", "qvel", "act", "ctrl", "xmat", "xpos")
    }
    before = model_digest(model)
    config = SomaConfig(tmp_path, "S1", tmp_path / "checkpoint", preset="myoElbow1D6")
    poses = np.stack((model.qpos0, model.qpos0 + 0.8))
    rotations = retarget_rotations(
        config,
        model,
        poses,
        ("Hips", "RightArm", "RightForeArm"),
        np.array([-1, 0, 1]),
    )
    np.testing.assert_allclose(
        rotations[0], np.broadcast_to(np.eye(3), (3, 3, 3)), atol=1e-12
    )
    assert np.isclose(np.trace(rotations[1, 2]), 1 + 2 * np.cos(0.8))
    assert before == model_digest(model)
    for name, original in snapshot.items():
        np.testing.assert_array_equal(getattr(data, name), original)


def test_activation_mapping_is_weighted_and_independent_of_column_order() -> None:
    weights = {"biceps_r": {"long": 3.0, "short": 1.0}}
    activation = np.array([[0.2, 0.6], [0.8, 0.4]])
    mapped = map_activations(weights, ["short", "long"], activation)
    np.testing.assert_allclose(mapped["biceps_r"], [0.5, 0.5])
    with pytest.raises(ValueError, match="missing"):
        map_activations(weights, ["short", "other"], activation)
    with pytest.raises(ValueError, match="unique"):
        map_activations(weights, ["long", "long"], activation)


def test_generation_uses_model_api_world_placement_and_pose_only_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    data_root = tmp_path / "data"
    data_root.mkdir()
    (tmp_path / "model.pth").touch()
    config = {
        "data_root": "data",
        "subject": "S1",
        "checkpoint": "model.pth",
        "body_map": {"RightForeArm": {"body": "lower", "parent_body": "upper"}},
        "output_translation": [1, 2, 3],
        "scale": 2,
        "muscle_map": {"biceps_r": {"biceps": 1}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    triangles = np.array([[0, 1, 2]])
    rest_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    spec = SimpleNamespace(
        name="biceps_r", kind="muscle", faces=triangles, texcoord=None
    )
    skin = SimpleNamespace(name="skin", kind="skin", faces=triangles, texcoord=None)

    def predict(rotations: np.ndarray, root_positions: np.ndarray) -> SimpleNamespace:
        np.testing.assert_array_equal(root_positions, [[1, 0, 0], [1, 0, 0]])
        points = np.broadcast_to(rest_points, (len(rotations), 3, 3)).copy()
        return SimpleNamespace(muscles={"biceps_r": points}, skin=points)

    predictor = SimpleNamespace(
        joint_names=("Hips", "RightForeArm"),
        parents=np.array([-1, 0]),
        rest_joints=np.array([[1, 0, 0], [1, 1, 0]]),
        meshes=(spec,),
        skin_mesh=skin,
        predict=predict,
        metadata={"dataset_version": 2},
    )
    monkeypatch.setitem(
        sys.modules,
        "myosuite_soma.soma_model",
        SimpleNamespace(load_soma_model=lambda *args, **kwargs: predictor),
    )
    animation = generate_animation(
        config_path,
        model,
        np.zeros((2, model.nq)),
        np.array([0, 0.05]),
        ["biceps"],
        np.array([[0.1], [0.9]]),
    )
    expected = 2 * rest_points + [1, 2, 3]
    np.testing.assert_allclose(animation.meshes[0].points, [expected, expected])
    np.testing.assert_array_equal(animation.meshes[0].activation, [0.1, 0.9])
    assert animation.metadata["deformation_driver"] == "pose_only"
    validate_animation(animation, model=model, qpos=np.zeros((2, model.nq)))


def test_generation_reports_missing_assets_with_config_path(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "data_root": "missing",
                "subject": "S1",
                "checkpoint": "missing.pth",
                "preset": "myoElbow1D6",
            }
        ),
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_string(_XML)
    with pytest.raises(FileNotFoundError, match="SKIM v2 dataset not found"):
        generate_animation(
            path, model, np.zeros((1, model.nq)), np.array([0]), [], np.empty((1, 0))
        )


def test_root_motion_rotates_local_offset_once_and_preserves_physics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis" pos=".4 -.2 1.2">'
        '<freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    original = data.qpos.copy(), data.xpos.copy(), model_digest(model)
    poses = np.repeat(model.qpos0[None], 3, axis=0)
    poses[1:, :3] += [[0.3, -0.1, 0.2], [-0.2, 0.4, 0.1]]
    for frame, angle in enumerate([0.0, np.pi / 2, -np.pi / 4]):
        mujoco.mju_axisAngle2Quat(poses[frame, 3:7], np.array([0.0, 0.0, 1.0]), angle)
    output_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    root_offset = np.array([0.12, 0.03, -0.04])
    translation = np.array([0.1, 0.2, 0.3])
    config_path = tmp_path / "root.json"
    config_path.write_text(
        json.dumps(
            {
                "data_root": ".",
                "subject": "S1",
                "checkpoint": "checkpoint.pth",
                "body_map": {"Hips": {"body": "pelvis"}},
                "basis": output_rotation.T.tolist(),
                "output_rotation": output_rotation.tolist(),
                "output_translation": translation.tolist(),
                "root_offset": root_offset.tolist(),
                "scale": 2.0,
                "anchor_body": "pelvis",
                "anchor_joint": "Hips",
                "root_body": "pelvis",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "checkpoint.pth").touch()
    hip = np.array([1.0, 2.0, 3.0])
    rest_points = hip + np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.2, 0]])

    def predict(rotations: np.ndarray, root_positions: np.ndarray) -> SimpleNamespace:
        np.testing.assert_array_equal(root_positions, np.repeat(hip[None], 3, axis=0))
        points = np.einsum("tij,vj->tvi", rotations[:, 0], rest_points - hip) + hip
        return SimpleNamespace(muscles={"muscle": points}, skin=points)

    predictor = SimpleNamespace(
        joint_names=("Hips",),
        parents=np.array([-1]),
        rest_joints=hip[None],
        meshes=(
            SimpleNamespace(
                name="muscle", kind="muscle", faces=np.array([[0, 1, 2]]), texcoord=None
            ),
        ),
        skin_mesh=SimpleNamespace(
            name="skin", kind="skin", faces=np.array([[0, 1, 2]]), texcoord=None
        ),
        predict=predict,
        metadata={},
    )
    monkeypatch.setitem(
        sys.modules,
        "myosuite_soma.soma_model",
        SimpleNamespace(load_soma_model=lambda *args, **kwargs: predictor),
    )
    animation = generate_animation(
        config_path, model, poses, np.arange(3) * 0.03, [], np.empty((3, 0))
    )
    expected = []
    scratch = mujoco.MjData(model)
    for pose in poses:
        scratch.qpos[:] = pose
        mujoco.mj_forward(model, scratch)
        rotation = scratch.xmat[1].reshape(3, 3)
        anchor = scratch.xpos[1] + rotation @ root_offset + translation
        expected.append(
            anchor + 2 * (rest_points - hip) @ output_rotation.T @ rotation.T
        )
    np.testing.assert_allclose(animation.meshes[0].points, expected, atol=2e-7)
    assert animation.metadata["retargeting"]["root_offset"] == root_offset.tolist()
    np.testing.assert_array_equal(data.qpos, original[0])
    np.testing.assert_array_equal(data.xpos, original[1])
    assert model_digest(model) == original[2]


def test_real_format_assets_generate_cache_from_synthetic_elbow(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ._fixtures import create_elbow_model, create_synthetic_release

    root, checkpoint = create_synthetic_release(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_root": str(root),
                "subject": "S1",
                "checkpoint": str(checkpoint),
                "preset": "myoElbow1D6",
                "muscle_map": {"biceps": {"BIClong": 0.7, "BICshort": 0.3}},
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    inventory = inspect_config(config_path)
    assert inventory["muscle_meshes"] == ["biceps", "triceps"]
    assert inventory["joints"][-1]["name"] == "RightForeArm"
    model = create_elbow_model()
    qpos = np.array([[0.2], [0.6], [1.0]])
    names = [model.actuator(index).name for index in range(model.nu)]
    activation = np.zeros((3, model.nu))
    activation[:, names.index("BIClong")] = [0.1, 0.5, 0.9]
    animation = generate_animation(
        config_path, model, qpos, np.array([0, 0.03, 0.06]), names, activation
    )
    path = tmp_path / "animation.npz"
    save_animation(path, animation)
    loaded = load_animation(path)
    validate_animation(
        loaded, qpos=qpos, model=model, actuator_names=names, activation=activation
    )
    assert [mesh.kind for mesh in loaded.meshes] == ["muscle", "muscle", "skin"]
    np.testing.assert_allclose(loaded.meshes[0].activation, [0.07, 0.35, 0.63])
    assert not np.allclose(loaded.meshes[0].points[0], loaded.meshes[0].points[-1])
