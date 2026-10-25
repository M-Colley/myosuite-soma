"""End-to-end local conversion using independently generated release fixtures."""

from __future__ import annotations

import json
import struct

import mujoco
import numpy as np
import pytest

from myosuite_soma import cli, glb_export, soma
from myosuite_soma.soma_cache import write_archive

from ._fixtures import create_synthetic_release
from .test_package_boundary import _run


_MODEL = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="root" pos=".2 .3 .4"><freejoint/><geom size=".03" mass="2"/>
  <body name="arm-base" pos="0 .1 0"><geom size=".02" mass=".2"/>
  <body name="arm" pos="0 .9 0"><joint name="shoulder" axis="0 0 1" ref=".15" range="-1 1"/>
    <geom size=".03" mass="1"/>
  <body name="forearm" pos="1 0 0"><joint name="elbow" axis="0 0 1" ref="-.2" range="-1 1"/>
    <geom size=".03" mass=".5"/>
  </body></body></body></body></worldbody></mujoco>"""


@pytest.fixture
def release(tmp_path):
    root, checkpoint = create_synthetic_release(tmp_path)
    model_path = tmp_path / "model.xml"
    model_path.write_text(_MODEL)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    rest = model.qpos0.copy()
    rest[-2:] = [0.6, -0.4]
    profile = {
        "data_root": str(root),
        "subject": "S1",
        "checkpoint": str(checkpoint),
        "device": "cpu",
        "body_map": {
            "Hips": {"body": "root", "parent_body": "world"},
            "RightArm": {"body": "arm", "parent_body": "root"},
            "RightForeArm": {"body": "forearm", "parent_body": "arm"},
        },
        "root_body": "root",
        "anchor_body": "root",
        "anchor_joint": "Hips",
        "rest_qpos": rest.tolist(),
    }
    config_path = tmp_path / "soma.json"
    config_path.write_text(json.dumps(profile))
    return model, model_path, profile, config_path


def _document(path):
    blob = path.read_bytes()
    size = struct.unpack_from("<I", blob, 12)[0]
    document = json.loads(blob[20 : 20 + size])
    return document, blob[28 + size :]


def _accessor(document, blob, index):
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = {5123: "<u2", 5125: "<u4", 5126: "<f4"}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[accessor["type"]]
    return np.frombuffer(
        blob,
        dtype,
        count=accessor["count"] * width,
        offset=view.get("byteOffset", 0) + accessor.get("byteOffset", 0),
    ).reshape(accessor["count"], width)


def _skin(document, blob):
    node = next(node for node in document["nodes"] if "skin" in node)
    mesh = document["meshes"][node["mesh"]]
    attributes = mesh["primitives"][0]["attributes"]
    rig = document["skins"][node["skin"]]
    names = [document["nodes"][joint]["name"] for joint in rig["joints"]]
    positions = _accessor(document, blob, attributes["POSITION"])
    # glTF Y-up -> MuJoCo Z-up, independently stated from the file contract.
    points = np.column_stack((positions[:, 0], -positions[:, 2], positions[:, 1]))
    weights = np.zeros((len(points), len(names)))
    block = 0
    while f"JOINTS_{block}" in attributes:
        joints = _accessor(document, blob, attributes[f"JOINTS_{block}"]).astype(int)
        values = _accessor(document, blob, attributes[f"WEIGHTS_{block}"])
        for slot in range(4):
            np.add.at(
                weights, (np.arange(len(points)), joints[:, slot]), values[:, slot]
            )
        block += 1
    return mesh, points, names, weights


def _source_parameters(model):
    return {
        name: getattr(model, name).copy()
        for name in (
            "qpos0",
            "body_pos",
            "body_quat",
            "body_mass",
            "body_inertia",
            "jnt_pos",
            "jnt_axis",
            "jnt_range",
            "dof_damping",
            "actuator_gainprm",
            "actuator_biasprm",
            "tendon_range",
        )
    }


def test_reference_predicts_at_model_qpos0_and_preserves_model(release, tmp_path):
    model, _, profile, config = release
    original = _source_parameters(model)
    qpos0 = model.qpos0.copy()
    reference = soma.generate_animation(
        config, model, qpos0[None], np.array([0.0]), (), np.empty((1, 0))
    )
    at_calibration = soma.generate_animation(
        config,
        model,
        np.asarray(profile["rest_qpos"])[None],
        np.array([0.0]),
        (),
        np.empty((1, 0)),
    )
    expected = next(mesh for mesh in reference.meshes if mesh.kind == "skin")
    other = next(mesh for mesh in at_calibration.meshes if mesh.kind == "skin")
    assert np.max(np.abs(expected.points - other.points)) > 0.05
    output = glb_export.export_reference_glb(config, model, tmp_path / "reference.glb")
    document, blob = _document(output)
    skin, points, names, weights = _skin(document, blob)
    assert skin["extras"]["kind"] == "skin"
    assert len(document["meshes"]) == 1
    np.testing.assert_allclose(points, expected.points[0], atol=1e-7)
    assert names == ["root", "arm-base", "arm", "forearm"]
    expected_weights = np.array([[1, 0, 0, 0], [0, 0, 0.4, 0.6], [0, 0, 0, 1]] * 2)
    np.testing.assert_allclose(weights, expected_weights, atol=3e-8)
    provenance = document["extras"]["myosuite_soma"]["source_metadata"]["glb_rigging"]
    assert provenance["reference"] == "model.qpos0"
    assert provenance["reference_qpos"] == qpos0.tolist()
    assert provenance["forward_physics_changed"] is False
    assert provenance["deformation_during_playback"] == "native_body_lbs_approximation"
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(model, name), value)


def test_reference_can_include_all_selected_individual_muscles(release, tmp_path):
    model, _, _, config = release
    output = glb_export.export_reference_glb(
        config, model, tmp_path / "anatomy.glb", include_muscles=True
    )
    document, blob = _document(output)
    assert len(document["meshes"]) == 3
    assert {mesh["name"] for mesh in document["meshes"]} == {
        "Skin",
        "biceps",
        "triceps",
    }
    assert _skin(document, blob)[0]["extras"]["kind"] == "skin"
    assert all("skin" in node for node in document["nodes"] if "mesh" in node)
    provenance = document["extras"]["myosuite_soma"]["source_metadata"]["glb_rigging"]
    assert provenance["includes_muscles"] is True


@pytest.mark.parametrize("override", [False, True])
def test_native_weights_follow_named_ancestor_or_explicit_override(
    release, tmp_path, override
):
    model, _, profile, config = release
    del profile["body_map"]["RightArm"]
    config.write_text(json.dumps(profile))
    output = glb_export.export_reference_glb(
        config,
        model,
        tmp_path / "binding.glb",
        joint_map={"RightArm": "arm-base"} if override else None,
    )
    document, blob = _document(output)
    _, _, names, actual = _skin(document, blob)
    expected_by_body = {
        "root": np.array([1, 0 if override else 0.4, 0] * 2),
        "arm-base": np.array([0, 0.4 if override else 0, 0] * 2),
        "arm": np.zeros(6),
        "forearm": np.array([0, 0.6, 1] * 2),
    }
    expected = np.column_stack([expected_by_body[name] for name in names])
    np.testing.assert_allclose(actual, expected, atol=3e-8)
    resolved = document["extras"]["myosuite_soma"]["source_metadata"]["glb_rigging"][
        "resolved_joint_bodies"
    ]
    assert resolved == {
        "Hips": "root",
        "RightArm": "arm-base" if override else "root",
        "RightForeArm": "forearm",
    }


def test_weighted_unmapped_root_fails_without_spatial_guess(release, tmp_path):
    model, _, profile, config = release
    del profile["body_map"]["Hips"]
    for key in ("root_body", "anchor_body", "anchor_joint"):
        del profile[key]
    config.write_text(json.dumps(profile))
    output = tmp_path / "not-created" / "missing-root.glb"
    with pytest.raises(ValueError, match="Weighted SOMA joints.*Hips"):
        glb_export.export_reference_glb(config, model, output)
    assert not output.exists() and not output.parent.exists()
    # The caller can resolve this explicitly without editing pose calibration.
    glb_export.export_reference_glb(config, model, output, joint_map={"Hips": "root"})
    assert output.is_file()


@pytest.mark.parametrize(
    "mapping,match",
    [
        ({"RightArm": "missing-body"}, "source body is missing"),
        ({"TypoJoint": "root"}, "Unknown SOMA joint"),
        ({"RightArm": 0}, "joint_map must map"),
        (["RightArm", "root"], "joint_map must map"),
    ],
)
def test_invalid_joint_mapping_does_not_create_output(
    release, tmp_path, mapping, match
):
    model, _, _, config = release
    path = tmp_path / "not-created" / "invalid.glb"
    with pytest.raises(ValueError, match=match):
        glb_export.export_reference_glb(config, model, path, joint_map=mapping)
    assert not path.exists() and not path.parent.exists()


def test_reference_rejects_conditioning_without_loading_artifact(release, tmp_path):
    model, _, profile, config = release
    profile["conditioning_model"] = "missing-conditioner.npz"
    config.write_text(json.dumps(profile))
    path = tmp_path / "conditioned.glb"
    with pytest.raises(ValueError, match="pose-only config"):
        glb_export.export_reference_glb(config, model, path)
    assert not path.exists()


def test_skinning_weights_are_independent_copies_of_native_influences(release):
    model, _, _, config_path = release
    config = soma.SomaConfig.from_json(config_path)
    predictor = soma._load_predictor(config, config_path)
    rotations = soma.retarget_rotations(
        config, model, model.qpos0[None], predictor.joint_names, predictor.parents
    )
    before = predictor.predict(rotations)
    original = predictor.skinning_weights()
    assert set(original) == {"Skin", "biceps", "triceps"}
    expected_skin = np.array([[1, 0, 0], [0, 0.4, 0.6], [0, 0, 1]] * 2)
    np.testing.assert_allclose(original["Skin"], expected_skin, atol=3e-8)
    for mesh, binding in zip(predictor.meshes, predictor._bindings):
        np.testing.assert_array_equal(original[mesh.name], binding[3].detach().numpy())
    untouched = {name: value.copy() for name, value in original.items()}
    for value in original.values():
        value[:] = -99
    copied_again = predictor.skinning_weights()
    for name in copied_again:
        np.testing.assert_array_equal(copied_again[name], untouched[name])
        assert not np.shares_memory(copied_again[name], original[name])
    after = predictor.predict(rotations)
    np.testing.assert_array_equal(after.skin, before.skin)
    for name in after.muscles:
        np.testing.assert_array_equal(after.muscles[name], before.muscles[name])


def test_rigged_xml_cli_runs_without_optional_myosuite(release, tmp_path):
    _, model_path, _, config = release
    output = tmp_path / "cli.glb"
    joint_map = tmp_path / "joint-map.json"
    joint_map.write_text(json.dumps({"RightArm": "root"}))
    arguments = [
        "export-glb",
        "rigged",
        "--config",
        str(config),
        "--model",
        str(model_path),
        "--joint-map",
        str(joint_map),
        str(output),
    ]
    _run(f"""
from myosuite_soma.cli import main
main({arguments!r})
assert not any(name == 'myosuite' or name.startswith('myosuite.') for name in sys.modules)
""")
    assert output.is_file()
    document, blob = _document(output)
    assert len(document["meshes"]) == 1
    _, _, names, weights = _skin(document, blob)
    np.testing.assert_allclose(weights[1, names.index("root")], 0.4, atol=3e-8)
    np.testing.assert_allclose(weights[1, names.index("forearm")], 0.6, atol=3e-8)


@pytest.mark.parametrize("archive", [False, True])
def test_animation_cli_converts_numeric_cache_without_torch_or_myosuite(
    tmp_path, archive
):
    points = np.array(
        [
            [[0.1, -0.2, 0.3], [0.8, -0.2, 0.3], [0.1, 0.4, 0.3]],
            [[0.1, -0.2, 0.4], [0.8, -0.2, 0.4], [0.1, 0.4, 0.6]],
            [[0.2, -0.2, 0.7], [0.8, -0.2, 0.7], [0.2, 0.4, 0.9]],
        ],
        dtype=np.float32,
    )
    animation = soma.SomaAnimation(
        np.array([1.25, 1.31, 1.43]),
        (soma.AnimatedMesh("Skin", np.array([[0, 1, 2]]), points, kind="skin"),),
    )
    cache = tmp_path / ("chunks" if archive else "cache.npz")
    if archive:
        write_archive(cache, animation, chunk_frames=1)
    else:
        soma.save_animation(cache, animation)
    output = tmp_path / "animated.glb"
    arguments = ["export-glb", "animation", str(cache), str(output)]
    _run(
        f"""
from myosuite_soma.cli import main
main({arguments!r})
assert 'torch' not in sys.modules
assert not any(name == 'myosuite' or name.startswith('myosuite.') for name in sys.modules)
""",
        block_torch=True,
    )
    document, blob = _document(output)
    metadata = document["extras"]["myosuite_soma"]
    assert metadata["source_sample_count"] == 3
    assert metadata["source_time_start_seconds"] == 1.25
    sampler = document["animations"][0]["samplers"][0]
    np.testing.assert_allclose(
        _accessor(document, blob, sampler["input"]).ravel(), [0, 0.06, 0.18], atol=1e-8
    )
    assert len(document["meshes"][0]["primitives"][0]["targets"]) == 2


def test_cli_reports_input_errors_and_closes_optional_environment(
    release, tmp_path, monkeypatch
):
    from types import SimpleNamespace
    import sys

    model, _, _, config = release
    closed = []
    fake_env = SimpleNamespace(
        unwrapped=SimpleNamespace(model=model), close=lambda: closed.append(True)
    )
    monkeypatch.setitem(
        sys.modules, "myosuite", SimpleNamespace(make_env=lambda _: fake_env)
    )
    existing = tmp_path / "existing.glb"
    existing.write_bytes(b"keep me")
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "export-glb",
                "rigged",
                "--env",
                "synthetic",
                "--config",
                str(config),
                str(existing),
            ]
        )
    assert stopped.value.code == 2
    assert closed == [True]
    assert existing.read_bytes() == b"keep me"
