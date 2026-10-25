"""Synthetic rigged GLBs exercised through MyoSuite's actual skin importer."""

# ruff: noqa: E402

from __future__ import annotations

import json
import struct

import mujoco
import numpy as np
import pytest

pytest.importorskip("myosuite.viz.skin_glb", reason="Optional MyoSuite GLB integration")

from myosuite.viz.skin import SkinPose, load_skin, load_skn, save_skn
from myosuite.viz.skin_glb import read_skin_glb

from myosuite_soma.soma import AnimatedMesh, SomaAnimation


def _fixture():
    """Six rotated bodies require more than one glTF influence attribute set."""
    bodies = []
    for index in range(6):
        bodies.append(
            f'<body name="link-{index}" pos=".12 -.03 .18" euler=".12 -.08 .17">'
            f'<joint name="joint-{index}" axis="1 2 3" ref="{.03 * index}"/>'
            '<geom type="sphere" size=".025" mass="1"/>'
        )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        + "".join(bodies)
        + "</body>" * 6
        + "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    names = tuple(f"link-{index}" for index in range(6))
    ids = np.array([model.body(name).id for name in names])
    bind = np.broadcast_to(np.eye(4), (6, 4, 4)).copy()
    bind[:, :3, :3] = data.xmat[ids].reshape(6, 3, 3)
    bind[:, :3, 3] = data.xpos[ids]
    points = np.array(
        [
            [0.05, -0.11, 0.2],
            [0.34, -0.11, 0.2],
            [0.34, 0.07, 0.2],
            [0.05, 0.07, 0.2],
            [0.05, -0.11, 0.73],
            [0.34, -0.11, 0.73],
            [0.34, 0.07, 0.73],
            [0.05, 0.07, 0.73],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4]],
        dtype=np.int32,
    )
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]] * 2, dtype=np.float32)
    raw = np.arange(1, 49, dtype=np.float64).reshape(8, 6)
    weights = raw / raw.sum(axis=1, keepdims=True)
    # Input order deliberately puts muscle first. The native importer must find skin.
    animation = SomaAnimation(
        np.array([3.2]),
        (
            AnimatedMesh("synthetic-muscle", faces, (points + 0.07)[None]),
            AnimatedMesh("synthetic-skin", faces, points[None], uv, kind="skin"),
        ),
    )
    return model, animation, names, bind, weights


def _expected_vertices(points, weights, bind, model, data, names):
    """Independent body-frame LBS, using only native MuJoCo forward kinematics."""
    expected = np.zeros_like(points, dtype=np.float64)
    for column, name in enumerate(names):
        body = model.body(name).id
        local = (points - bind[column, :3, 3]) @ bind[column, :3, :3]
        moved = local @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
        expected += weights[:, column, None] * moved
    return expected


def test_rigged_glb_import_preserves_skin_and_all_native_lbs_influences(tmp_path):
    from myosuite_soma.glb import export_rigged_glb

    model, animation, names, bind, weights = _fixture()
    source = animation.meshes[1]
    original_bind, original_weights = bind.copy(), weights.copy()
    path = export_rigged_glb(
        animation,
        tmp_path / "synthetic.glb",
        body_names=names,
        parents=np.arange(-1, 5),
        bind_matrices=bind,
        weights={"synthetic-skin": weights, "synthetic-muscle": weights[:, ::-1]},
    )
    imported = read_skin_glb(path, model)
    assert imported.bone_names == list(names)
    np.testing.assert_allclose(imported.vert, source.points[0], atol=5e-8)
    np.testing.assert_array_equal(imported.face, source.faces)
    np.testing.assert_allclose(imported.texcoord, source.texcoord, atol=5e-8)
    recovered_weights = np.zeros_like(weights)
    for column, (ids, values) in enumerate(zip(imported.vertid, imported.vertweight)):
        recovered_weights[ids, column] = values
    np.testing.assert_allclose(recovered_weights, weights, rtol=2e-7, atol=1e-8)
    assert np.all((recovered_weights > 0).sum(axis=1) == 6)
    # Public --skin loading follows the same route and the skin survives .skn I/O.
    through_public_loader = load_skin(path, model)
    save_skn(through_public_loader, tmp_path / "synthetic.skn")
    skin = load_skn(tmp_path / "synthetic.skn")
    pose = SkinPose.bind(skin, model)
    data = mujoco.MjData(model)
    for displacement in (np.zeros(6), np.linspace(-0.27, 0.31, 6), np.full(6, 0.47)):
        data.qpos[:] = model.qpos0 + displacement
        mujoco.mj_forward(model, data)
        expected = _expected_vertices(
            source.points[0], weights, bind, model, data, names
        )
        np.testing.assert_allclose(pose.vertices(data), expected, rtol=3e-6, atol=2e-7)
    np.testing.assert_array_equal(bind, original_bind)
    np.testing.assert_array_equal(weights, original_weights)


def test_rigged_glb_keeps_other_anatomy_out_of_native_skin_selection(tmp_path):
    from myosuite_soma.glb import export_rigged_glb

    model, animation, names, bind, weights = _fixture()
    path = export_rigged_glb(
        animation,
        tmp_path / "anatomy.glb",
        body_names=names,
        parents=np.arange(-1, 5),
        bind_matrices=bind,
        weights={mesh.name: weights for mesh in animation.meshes},
    )
    binary = path.read_bytes()
    json_size = struct.unpack_from("<I", binary, 12)[0]
    document = json.loads(binary[20 : 20 + json_size])
    skinned_nodes = [node for node in document["nodes"] if "skin" in node]
    assert len(skinned_nodes) == 2
    assert len(document["meshes"]) == 2
    chosen = document["meshes"][skinned_nodes[0]["mesh"]]
    assert chosen["name"] == "synthetic-skin"
    assert {mesh["name"] for mesh in document["meshes"]} == {
        "synthetic-skin",
        "synthetic-muscle",
    }
    attributes = chosen["primitives"][0]["attributes"]
    assert "JOINTS_1" in attributes and "WEIGHTS_1" in attributes
    imported = read_skin_glb(path, model)
    assert len(imported.vert) == len(animation.meshes[1].points[0])
    assert len(imported.face) == len(animation.meshes[1].faces)
