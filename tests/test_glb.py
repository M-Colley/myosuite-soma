"""Decode exported GLBs independently and reconstruct their geometry numerically."""

import json
import struct
from dataclasses import replace

import numpy as np
import pytest

from myosuite_soma import glb
from myosuite_soma.soma import AnimatedMesh, SomaAnimation


def _animation():
    vertices = np.array(
        [
            [[0.4, -0.2, 1.1], [0.7, -0.2, 1.1], [0.4, 0.1, 1.1]],
            [[0.5, -0.1, 1.2], [0.8, -0.1, 1.2], [0.5, 0.2, 1.3]],
            [[0.2, -0.3, 1.3], [0.5, -0.3, 1.3], [0.2, 0.0, 1.5]],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return SomaAnimation(
        np.array([5.25, 5.30, 5.43]),
        (
            AnimatedMesh(
                "Biceps", faces, vertices, activation=np.array([0.1, 0.5, 0.9])
            ),
            AnimatedMesh(
                "Skin",
                faces,
                vertices + np.float32(0.02),
                texcoord=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
                kind="skin",
            ),
        ),
        metadata={"subject": "synthetic", "nested": {"calibration": [1, 2, 3]}},
    )


def _decode(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    assert magic == b"glTF" and version == 2 and length == len(data)
    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    assert json_type == b"JSON" and json_length % 4 == 0
    description = json.loads(data[20 : 20 + json_length])
    bin_length, bin_type = struct.unpack_from("<I4s", data, 20 + json_length)
    assert bin_type == b"BIN\0" and bin_length % 4 == 0
    binary = data[28 + json_length :]
    assert len(binary) == bin_length
    assert 0 <= bin_length - description["buffers"][0]["byteLength"] <= 3
    assert "uri" not in description["buffers"][0]

    def accessor(index):
        spec = description["accessors"][index]
        view = description["bufferViews"][spec["bufferView"]]
        assert view["byteOffset"] % 4 == 0
        dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[spec["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[spec["type"]]
        result = np.frombuffer(
            binary, dtype, count=spec["count"] * width, offset=view["byteOffset"]
        ).reshape(spec["count"], width)
        assert result.nbytes == view["byteLength"]
        if "min" in spec:
            np.testing.assert_array_equal(spec["min"], result.min(axis=0))
            np.testing.assert_array_equal(spec["max"], result.max(axis=0))
        return result

    return description, accessor


def _mujoco_axes(points):
    return points[..., [0, 2, 1]] * [1, -1, 1]


def _worlds(description):
    nodes = description["nodes"]
    parent = {
        child: index
        for index, node in enumerate(nodes)
        for child in node.get("children", [])
    }

    def world(index):
        local = (
            np.asarray(nodes[index].get("matrix", np.eye(4).T.ravel())).reshape(4, 4).T
        )
        return world(parent[index]) @ local if index in parent else local

    return [world(index) for index in range(len(nodes))]


def _reference(animation=None):
    source = _animation() if animation is None else animation
    return replace(
        source,
        time=source.time[:1],
        meshes=tuple(
            replace(
                mesh,
                points=mesh.points[:1],
                activation=mesh.activation[:1] if mesh.activation is not None else None,
            )
            for mesh in source.meshes
        ),
    )


def _binding(animation):
    names = ("pelvis", "thigh_r", "shin_r", "torso", "arm_r", "hand_r")
    parents = np.array([-1, 0, 1, 0, 3, 4])
    matrices = np.broadcast_to(np.eye(4), (len(names), 4, 4)).copy()
    angle = 0.37
    rotation = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    matrices[1, :3, :3] = rotation
    matrices[1, :3, 3] = [0.2, -0.1, 0.6]
    matrices[2, :3, 3] = [-0.1, 0.3, 0.2]
    matrices[3:, :3, 3] = [[0, 0, 1], [0.2, 0, 1.3], [0.4, 0, 1.2]]
    weights = {
        mesh.name: np.array(
            [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], [1, 1, 1, 1, 1, 1]], dtype=float
        )
        for mesh in animation.meshes
    }
    return {
        "body_names": names,
        "parents": parents,
        "bind_matrices": matrices,
        "weights": weights,
    }


def test_baked_nonuniform_samples_and_intermediate_geometry_are_reconstructed(tmp_path):
    source = _animation()
    originals = [mesh.points.copy() for mesh in source.meshes]
    output = glb.export_glb(source, tmp_path / "animation.glb")
    description, accessor = _decode(output)
    assert output.is_absolute()
    assert description["asset"]["version"] == "2.0"
    assert "skins" not in description
    extra = description["extras"]["myosuite_soma"]
    assert extra["source_metadata"] == source.metadata
    assert extra["source_time_start_seconds"] == 5.25
    assert extra["source_time_end_seconds"] == 5.43
    assert extra["coordinate_system"] == "gltf_y_up" and extra["units"] == "m"
    animation = description["animations"][0]
    sampler = animation["samplers"][0]
    assert sampler["interpolation"] == "LINEAR"
    np.testing.assert_allclose(accessor(sampler["input"])[:, 0], [0, 0.05, 0.18])
    keys = accessor(sampler["output"]).reshape(3, 2)
    np.testing.assert_array_equal(keys, [[0, 0], [1, 0], [0, 1]])
    assert len(animation["channels"]) == len(source.meshes)
    for index, (expected, node) in enumerate(zip(source.meshes, description["nodes"])):
        assert node["name"] == expected.name and "skin" not in node
        primitive = description["meshes"][node["mesh"]]["primitives"][0]
        np.testing.assert_array_equal(
            accessor(primitive["indices"]).reshape(-1, 3), expected.faces
        )
        base = accessor(primitive["attributes"]["POSITION"])
        deltas = np.stack(
            [accessor(target["POSITION"]) for target in primitive["targets"]]
        )
        rebuilt = base + np.einsum("tf,fvc->tvc", keys, deltas)
        np.testing.assert_allclose(_mujoco_axes(rebuilt), expected.points, atol=2e-7)
        midway = base + np.einsum("f,fvc->vc", (keys[1] + keys[2]) / 2, deltas)
        np.testing.assert_allclose(
            _mujoco_axes(midway),
            (expected.points[1] + expected.points[2]) / 2,
            atol=2e-7,
        )
        base_normals = accessor(primitive["attributes"]["NORMAL"])
        normal_deltas = np.stack(
            [accessor(target["NORMAL"]) for target in primitive["targets"]]
        )
        normals = _mujoco_axes(
            base_normals + np.einsum("tf,fvc->tvc", keys, normal_deltas)
        )
        for frame in range(3):
            triangle = expected.points[frame].astype(float)
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal /= np.linalg.norm(normal)
            np.testing.assert_allclose(
                normals[frame], np.broadcast_to(normal, (3, 3)), atol=1e-7
            )
        if expected.texcoord is not None:
            uv = accessor(primitive["attributes"]["TEXCOORD_0"])
            np.testing.assert_array_equal(uv * [1, -1] + [0, 1], expected.texcoord)
        np.testing.assert_array_equal(expected.points, originals[index])
        assert description["materials"][primitive["material"]]["doubleSided"]


def test_single_frame_needs_no_morph_animation(tmp_path):
    source = _reference()
    description, accessor = _decode(
        glb.export_glb(source, tmp_path / "static.glb", max_targets=0)
    )
    assert "animations" not in description
    for expected, mesh in zip(source.meshes, description["meshes"]):
        primitive = mesh["primitives"][0]
        assert "targets" not in primitive
        np.testing.assert_allclose(
            _mujoco_axes(accessor(primitive["attributes"]["POSITION"])),
            expected.points[0],
        )


def test_reference_rig_keeps_all_influences_and_reconstructs_world_bind_geometry(
    tmp_path,
):
    source = _reference()
    binding = _binding(source)
    original_weights = {
        name: weights.copy() for name, weights in binding["weights"].items()
    }
    description, accessor = _decode(
        glb.export_rigged_glb(source, tmp_path / "rigged.glb", **binding)
    )
    assert "animations" not in description
    skin = description["skins"][0]
    joints = skin["joints"]
    assert (
        tuple(description["nodes"][index]["name"] for index in joints)
        == binding["body_names"]
    )
    worlds = _worlds(description)
    matrices = (
        accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    )
    np.testing.assert_allclose(
        np.stack(worlds)[:6] @ matrices,
        np.broadcast_to(np.eye(4), (6, 4, 4)),
        atol=2e-7,
    )
    mesh_nodes = [node for node in description["nodes"] if "skin" in node]
    assert description["meshes"][mesh_nodes[0]["mesh"]]["name"] == "Skin"
    by_name = {mesh.name: mesh for mesh in source.meshes}
    for node in mesh_nodes:
        mesh = description["meshes"][node["mesh"]]
        expected = by_name[mesh["name"]]
        primitive = mesh["primitives"][0]
        attributes = primitive["attributes"]
        assert "JOINTS_1" in attributes and "WEIGHTS_1" in attributes
        indices = np.concatenate(
            [accessor(attributes[f"JOINTS_{i}"]) for i in range(2)], axis=1
        )
        weights = np.concatenate(
            [accessor(attributes[f"WEIGHTS_{i}"]) for i in range(2)], axis=1
        )
        dense = np.zeros((3, 6))
        for column in range(8):
            np.add.at(dense, (np.arange(3), indices[:, column]), weights[:, column])
        raw = binding["weights"][expected.name]
        np.testing.assert_allclose(
            dense, raw / raw.sum(axis=1, keepdims=True), atol=2e-8
        )
        vertices = accessor(attributes["POSITION"])
        homogeneous = np.c_[vertices, np.ones(len(vertices))]
        transforms = np.stack([worlds[j] for j in joints]) @ matrices
        rebuilt = np.einsum("vb,bij,vj->vi", dense, transforms, homogeneous)
        np.testing.assert_allclose(
            _mujoco_axes(rebuilt[:, :3]), expected.points[0], atol=2e-7
        )
        np.testing.assert_array_equal(
            binding["weights"][expected.name], original_weights[expected.name]
        )


def test_unused_joint_slots_are_zero_even_with_different_influence_counts(tmp_path):
    source = _reference()
    binding = _binding(source)
    for weights in binding["weights"].values():
        weights[1] = [0, 0, 0, 0, 0, 1]
        weights[2] = [0, 0, 1, 0, 0, 0]
    description, accessor = _decode(
        glb.export_rigged_glb(source, tmp_path / "sparse.glb", **binding)
    )
    for mesh in description["meshes"]:
        attributes = mesh["primitives"][0]["attributes"]
        indices = np.concatenate(
            [accessor(attributes[f"JOINTS_{i}"]) for i in range(2)], axis=1
        )
        weights = np.concatenate(
            [accessor(attributes[f"WEIGHTS_{i}"]) for i in range(2)], axis=1
        )
        np.testing.assert_array_equal(indices[weights == 0], 0)
        assert indices[1, 0] == 5 and weights[1, 0] == 1
        assert indices[2, 0] == 2 and weights[2, 0] == 1


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_target_limit_rejects_invalid_values_without_creating_output(tmp_path, limit):
    output = tmp_path / "new-directory" / "invalid.glb"
    with pytest.raises(ValueError, match="max_targets"):
        glb.export_glb(_animation(), output, max_targets=limit)
    assert not output.parent.exists()


def test_target_count_size_and_float32_time_guards_are_early(tmp_path, monkeypatch):
    output = tmp_path / "new-directory" / "invalid.glb"
    source = _animation()
    with pytest.raises(ValueError, match="2 morph targets"):
        glb.export_glb(source, output, max_targets=1)
    assert not output.parent.exists()
    collapsed = replace(source, time=np.array([0, 1, 1 + 1e-8]))
    with pytest.raises(ValueError, match="float32 precision"):
        glb.export_glb(collapsed, output)
    assert not output.parent.exists()
    monkeypatch.setattr(glb, "_BINARY_LIMIT", 10)
    with pytest.raises(ValueError, match="4 GiB"):
        glb.export_glb(source, output)
    assert not output.parent.exists()


def test_overflowing_morph_deltas_and_invalid_faces_leave_no_output(tmp_path):
    source = _animation()
    output = tmp_path / "new-directory" / "invalid.glb"
    points = source.meshes[0].points.astype(np.float64)
    points[0] = -3e38
    points[1:] = 3e38
    extreme = replace(source.meshes[0], points=points)
    with pytest.raises(ValueError, match="morph position deltas"):
        glb.export_glb(replace(source, meshes=(extreme,)), output)
    assert not output.parent.exists()
    invalid = replace(source.meshes[0], faces=np.array([[0, 1, 99]]))
    with pytest.raises(ValueError, match="out-of-range"):
        glb.export_glb(replace(source, meshes=(invalid,)), output)
    assert not output.parent.exists()


@pytest.mark.parametrize("method", ["baked", "rigged"])
def test_export_refuses_overwrite_and_wrong_extension(tmp_path, method):
    source = _reference()
    export = glb.export_glb if method == "baked" else glb.export_rigged_glb
    kwargs = {} if method == "baked" else _binding(source)
    output = tmp_path / "existing.glb"
    output.write_bytes(b"keep existing file")
    with pytest.raises(ValueError, match="new .glb"):
        export(source, output, **kwargs)
    assert output.read_bytes() == b"keep existing file"
    with pytest.raises(ValueError, match="new .glb"):
        export(source, tmp_path / "wrong.gltf", **kwargs)


def test_atomic_publication_preserves_a_concurrent_file_and_cleans_temporary(
    tmp_path, monkeypatch
):
    output = tmp_path / "raced.glb"

    def concurrent_file(temporary, destination):
        destination.write_bytes(b"other process")
        raise FileExistsError("concurrent creation")

    monkeypatch.setattr(glb.os, "link", concurrent_file)
    with pytest.raises(ValueError, match="never overwritten"):
        glb.export_glb(_animation(), output)
    assert output.read_bytes() == b"other process"
    assert list(tmp_path.iterdir()) == [output]


def test_failed_write_does_not_leave_a_partial_file(tmp_path, monkeypatch):
    output = tmp_path / "failed.glb"

    def disk_error(_descriptor):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(glb.os, "fsync", disk_error)
    with pytest.raises(OSError, match="disk failure"):
        glb.export_glb(_animation(), output)
    assert list(tmp_path.iterdir()) == []


def test_unsupported_atomic_publication_is_actionable_and_leaves_no_partial(
    tmp_path, monkeypatch
):
    def unsupported(_temporary, _destination):
        raise OSError("hard links unsupported")

    monkeypatch.setattr(glb.os, "link", unsupported)
    with pytest.raises(OSError, match="filesystem that supports hard links"):
        glb.export_glb(_animation(), tmp_path / "unsupported.glb")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("body_names", ("same",) * 6, "unique"),
        ("parents", np.array([-1, 0, 3, 0, 3, 4]), "topologically"),
        ("parents", np.array([-1.0, 0, 1, 0, 3, 4]), "topologically"),
        ("bind_matrices", np.zeros((6, 4, 4)), "rigid"),
        ("weights", {}, "mesh names"),
    ],
)
def test_invalid_reference_binding_is_rejected_before_output(
    tmp_path, field, value, match
):
    source = _reference()
    binding = _binding(source)
    binding[field] = value
    output = tmp_path / "new-directory" / "invalid.glb"
    with pytest.raises(ValueError, match=match):
        glb.export_rigged_glb(source, output, **binding)
    assert not output.parent.exists()


@pytest.mark.parametrize("bad", [-1.0, np.nan, 0.0])
def test_invalid_weights_are_rejected_without_modifying_arrays(tmp_path, bad):
    source = _reference()
    binding = _binding(source)
    binding["weights"]["Skin"][:] = bad
    with pytest.raises(ValueError, match="weights"):
        glb.export_rigged_glb(source, tmp_path / "invalid.glb", **binding)
    assert list(tmp_path.iterdir()) == []


def test_reference_export_requires_one_frame_and_exactly_one_skin(tmp_path):
    source = _animation()
    binding = _binding(source)
    with pytest.raises(ValueError, match="one reference frame"):
        glb.export_rigged_glb(source, tmp_path / "multiple.glb", **binding)
    source = _reference()
    no_skin = replace(source, meshes=(source.meshes[0],))
    with pytest.raises(ValueError, match="one skin"):
        glb.export_rigged_glb(no_skin, tmp_path / "no-skin.glb", **binding)
