# Copyright (c) 2026 Mark Colley.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained glTF 2.0 exports of numeric anatomy, without a render backend.

Animated GLBs bake sampled world-space geometry into standard morph targets.
Rigged GLBs instead bind one reference frame to named MuJoCo bodies and are
compatible with MyoSuite's ``skin_glb`` importer. The two modes deliberately
remain separate: a reference rig does not preserve SOMA's learned deformations.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_soma.soma import SomaAnimation, validate_animation

_TO_GLTF = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
_GLB_LIMIT = 2**32 - 1
# Reserve space for JSON and alignment before constructing potentially large arrays.
_BINARY_LIMIT = _GLB_LIMIT - 64 * 1024**2


def _float32(values: Any, label: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype="<f4", order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} cannot be represented by finite glTF float32 values")
    return result


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = points[faces]
    normals = np.zeros(points.shape, dtype=np.float64)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    length = np.linalg.norm(normals, axis=1)
    nonzero = length > 0
    normals[nonzero] /= length[nonzero, None]
    # A degenerate triangle or unused vertex still needs a finite unit normal.
    normals[~nonzero] = [0, 0, 1]
    return normals


def _target(output: Path | str) -> Path:
    requested = Path(output)
    target = requested.resolve()
    if target.suffix.lower() != ".glb" or requested.is_symlink() or target.exists():
        raise ValueError("Choose a new .glb file; existing files are never overwritten")
    return target


def _size_guard(size: int) -> None:
    if size > _BINARY_LIMIT:
        raise ValueError(
            "GLB would exceed its 4 GiB size limit; export fewer frames/meshes"
        )


class _Document:
    def __init__(self, animation: SomaAnimation, mode: str) -> None:
        self.binary = bytearray()
        self.data: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": "myosuite-soma"},
            "scene": 0,
            "scenes": [{"name": "SOMA", "nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [
                {
                    "name": "Muscle",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.42, 0.11, 0.10, 1],
                        "metallicFactor": 0,
                        "roughnessFactor": 0.7,
                    },
                    "doubleSided": True,
                },
                {
                    "name": "Skin",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.72, 0.62, 0.58, 1],
                        "metallicFactor": 0,
                        "roughnessFactor": 0.8,
                    },
                    "doubleSided": True,
                },
            ],
            "accessors": [],
            "bufferViews": [],
            "extras": {
                "myosuite_soma": {
                    "mode": mode,
                    "coordinate_system": "gltf_y_up",
                    "units": "m",
                    "source_coordinate_system": "mujoco_world",
                    "source_time_start_seconds": float(animation.time[0]),
                    "source_time_end_seconds": float(animation.time[-1]),
                    "source_sample_count": len(animation.time),
                    "source_metadata": animation.metadata,
                    "materials": "uniform; source activation is not material animation",
                }
            },
        }

    def add(
        self,
        values: np.ndarray,
        kind: str,
        component: int,
        *,
        target: int | None = None,
        bounds: bool = False,
    ) -> int:
        values = np.ascontiguousarray(values)
        _size_guard(len(self.binary) + values.nbytes + 3)
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        view = {
            "buffer": 0,
            "byteOffset": len(self.binary),
            "byteLength": values.nbytes,
        }
        if target is not None:
            view["target"] = target
        self.data["bufferViews"].append(view)
        self.binary.extend(values.tobytes())
        accessor: dict[str, Any] = {
            "bufferView": len(self.data["bufferViews"]) - 1,
            "componentType": component,
            "count": len(values),
            "type": kind,
        }
        if bounds:
            rows = values.reshape(len(values), -1)
            accessor["min"] = rows.min(axis=0).tolist()
            accessor["max"] = rows.max(axis=0).tolist()
        self.data["accessors"].append(accessor)
        return len(self.data["accessors"]) - 1

    def primitive(self, mesh: Any) -> tuple[dict[str, Any], np.ndarray]:
        vertices = np.asarray(mesh.points[0], dtype=np.float64)
        normal = _normals(vertices, mesh.faces)
        attributes = {
            "POSITION": self.add(
                _float32(vertices @ _TO_GLTF.T, "positions"),
                "VEC3",
                5126,
                target=34962,
                bounds=True,
            ),
            "NORMAL": self.add(
                _float32(normal @ _TO_GLTF.T, "normals"),
                "VEC3",
                5126,
                target=34962,
            ),
        }
        if mesh.texcoord is not None:
            uv = np.asarray(mesh.texcoord, dtype=np.float64) * [1, -1] + [0, 1]
            attributes["TEXCOORD_0"] = self.add(
                _float32(uv, "texture coordinates"), "VEC2", 5126, target=34962
            )
        primitive = {
            "attributes": attributes,
            "indices": self.add(
                np.asarray(mesh.faces, dtype="<u4").ravel(),
                "SCALAR",
                5125,
                target=34963,
            ),
            "mode": 4,
            "material": 0 if mesh.kind == "muscle" else 1,
        }
        return primitive, normal

    def mesh(self, mesh: Any, primitive: dict[str, Any]) -> int:
        extras: dict[str, Any] = {"kind": mesh.kind}
        if mesh.activation is not None:
            extras["source_activation"] = np.asarray(mesh.activation).tolist()
        self.data["meshes"].append(
            {"name": mesh.name, "primitives": [primitive], "extras": extras}
        )
        return len(self.data["meshes"]) - 1

    def write(self, target: Path) -> Path:
        self.data["buffers"] = [{"byteLength": len(self.binary)}]
        encoded = json.dumps(
            self.data, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        encoded += b" " * (-len(encoded) % 4)
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        total = 12 + 8 + len(encoded) + 8 + len(self.binary)
        if total > _GLB_LIMIT:
            raise ValueError("GLB exceeds its 4 GiB size limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=".soma-glb-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(struct.pack("<4sII", b"glTF", 2, total))
                stream.write(struct.pack("<I4s", len(encoded), b"JSON"))
                stream.write(encoded)
                stream.write(struct.pack("<I4s", len(self.binary), b"BIN\0"))
                stream.write(self.binary)
                stream.flush()
                os.fsync(stream.fileno())
            # A same-directory hard link publishes a complete file atomically,
            # and fails if another process created the target in the meantime.
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise ValueError("Existing .glb files are never overwritten") from error
            except OSError as error:
                raise OSError(
                    "Could not atomically publish the GLB using a same-directory "
                    "hard link. Choose a filesystem that supports hard links "
                    f"and allows file creation: {error}"
                ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target


def export_glb(
    animation: SomaAnimation, output: Path | str, *, max_targets: int = 256
) -> Path:
    """Bake a numeric animation into a self-contained, animated GLB.

    Every source sample is preserved as a standard position/normal morph target
    relative to frame zero. The animation starts at zero seconds, supports
    nonuniform sampling, and interpolates vertices linearly between samples.
    Absolute source times and provenance remain in ``extras.myosuite_soma``.

    ``max_targets`` limits the number of targets per mesh before allocation.
    Long recordings should be sliced into shorter clips: not all viewers support
    hundreds of targets. This GLB is for playback; MyoSuite's reference-skin
    importer does not evaluate its morph animation. Use ``export_rigged_glb``
    to produce a skin bound to live native body transforms instead.
    """
    target = _target(output)
    validate_animation(animation)
    if (
        not isinstance(max_targets, int)
        or isinstance(max_targets, bool)
        or max_targets < 0
    ):
        raise ValueError("max_targets must be a nonnegative integer")
    frame_count = len(animation.time)
    target_count = frame_count - 1
    if target_count > max_targets:
        raise ValueError(
            f"Animation needs {target_count} morph targets; max_targets={max_targets}. "
            "Slice the recording or explicitly raise the limit."
        )
    estimate = frame_count * target_count * 4 + frame_count * 4
    for mesh in animation.meshes:
        estimate += mesh.points.shape[1] * (frame_count * 24 + 8)
        estimate += np.asarray(mesh.faces).size * 4 + frame_count * 8
    _size_guard(estimate)
    times = _float32(
        np.asarray(animation.time, dtype=np.float64) - float(animation.time[0]),
        "relative sample times",
    )
    if len(times) > 1 and np.any(times[1:] <= times[:-1]):
        raise ValueError(
            "Sample times are not strictly increasing at glTF float32 precision"
        )
    document = _Document(animation, "baked_morph_animation")
    if target_count:
        input_accessor = document.add(times, "SCALAR", 5126, bounds=True)
        keys = np.zeros((frame_count, target_count), dtype="<f4")
        keys[np.arange(1, frame_count), np.arange(target_count)] = 1
        output_accessor = document.add(keys.ravel(), "SCALAR", 5126)
        document.data["animations"] = [
            {
                "name": "SOMA",
                "samplers": [
                    {
                        "input": input_accessor,
                        "output": output_accessor,
                        "interpolation": "LINEAR",
                    }
                ],
                "channels": [],
            }
        ]
    for mesh in animation.meshes:
        primitive, base_normal = document.primitive(mesh)
        if target_count:
            targets = []
            base = np.asarray(mesh.points[0], dtype=np.float64)
            for points in mesh.points[1:]:
                displacement = (
                    np.asarray(points, dtype=np.float64) - base
                ) @ _TO_GLTF.T
                normal_delta = (_normals(points, mesh.faces) - base_normal) @ _TO_GLTF.T
                targets.append(
                    {
                        "POSITION": document.add(
                            _float32(displacement, "morph position deltas"),
                            "VEC3",
                            5126,
                            target=34962,
                            bounds=True,
                        ),
                        "NORMAL": document.add(
                            _float32(normal_delta, "morph normal deltas"),
                            "VEC3",
                            5126,
                            target=34962,
                        ),
                    }
                )
            primitive["targets"] = targets
        mesh_index = document.mesh(mesh, primitive)
        if target_count:
            document.data["meshes"][mesh_index]["weights"] = [0.0] * target_count
            document.data["meshes"][mesh_index]["extras"]["targetNames"] = [
                f"sample_{frame:06d}" for frame in range(1, frame_count)
            ]
        node_index = len(document.data["nodes"])
        document.data["nodes"].append({"name": mesh.name, "mesh": mesh_index})
        document.data["scenes"][0]["nodes"].append(node_index)
        if target_count:
            document.data["animations"][0]["channels"].append(
                {"sampler": 0, "target": {"node": node_index, "path": "weights"}}
            )
    return document.write(target)


def _rig(
    body_names: Sequence[str], parents: Any, bind_matrices: Any
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if isinstance(body_names, (str, bytes)):
        raise ValueError("body_names must be a sequence of complete body names")
    names = tuple(body_names)
    if (
        not names
        or len(names) > 65536
        or any(not isinstance(name, str) or not name or "\0" in name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("body_names must contain 1..65536 unique nonempty body names")
    parent = np.asarray(parents)
    if (
        parent.dtype.kind not in "iu"
        or parent.shape != (len(names),)
        or any(p < -1 or p >= i for i, p in enumerate(parent))
    ):
        raise ValueError("parents must be topologically ordered, with -1 at each root")
    world = np.asarray(bind_matrices)
    if world.dtype.kind not in "fiu" or world.shape != (len(names), 4, 4):
        raise ValueError("bind_matrices must be finite rigid transforms shaped (B,4,4)")
    world = world.astype(np.float64)
    rotation = world[:, :3, :3]
    if (
        not np.isfinite(world).all()
        or not np.allclose(world[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-6)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("bind_matrices must be finite proper rigid transforms")
    _float32(world[:, :3, 3], "body bind translations")
    return names, parent.astype(np.int64), world


def export_rigged_glb(
    animation: SomaAnimation,
    output: Path | str,
    *,
    body_names: Sequence[str],
    parents: Any,
    bind_matrices: Any,
    weights: Mapping[str, np.ndarray],
) -> Path:
    """Export one reference anatomy frame bound to native MuJoCo bodies.

    ``bind_matrices`` are world rigid transforms at the target model's ``qpos0``
    in MuJoCo metres/axes. ``parents`` indexes ``body_names`` in parent-first
    order, with -1 for roots. Each named mesh needs a nonnegative ``(V,B)``
    weight matrix with positive row sums. All influences are retained and
    normalized, including influences beyond the first four.

    Exactly one skin mesh is required and exported first so MyoSuite's
    ``read_skin_glb`` selects it. Other meshes remain separate skinned objects
    in the same file; that importer currently loads only the first skin mesh.
    This reference rig uses ordinary linear-blend skinning and does not encode
    the SOMA network or its pose-dependent corrections.
    """
    target = _target(output)
    validate_animation(animation)
    if len(animation.time) != 1:
        raise ValueError("Rigged GLB export requires exactly one reference frame")
    if sum(mesh.kind == "skin" for mesh in animation.meshes) != 1:
        raise ValueError("Rigged GLB export requires exactly one skin mesh")
    names, parent, world = _rig(body_names, parents, bind_matrices)
    if not isinstance(weights, Mapping) or set(weights) != {
        mesh.name for mesh in animation.meshes
    }:
        raise ValueError("weights must match exactly the exported mesh names")
    raw_weights = {}
    estimate = len(names) * 64
    for mesh in animation.meshes:
        values = np.asarray(weights[mesh.name])
        if (
            values.dtype.kind not in "fiu"
            or values.shape != (mesh.points.shape[1], len(names))
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError(
                f"{mesh.name} weights must be finite nonnegative (V,B) values"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            sums = values.sum(axis=1, keepdims=True, dtype=np.float64)
        if not np.isfinite(sums).all() or np.any(sums <= 0):
            raise ValueError(f"{mesh.name} weights need positive finite row sums")
        raw_weights[mesh.name] = (values, sums)
        sets = max(1, (int((values > 0).sum(axis=1).max()) + 3) // 4)
        estimate += mesh.points.shape[1] * (32 + sets * 24)
        estimate += np.asarray(mesh.faces).size * 4
    _size_guard(estimate)
    document = _Document(animation, "native_body_reference_rig")
    document.data["extras"]["myosuite_soma"]["bind_pose"] = "target model qpos0"
    axis = np.eye(4)
    axis[:3, :3] = _TO_GLTF
    converted = axis @ world @ axis.T
    nodes = document.data["nodes"]
    for index, name in enumerate(names):
        local = (
            converted[index]
            if parent[index] == -1
            else np.linalg.inv(converted[parent[index]]) @ converted[index]
        )
        node = {"name": name, "matrix": local.T.ravel().tolist()}
        children = np.flatnonzero(parent == index).tolist()
        if children:
            node["children"] = children
        nodes.append(node)
    nodes.append(
        {"name": "Skeleton", "children": np.flatnonzero(parent == -1).tolist()}
    )
    document.data["scenes"][0]["nodes"].append(len(names))
    inverse = _float32(
        np.linalg.inv(converted).transpose(0, 2, 1).reshape(-1, 16),
        "inverse bind matrices",
    )
    inverse_accessor = document.add(inverse, "MAT4", 5126)
    document.data["skins"] = [
        {
            "name": "NativeMyoSuiteRig",
            "joints": list(range(len(names))),
            "skeleton": len(names),
            "inverseBindMatrices": inverse_accessor,
        }
    ]
    ordered = sorted(animation.meshes, key=lambda mesh: mesh.kind != "skin")
    for mesh in ordered:
        primitive, _ = document.primitive(mesh)
        raw, sums = raw_weights[mesh.name]
        values = np.asarray(raw, dtype=np.float64) / sums
        sets = max(1, (int((values > 0).sum(axis=1).max()) + 3) // 4)
        width = 4 * sets
        order = np.argsort(-values, axis=1, kind="stable")[:, :width]
        ordered_weights = np.take_along_axis(values, order, axis=1)
        padding = width - order.shape[1]
        joints = np.pad(order, ((0, 0), (0, padding))).astype("<u2")
        values32 = _float32(
            np.pad(ordered_weights, ((0, 0), (0, padding))), "skin weights"
        )
        # Unused slots reference joint zero, including weights that underflow
        # to float32 zero. This follows the glTF validator's recommendation.
        joints[values32 == 0] = 0
        for block in range(sets):
            start, end = 4 * block, 4 * block + 4
            primitive["attributes"][f"JOINTS_{block}"] = document.add(
                joints[:, start:end], "VEC4", 5123, target=34962
            )
            primitive["attributes"][f"WEIGHTS_{block}"] = document.add(
                values32[:, start:end], "VEC4", 5126, target=34962
            )
        mesh_index = document.mesh(mesh, primitive)
        node_index = len(nodes)
        nodes.append({"name": mesh.name, "mesh": mesh_index, "skin": 0})
        document.data["scenes"][0]["nodes"].append(node_index)
    return document.write(target)
