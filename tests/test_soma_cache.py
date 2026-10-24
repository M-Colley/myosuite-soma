# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Bounded-error chunked geometry and tamper-resistant streaming contracts."""

import json
import hashlib
import io
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from myosuite_soma.soma import AnimatedMesh, SomaAnimation, validate_animation
from myosuite_soma.soma_cache import AnimationArchive, write_archive


def _animation() -> SomaAnimation:
    time = np.arange(19) / 25
    rng = np.random.default_rng(7)
    points = rng.normal(0, 0.015, (len(time), 4, 3)).astype(np.float32) + np.array(
        [1, 2, 3], np.float32
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    return SomaAnimation(
        time,
        (
            AnimatedMesh("muscle", faces, points, activation=time),
            AnimatedMesh("skin", faces, points + 0.03, kind="skin"),
        ),
        {"coordinate_system": "mujoco_world", "units": "m", "source": "test"},
    )


@pytest.mark.parametrize("tolerance", [None, 0.0001])
def test_archive_streaming_and_full_load(
    tmp_path: Path, tolerance: float | None
) -> None:
    original = _animation()
    manifest = write_archive(
        tmp_path / "cache", original, chunk_frames=4, max_error_m=tolerance
    )
    archive = AnimationArchive(tmp_path / "cache")
    chunks = list(archive.iter_chunks(meshes=["skin"]))
    assert [len(chunk.time) for chunk in chunks] == [4, 4, 4, 4, 3]
    assert all(len(chunk.meshes) == 1 for chunk in chunks)
    decoded = archive.load()
    np.testing.assert_array_equal(decoded.time, original.time)
    for source, target in zip(original.meshes, decoded.meshes):
        error = np.linalg.norm(
            source.points.astype(float) - target.points, axis=-1
        ).max()
        assert error <= (tolerance or 0)
        np.testing.assert_array_equal(source.faces, target.faces)
        if source.activation is not None:
            np.testing.assert_array_equal(source.activation, target.activation)
    assert manifest["measured_max_error_m"] <= (tolerance or 0)


def test_archive_rejects_overwrite_and_out_of_range_quantization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache"
    path.mkdir()
    with pytest.raises(FileExistsError):
        write_archive(path, _animation())
    with pytest.raises(ValueError, match="int16 range"):
        write_archive(tmp_path / "too-precise", _animation(), max_error_m=1e-9)
    assert not (tmp_path / "too-precise").exists()


def test_archive_detects_corrupt_geometry_before_return(tmp_path: Path) -> None:
    write_archive(tmp_path / "cache", _animation())
    archive = AnimationArchive(tmp_path / "cache")
    chunk = tmp_path / "cache" / "chunk_000000.npz"
    chunk.write_bytes(chunk.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        next(archive.iter_chunks())


def test_archive_rejects_missing_frames_and_unknown_mesh(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    write_archive(root, _animation(), chunk_frames=4)
    with pytest.raises(ValueError, match="existing archive meshes"):
        next(AnimationArchive(root).iter_chunks(meshes=["absent"]))
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["chunks"][1]["start"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contiguous"):
        AnimationArchive(root)


@pytest.mark.parametrize("chunk_frames", [0, 1.5, True])
def test_archive_rejects_invalid_chunk_size(tmp_path: Path, chunk_frames: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        write_archive(tmp_path / "cache", _animation(), chunk_frames=chunk_frames)


def test_error_bound_includes_float64_to_float32_loss(tmp_path: Path) -> None:
    original = _animation()
    original = replace(
        original,
        meshes=tuple(
            replace(
                mesh, points=np.full(mesh.points.shape, 1.00000001, dtype=np.float64)
            )
            for mesh in original.meshes
        ),
    )
    with pytest.raises(ValueError, match="float32-representable"):
        write_archive(tmp_path / "exact", original)
    with pytest.raises(ValueError, match="float32 reconstruction"):
        write_archive(tmp_path / "too-small", original, max_error_m=1e-9)
    manifest = write_archive(tmp_path / "bounded", original, max_error_m=2e-8)
    decoded = AnimationArchive(tmp_path / "bounded").load()
    error = max(
        np.linalg.norm(source.points - target.points.astype(np.float64), axis=-1).max()
        for source, target in zip(original.meshes, decoded.meshes)
    )
    assert error > 1e-8
    assert error <= 2e-8
    assert error == manifest["measured_max_error_m"]


def _rehash_chunk(root: Path) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["chunks"][0]["sha256"] = hashlib.sha256(
        (root / "chunk_000000.npz").read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(manifest))


def test_forged_npy_shape_rejected_before_allocation(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    write_archive(root, _animation())
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 4, 3)}
    )
    with ZipFile(root / "chunk_000000.npz", "w") as archive:
        archive.writestr("mesh_0000.npy", header.getvalue())
    _rehash_chunk(root)
    with pytest.raises(ValueError, match="shape or dtype"):
        next(AnimationArchive(root).iter_chunks())


def test_chunk_object_array_rejected_without_pickle_execution(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    write_archive(root, _animation())
    marker = tmp_path / "pickle-executed"

    class Payload:
        def __reduce__(self):
            return eval, (
                f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",
            )

    np.savez(root / "chunk_000000.npz", mesh_0000=np.array([Payload()], dtype=object))
    _rehash_chunk(root)
    with pytest.raises(ValueError, match="shape or dtype"):
        next(AnimationArchive(root).iter_chunks())
    assert not marker.exists()


def test_manifest_vertex_allocation_limit_and_full_load_limit(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    write_archive(root, _animation())
    archive = AnimationArchive(root)
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        archive.load(max_bytes=1)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["meshes"][0]["vertices"] = 10**12
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exceeds 512 MiB"):
        AnimationArchive(root)


def test_conditioned_cache_checks_all_features_without_colors(tmp_path: Path) -> None:
    from myosuite_soma.soma_conditioning import physiology_digest

    original = _animation()
    names = ("native_muscle",)
    physiology = {
        "activation": np.linspace(0, 1, len(original.time))[:, None],
        "actuator_force": np.linspace(0, -100, len(original.time))[:, None],
    }
    conditioning = {
        "required_features": ["activation", "actuator_force"],
        "source_actuator_names": list(names),
        "feature_units": {
            "activation": "dimensionless",
            "actuator_force": "N_signed_transmission",
        },
        "source_physiology_sha256": physiology_digest(
            names, physiology, list(physiology)
        ),
    }
    original = replace(
        original,
        meshes=tuple(replace(mesh, activation=None) for mesh in original.meshes),
        metadata=dict(original.metadata, conditioning=conditioning),
    )
    write_archive(tmp_path / "cache", original)
    restored = AnimationArchive(tmp_path / "cache").load()
    validate_animation(
        restored,
        actuator_names=names,
        activation=physiology["activation"],
        physiology=physiology,
    )
    with pytest.raises(ValueError, match="physiology does not match"):
        validate_animation(
            restored,
            actuator_names=names,
            activation=physiology["activation"],
            physiology=dict(
                physiology, actuator_force=physiology["actuator_force"] + 1
            ),
        )
    with pytest.raises(ValueError, match="Missing conditioning physiology"):
        validate_animation(
            restored, actuator_names=names, activation=physiology["activation"]
        )


def test_exact_uv_roundtrip_and_timestamp_encoding_guard(tmp_path: Path) -> None:
    original = _animation()
    uv = np.arange(8, dtype=np.float64).reshape(4, 2) * 0.1234567890123
    original = replace(
        original, meshes=(replace(original.meshes[0], texcoord=uv), original.meshes[1])
    )
    write_archive(tmp_path / "cache", original)
    restored = AnimationArchive(tmp_path / "cache").load()
    assert restored.meshes[0].texcoord.dtype == np.float64
    np.testing.assert_array_equal(restored.meshes[0].texcoord, uv)
    integer_times = np.arange(len(original.time), dtype=np.uint64) + 2**60
    with pytest.raises(ValueError, match="timestamps collapse"):
        write_archive(
            tmp_path / "collapsed-time", replace(original, time=integer_times)
        )
    assert not (tmp_path / "collapsed-time").exists()
