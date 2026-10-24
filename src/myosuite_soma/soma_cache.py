# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
# Extracted from the audited MyoSuite SOMA integration; imports adapted.
# Source attribution: THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Chunked numeric SOMA archives with bounded-memory playback."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from collections.abc import Iterator, Sequence
from zipfile import BadZipFile

import numpy as np

from myosuite_soma.soma import (
    AnimatedMesh,
    SomaAnimation,
    load_animation,
    validate_animation,
)
from myosuite_soma.soma_numeric import inspect_numeric_archive

_MAX_CHUNK_BYTES = 512 * 1024**2


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_archive(
    path: str | Path,
    animation: SomaAnimation,
    *,
    chunk_frames: int = 32,
    max_error_m: float | None = None,
) -> dict[str, Any]:
    """Write an immutable chunked archive, optionally quantizing displacements.

    Args:
        path: New directory; existing files/directories are never overwritten.
        animation: Validated world-space geometry and source provenance.
        chunk_frames: Maximum frames decompressed during streaming playback.
        max_error_m: Optional maximum Euclidean vertex error in metres. None
            preserves float32 coordinates exactly. Quantization is opt-in.

    Returns:
        Archive manifest, including measured encoding error and hashes.
    """
    validate_animation(animation)
    if (
        isinstance(chunk_frames, bool)
        or not isinstance(chunk_frames, int)
        or chunk_frames < 1
    ):
        raise ValueError("chunk_frames must be a positive integer")
    if max_error_m is not None and (
        isinstance(max_error_m, bool)
        or not isinstance(max_error_m, Real)
        or not np.isfinite(max_error_m)
        or max_error_m <= 0
    ):
        raise ValueError("max_error_m must be positive and finite")
    if max_error_m is not None:
        max_error_m = float(max_error_m)
    total_vertices = sum(mesh.points.shape[1] for mesh in animation.meshes)
    if (
        min(chunk_frames, len(animation.time)) * total_vertices * 3 * 4
        > _MAX_CHUNK_BYTES
    ):
        raise ValueError("Decoded geometry chunk exceeds 512 MiB; reduce chunk_frames")
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_time = np.asarray(animation.time, np.float64)
    if not np.isfinite(encoded_time).all() or np.any(np.diff(encoded_time) <= 0):
        raise ValueError("Archive timestamps collapse after float64 encoding")
    topology: dict[str, np.ndarray] = {"time": encoded_time}
    specs = []
    for index, mesh in enumerate(animation.meshes):
        key = f"mesh_{index:04d}"
        topology[f"{key}_faces"] = np.asarray(mesh.faces, np.int64)
        if mesh.texcoord is not None:
            topology[f"{key}_texcoord"] = np.asarray(mesh.texcoord)
        if mesh.activation is not None:
            topology[f"{key}_activation"] = np.asarray(mesh.activation)
        specs.append(
            {
                "name": mesh.name,
                "kind": mesh.kind,
                "vertices": mesh.points.shape[1],
                "uv": mesh.texcoord is not None,
                "activation": mesh.activation is not None,
            }
        )
    chunks = []
    error = 0.0
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "archive"
        staging.mkdir()
        np.savez_compressed(staging / "topology.npz", **topology)
        for begin in range(0, len(animation.time), chunk_frames):
            end = min(begin + chunk_frames, len(animation.time))
            arrays: dict[str, np.ndarray] = {}
            for index, mesh in enumerate(animation.meshes):
                key = f"mesh_{index:04d}"
                original = np.asarray(mesh.points[begin:end], np.float64)
                points = original.astype(np.float32)
                if max_error_m is None:
                    if not np.array_equal(original, points.astype(np.float64)):
                        raise ValueError(
                            "Lossless archives require float32-representable coordinates"
                        )
                    arrays[key] = points
                    continue
                # A float32 anchor plus signed int16 offsets avoids quantizing
                # absolute world translation and bounds the full vector error.
                anchor = points[0]
                step = float(max_error_m) / np.sqrt(3.0)
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = np.rint((original - anchor) / step)
                if not np.isfinite(delta).all():
                    raise ValueError("Quantized displacement exceeds int16 range")
                if np.max(np.abs(delta)) > np.iinfo(np.int16).max:
                    raise ValueError(
                        "Quantized displacement exceeds int16 range; reduce chunk_frames or increase max_error_m"
                    )
                arrays[f"{key}_anchor"] = anchor
                arrays[f"{key}_delta"] = delta.astype(np.int16)
                decoded = (anchor.astype(np.float64) + delta * step).astype(np.float32)
                measured = float(
                    np.linalg.norm(decoded.astype(np.float64) - original, axis=-1).max()
                )
                if measured > max_error_m:
                    raise ValueError(
                        "float32 reconstruction exceeds max_error_m at these world coordinates"
                    )
                error = max(error, measured)
            filename = f"chunk_{len(chunks):06d}.npz"
            np.savez_compressed(staging / filename, **arrays)
            chunks.append(
                {"start": begin, "stop": end, "sha256": _sha256(staging / filename)}
            )
        manifest = {
            "format": "soma-chunks",
            "version": 1,
            "frames": len(animation.time),
            "chunk_frames": chunk_frames,
            "max_error_m": max_error_m,
            "measured_max_error_m": error,
            "meshes": specs,
            "metadata": animation.metadata,
            "chunks": chunks,
            "topology_sha256": _sha256(staging / "topology.npz"),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    return manifest


class AnimationArchive:
    """Read one numeric geometry chunk at a time without Torch or SOMA assets."""

    def __init__(self, path: str | Path) -> None:
        """Open and verify an archive's index and topology; chunks verify on read."""
        self.path = Path(path).resolve()
        try:
            manifest_path = self._member("manifest.json")
            if manifest_path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Archive manifest is too large")
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = self.manifest
            if not isinstance(manifest, dict):
                raise ValueError("Archive manifest must be a JSON object")
            if (
                manifest["format"] != "soma-chunks"
                or not _integer(manifest["version"])
                or manifest["version"] != 1
            ):
                raise ValueError("Unsupported SOMA archive")
            frames = manifest["frames"]
            chunk_frames = manifest["chunk_frames"]
            if (
                not _integer(frames)
                or frames < 1
                or not _integer(chunk_frames)
                or chunk_frames < 1
            ):
                raise ValueError("Invalid archive frame counts")
            tolerance = manifest["max_error_m"]
            error = manifest["measured_max_error_m"]
            if tolerance is not None and (
                type(tolerance) not in (float, int)
                or not np.isfinite(tolerance)
                or tolerance <= 0
            ):
                raise ValueError("Invalid archive tolerance")
            if (
                type(error) not in (float, int)
                or not np.isfinite(error)
                or error < 0
                or error > (tolerance or 0)
            ):
                raise ValueError("Invalid archive encoding error")
            topology_path = self._member("topology.npz", manifest["topology_sha256"])
            if not isinstance(manifest["meshes"], list) or not manifest["meshes"]:
                raise ValueError("Archive requires mesh specifications")
            if any(not isinstance(spec, dict) for spec in manifest["meshes"]):
                raise ValueError("Invalid archive mesh specification")
            total_vertices = 0
            for spec in manifest["meshes"]:
                vertices = spec["vertices"]
                if not _integer(vertices) or vertices < 1:
                    raise ValueError("Invalid archive vertex count")
                total_vertices += vertices
            if min(chunk_frames, frames) * total_vertices * 3 * 4 > _MAX_CHUNK_BYTES:
                raise ValueError("Decoded geometry chunk exceeds 512 MiB")
            inspect_numeric_archive(topology_path, max_bytes=_MAX_CHUNK_BYTES)
            with np.load(topology_path, allow_pickle=False) as data:
                self.time = data["time"]
                if self.time.shape != (frames,):
                    raise ValueError("Archive timestamps do not match frame count")
                expected = {"time"}
                meshes = []
                for index, spec in enumerate(manifest["meshes"]):
                    key = f"mesh_{index:04d}"
                    vertices = spec["vertices"]
                    if (
                        not _integer(vertices)
                        or vertices < 1
                        or not isinstance(spec["uv"], bool)
                        or not isinstance(spec["activation"], bool)
                    ):
                        raise ValueError("Invalid archive mesh specification")
                    expected.add(f"{key}_faces")
                    for flag, suffix in (
                        ("uv", "texcoord"),
                        ("activation", "activation"),
                    ):
                        if spec[flag]:
                            expected.add(f"{key}_{suffix}")
                    active = data[f"{key}_activation"] if spec["activation"] else None
                    if active is not None and active.shape != (frames,):
                        raise ValueError("Archive activation frame count mismatch")
                    meshes.append(
                        AnimatedMesh(
                            spec["name"],
                            data[f"{key}_faces"],
                            np.zeros((1, vertices, 3), np.float32),
                            data[f"{key}_texcoord"] if spec["uv"] else None,
                            active,
                            spec["kind"],
                        )
                    )
                if set(data.files) != expected or len(data.files) != len(expected):
                    raise ValueError("Unexpected archive topology arrays")
            self._meshes = tuple(meshes)
            probe = tuple(
                replace(
                    mesh,
                    activation=None if mesh.activation is None else mesh.activation[:1],
                )
                for mesh in meshes
            )
            validate_animation(
                SomaAnimation(self.time[:1], probe, manifest["metadata"])
            )
            if (
                self.time.dtype.kind not in "fiu"
                or not np.isfinite(self.time).all()
                or np.any(np.diff(self.time) <= 0)
            ):
                raise ValueError("Invalid archive timestamps")
            for mesh in meshes:
                if mesh.activation is not None and (
                    mesh.activation.dtype.kind not in "fiu"
                    or not np.isfinite(mesh.activation).all()
                    or np.any(mesh.activation < 0)
                    or np.any(mesh.activation > 1)
                ):
                    raise ValueError("Invalid archive activation")
            cursor = 0
            for chunk in manifest["chunks"]:
                begin, end = chunk["start"], chunk["stop"]
                if (
                    not _integer(begin)
                    or not _integer(end)
                    or begin != cursor
                    or not begin < end <= min(begin + chunk_frames, frames)
                ):
                    raise ValueError("Archive chunks do not cover contiguous frames")
                cursor = end
            if cursor != frames:
                raise ValueError("Archive chunks do not cover all frames")
        except (
            OSError,
            KeyError,
            TypeError,
            AttributeError,
            EOFError,
            BadZipFile,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(f"Invalid SOMA archive: {path}") from exc

    def _member(self, name: str, digest: str | None = None) -> Path:
        member = self.path / name
        if (
            member.is_symlink()
            or member.resolve().parent != self.path
            or not member.is_file()
        ):
            raise ValueError(f"Missing or unsafe archive member: {name}")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or _sha256(member) != digest
        ):
            raise ValueError(f"Archive checksum mismatch: {name}")
        return member

    def iter_chunks(
        self, *, meshes: Sequence[str] | None = None
    ) -> Iterator[SomaAnimation]:
        """Yield validated bounded-size chunks, optionally selecting mesh names."""
        selected = (
            {mesh.name for mesh in self._meshes} if meshes is None else set(meshes)
        )
        if not selected or selected - {mesh.name for mesh in self._meshes}:
            raise ValueError("Select existing archive meshes")
        for index, spec in enumerate(self.manifest["chunks"]):
            begin, end = spec["start"], spec["stop"]
            path = self._member(f"chunk_{index:06d}.npz", spec["sha256"])
            inspect_numeric_archive(path, max_bytes=_MAX_CHUNK_BYTES)
            result = []
            tolerance = self.manifest["max_error_m"]
            try:
                with np.load(path, allow_pickle=False) as data:
                    expected = set()
                    for mesh_index, mesh in enumerate(self._meshes):
                        key = f"mesh_{mesh_index:04d}"
                        expected.update(
                            {key}
                            if tolerance is None
                            else {f"{key}_anchor", f"{key}_delta"}
                        )
                        if mesh.name not in selected:
                            continue
                        if tolerance is None:
                            points = data[key]
                            if points.dtype != np.float32:
                                raise ValueError("Archive points must be float32")
                        else:
                            anchor, delta = data[f"{key}_anchor"], data[f"{key}_delta"]
                            if (
                                anchor.dtype != np.float32
                                or anchor.shape != mesh.points.shape[1:]
                                or delta.dtype != np.int16
                                or delta.shape != (end - begin, mesh.points.shape[1], 3)
                            ):
                                raise ValueError("Invalid quantized archive arrays")
                            points = (
                                anchor.astype(np.float64)
                                + delta.astype(np.float64) * (tolerance / np.sqrt(3.0))
                            ).astype(np.float32)
                        if points.shape != (end - begin, mesh.points.shape[1], 3):
                            raise ValueError("Archive points shape mismatch")
                        result.append(
                            replace(
                                mesh,
                                points=points,
                                activation=None
                                if mesh.activation is None
                                else mesh.activation[begin:end],
                            )
                        )
                    if set(data.files) != expected or len(data.files) != len(expected):
                        raise ValueError("Unexpected archive chunk arrays")
            except (OSError, KeyError, EOFError, BadZipFile) as exc:
                raise ValueError(f"Invalid SOMA archive chunk: {path}") from exc
            metadata = dict(self.manifest["metadata"])
            metadata["archive"] = {
                "format": "soma-chunks",
                "frame_start": begin,
                "frame_stop": end,
                "max_error_m": tolerance,
                "measured_max_error_m": self.manifest["measured_max_error_m"],
            }
            animation = SomaAnimation(self.time[begin:end], tuple(result), metadata)
            validate_animation(animation)
            yield animation

    def load(self, *, max_bytes: int = 2 * 1024**3) -> SomaAnimation:
        """Materialize geometry once; use streaming beyond the allocation limit.

        Args:
            max_bytes: Explicit maximum full geometry allocation in bytes.
                Chunk arrays and topology use additional bounded memory.
        """
        if not _integer(max_bytes) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        required = (
            len(self.time) * sum(mesh.points.shape[1] for mesh in self._meshes) * 3 * 4
        )
        if required > max_bytes:
            raise ValueError(
                "Full geometry exceeds max_bytes; use iter_chunks or explicitly raise the limit"
            )
        meshes = tuple(
            replace(
                mesh,
                points=np.empty((len(self.time), mesh.points.shape[1], 3), np.float32),
            )
            for mesh in self._meshes
        )
        cursor = 0
        for chunk in self.iter_chunks():
            end = cursor + len(chunk.time)
            for source, destination in zip(chunk.meshes, meshes):
                destination.points[cursor:end] = source.points
            cursor = end
        metadata = dict(self.manifest["metadata"])
        metadata["archive"] = {
            "format": "soma-chunks",
            "max_error_m": self.manifest["max_error_m"],
            "measured_max_error_m": self.manifest["measured_max_error_m"],
        }
        animation = SomaAnimation(self.time.copy(), meshes, metadata)
        validate_animation(animation)
        return animation


def main() -> None:
    """Convert an existing NPZ cache to a chunked, verified archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--max-error-mm", type=float)
    args = parser.parse_args()
    manifest = write_archive(
        args.output,
        load_animation(args.input),
        chunk_frames=args.chunk_frames,
        max_error_m=None if args.max_error_mm is None else args.max_error_mm / 1000,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "frames": manifest["frames"],
                "measured_max_error_m": manifest["measured_max_error_m"],
            }
        )
    )


if __name__ == "__main__":
    main()
