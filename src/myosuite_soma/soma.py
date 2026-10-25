# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
# Extracted from the audited MyoSuite SOMA integration; imports adapted.
# Source attribution: THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Retarget MuJoCo poses to SOMA and exchange validated, portable mesh caches."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Sequence
from zipfile import BadZipFile

import mujoco
import numpy as np

if TYPE_CHECKING:
    from myosuite_soma.soma_geometry import GeometryConfig

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_CACHE_VERSION = 1


def _finite(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _rotation(value: Any, name: str) -> np.ndarray:
    matrix = _finite(value, name).astype(np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a proper 3x3 rotation matrix")
    return matrix


def _safe_name(name: str) -> bool:
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    return (
        isinstance(name, str)
        and _NAME.fullmatch(name) is not None
        and ".." not in name
        and not name.endswith(".")
        and name.split(".")[0].upper() not in reserved
    )


@dataclass(frozen=True)
class BodyMapping:
    """A source body pair and calibration for one SOMA local joint rotation.

    ``basis`` maps source-parent axes to SOMA-parent axes. At ``rest_qpos``,
    the predicted local rotation is ``rest_rotation`` (identity by default).
    """

    body: str
    parent_body: str = "world"
    basis: np.ndarray = field(default_factory=lambda: np.eye(3))
    rest_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass(frozen=True)
class SomaConfig:
    """Explicit SOMA assets, retarget calibration and world placement.

    JSON paths resolve relative to the JSON file. ``body_map`` keys are exact
    SOMA joint names, ``muscle_map`` keys are exact exported mesh names, and its
    values map exact actuator names to nonnegative averaging weights.
    ``root_offset`` is a displacement in source root-body local metres, rotated
    each frame; ``output_translation`` is a constant MuJoCo world displacement.
    """

    data_root: Path
    subject: str
    checkpoint: Path
    preset: str | None = None
    body_map: dict[str, BodyMapping] = field(default_factory=dict)
    basis: np.ndarray = field(default_factory=lambda: np.eye(3))
    rest_qpos: np.ndarray | None = None
    output_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    output_translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    anchor_body: str | None = None
    anchor_joint: str | None = None
    root_body: str | None = None
    muscle_map: dict[str, dict[str, float]] = field(default_factory=dict)
    individual_muscles: tuple[str, ...] | None = None
    device: str = "auto"
    batch_size: int = 32
    environment: str | None = None
    geometry: GeometryConfig | None = None
    conditioning_model: Path | None = None
    allow_synthetic_conditioning: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> SomaConfig:
        """Read a config without executing code or importing an upstream repo.

        Args:
            path: JSON configuration file.

        Returns:
            A typed configuration with absolute asset paths.

        Raises:
            ValueError: Unknown fields or invalid calibration/settings.
        """
        path = Path(path).resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("SOMA config must be a JSON object")
        unknown = raw.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown SOMA config fields: {sorted(unknown)}")
        for key in ("data_root", "subject", "checkpoint"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ValueError(f"SOMA config requires a nonempty {key}")
        for key in ("data_root", "checkpoint"):
            raw[key] = (path.parent / raw[key]).resolve()
        if raw.get("conditioning_model") is not None:
            if (
                not isinstance(raw["conditioning_model"], str)
                or not raw["conditioning_model"]
            ):
                raise ValueError("conditioning_model must be a nonempty path")
            raw["conditioning_model"] = (
                path.parent / raw["conditioning_model"]
            ).resolve()
        if raw.get("geometry") is not None:
            from myosuite_soma.soma_geometry import GeometryConfig

            raw["geometry"] = GeometryConfig.from_dict(raw["geometry"])
        mapping = raw.get("body_map", {})
        if not isinstance(mapping, dict):
            raise ValueError("body_map must be an object keyed by SOMA joint names")
        raw["body_map"] = {}
        for name, spec in mapping.items():
            if not isinstance(spec, dict) or "body" not in spec:
                raise ValueError(f"body_map.{name} requires a body")
            if spec.keys() - BodyMapping.__dataclass_fields__.keys():
                raise ValueError(f"Unknown body mapping fields for {name}")
            spec = dict(spec)
            for key in ("basis", "rest_rotation"):
                if key in spec:
                    spec[key] = _rotation(spec[key], f"body_map.{name}.{key}")
            raw["body_map"][name] = BodyMapping(**spec)
        for key in ("basis", "output_rotation"):
            if key in raw:
                raw[key] = _rotation(raw[key], key)
        for key in ("output_translation", "root_offset"):
            if key in raw:
                raw[key] = _finite(raw[key], key).astype(float)
        if "rest_qpos" in raw and raw["rest_qpos"] is not None:
            raw["rest_qpos"] = _finite(raw["rest_qpos"], "rest_qpos").astype(float)
        if raw.get("individual_muscles") is not None:
            if not isinstance(raw["individual_muscles"], list):
                raise ValueError("individual_muscles must be a list of mesh names")
            raw["individual_muscles"] = tuple(raw["individual_muscles"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject invalid configuration before loading any optional dependencies."""
        if not _safe_name(self.subject):
            raise ValueError("subject must be a safe dataset directory name")
        if not isinstance(self.allow_synthetic_conditioning, bool):
            raise ValueError("allow_synthetic_conditioning must be boolean")
        if self.conditioning_model is not None and not isinstance(
            self.conditioning_model, Path
        ):
            raise ValueError("conditioning_model must be a Path")
        if self.geometry is not None:
            from myosuite_soma.soma_geometry import GeometryConfig

            if not isinstance(self.geometry, GeometryConfig):
                raise ValueError("geometry must be a GeometryConfig")
            self.geometry.validate()
        if self.preset not in (None, "myoElbow1D6"):
            raise ValueError("Supported SOMA preset: myoElbow1D6")
        _rotation(self.basis, "basis")
        _rotation(self.output_rotation, "output_rotation")
        if _finite(self.output_translation, "output_translation").shape != (3,):
            raise ValueError("output_translation must have shape (3,)")
        offset = _finite(self.root_offset, "root_offset")
        if offset.shape != (3,):
            raise ValueError("root_offset must have shape (3,)")
        if np.any(offset) and self.root_body is None and self.preset is None:
            raise ValueError("A nonzero root_offset requires root_body or a preset")
        if (
            not isinstance(self.scale, Real)
            or isinstance(self.scale, bool)
            or not np.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("scale must be positive and finite")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if self.device not in ("auto", "cpu", "cuda") and not (
            isinstance(self.device, str) and re.fullmatch(r"cuda:\d+", self.device)
        ):
            raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
        if (self.anchor_body is None) != (self.anchor_joint is None):
            raise ValueError("anchor_body and anchor_joint must be specified together")
        for key in ("anchor_body", "anchor_joint", "root_body", "environment"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{key} must be a nonempty name")
        if (
            self.rest_qpos is not None
            and _finite(self.rest_qpos, "rest_qpos").ndim != 1
        ):
            raise ValueError("rest_qpos must be a vector")
        if not isinstance(self.body_map, dict):
            raise ValueError("body_map must contain named BodyMapping values")
        for name, spec in self.body_map.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(spec, BodyMapping)
            ):
                raise ValueError("body_map must contain named BodyMapping values")
            if not isinstance(spec.body, str) or not isinstance(spec.parent_body, str):
                raise ValueError("Mapped source bodies must be names")
            _rotation(spec.basis, f"body_map.{name}.basis")
            _rotation(spec.rest_rotation, f"body_map.{name}.rest_rotation")
        if self.individual_muscles is not None and (
            not self.individual_muscles
            or not all(_safe_name(n) for n in self.individual_muscles)
            or len(set(self.individual_muscles)) != len(self.individual_muscles)
        ):
            raise ValueError("individual_muscles must contain unique safe mesh names")
        if not isinstance(self.muscle_map, dict):
            raise ValueError("muscle_map must be an object")
        for mesh, weights in self.muscle_map.items():
            if not _safe_name(mesh) or not isinstance(weights, dict) or not weights:
                raise ValueError(
                    "muscle_map requires safe mesh names and actuator weights"
                )
            if not all(isinstance(n, str) and n for n in weights):
                raise ValueError(f"Invalid actuator names for {mesh}")
            values = _finite(list(weights.values()), f"muscle_map.{mesh}")
            if np.any(values < 0) or values.max() <= 0:
                raise ValueError(
                    f"Muscle weights for {mesh} must be nonnegative with positive sum"
                )


@dataclass(frozen=True)
class AnimatedMesh:
    """One fixed-topology muscle/skin mesh sampled at animation times."""

    name: str
    faces: np.ndarray
    points: np.ndarray
    texcoord: np.ndarray | None = None
    activation: np.ndarray | None = None
    kind: str = "muscle"


@dataclass(frozen=True)
class SomaAnimation:
    """Meshes in MuJoCo world metres, timestamps in seconds, JSON provenance."""

    time: np.ndarray
    meshes: tuple[AnimatedMesh, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def qpos_digest(qpos: np.ndarray) -> str:
    """Hash a source trajectory in a platform-independent float64 encoding."""
    array = _finite(qpos, "qpos")
    digest = hashlib.sha256(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def model_digest(model: mujoco.MjModel) -> str:
    """Hash the kinematic model, including dimensions, names and calibration."""
    digest = hashlib.sha256(bytes(model.names))
    for value in (
        [model.nq, model.nv, model.nu, model.na, model.nbody, model.njnt, model.nmocap],
        model.body_parentid,
        model.body_jntadr,
        model.body_jntnum,
        model.body_mocapid,
        model.body_pos,
        model.body_quat,
        model.jnt_type,
        model.jnt_bodyid,
        model.jnt_qposadr,
        model.jnt_dofadr,
        model.jnt_pos,
        model.jnt_axis,
        model.qpos0,
    ):
        digest.update(qpos_digest(np.asarray(value)).encode("ascii"))
    return digest.hexdigest()


def activation_digest(actuator_names: Sequence[str], activation: np.ndarray) -> str:
    """Hash named activation columns so cached colors cannot become stale."""
    digest = hashlib.sha256(
        json.dumps(list(actuator_names), ensure_ascii=True).encode("utf-8")
    )
    digest.update(qpos_digest(activation).encode("ascii"))
    return digest.hexdigest()


def validate_animation(
    animation: SomaAnimation,
    *,
    time: np.ndarray | None = None,
    qpos: np.ndarray | None = None,
    model: mujoco.MjModel | None = None,
    actuator_names: Sequence[str] | None = None,
    activation: np.ndarray | None = None,
    physiology: dict[str, np.ndarray] | None = None,
) -> None:
    """Validate mesh/cache data and optionally its source trajectory/model.

    Args:
        animation: In-memory animation or a loaded cache.
        time: Expected frame times, when matching a recorded rollout.
        qpos: Expected source positions, checked against the recorded digest.
        model: Expected source MuJoCo kinematic model.
        actuator_names: Names of expected activation columns.
        activation: Expected activation matrix, when checking cached colors.
        physiology: Named native features required by optional conditioning.

    Raises:
        ValueError: Invalid geometry, time, metadata or mismatched provenance.
    """
    times = _finite(animation.time, "time")
    if times.ndim != 1 or not len(times) or np.any(times[1:] <= times[:-1]):
        raise ValueError("time must be a nonempty strictly increasing vector")
    if not isinstance(animation.metadata, dict):
        raise ValueError("metadata must be a JSON object")
    try:
        json.dumps(animation.metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON values") from exc
    if animation.metadata.get("coordinate_system", "mujoco_world") != "mujoco_world":
        raise ValueError("SOMA cache coordinate_system must be mujoco_world")
    if animation.metadata.get("units", "m") != "m":
        raise ValueError("SOMA cache units must be m")
    if not animation.meshes:
        raise ValueError("SOMA animation must contain at least one mesh")
    names: set[str] = set()
    for mesh in animation.meshes:
        if not _safe_name(mesh.name) or mesh.name in names:
            raise ValueError(
                f"Mesh names must be unique and filename-safe: {mesh.name!r}"
            )
        names.add(mesh.name)
        if mesh.kind not in ("muscle", "skin"):
            raise ValueError(f"Invalid mesh kind: {mesh.kind}")
        points = _finite(mesh.points, f"{mesh.name}.points")
        limit = np.finfo(np.float32).max
        if np.any(points > limit) or np.any(points < -limit):
            raise ValueError(
                f"{mesh.name}.points exceed finite USD float32 coordinates"
            )
        if (
            points.ndim != 3
            or points.shape[0] != len(times)
            or points.shape[2] != 3
            or not points.shape[1]
        ):
            raise ValueError(f"{mesh.name}.points must have shape (T, V, 3)")
        faces = np.asarray(mesh.faces)
        if (
            faces.dtype.kind not in "iu"
            or faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
        ):
            raise ValueError(
                f"{mesh.name}.faces must contain fixed integer triangles (F, 3)"
            )
        if np.any(faces < 0) or np.any(faces >= points.shape[1]):
            raise ValueError(f"{mesh.name}.faces contains out-of-range vertex indices")
        if mesh.texcoord is not None:
            uv = _finite(mesh.texcoord, f"{mesh.name}.texcoord")
            if np.any(uv > limit) or np.any(uv < -limit):
                raise ValueError(
                    f"{mesh.name}.texcoord exceed finite USD float32 values"
                )
            if uv.shape != (points.shape[1], 2):
                raise ValueError(f"{mesh.name}.texcoord must have shape (V, 2)")
        if mesh.activation is not None:
            mesh_activation = _finite(mesh.activation, f"{mesh.name}.activation")
            if (
                mesh_activation.shape != times.shape
                or np.any(mesh_activation < 0)
                or np.any(mesh_activation > 1)
            ):
                raise ValueError(
                    f"{mesh.name}.activation must have shape (T,) in [0, 1]"
                )
    if time is not None and not np.array_equal(times, np.asarray(time)):
        raise ValueError("SOMA animation times do not match the recorded rollout")
    if qpos is not None and animation.metadata.get("source_qpos_sha256") != qpos_digest(
        qpos
    ):
        raise ValueError(
            "SOMA animation source qpos does not match the recorded rollout"
        )
    if model is not None and animation.metadata.get(
        "source_model_sha256"
    ) != model_digest(model):
        raise ValueError(
            "SOMA animation source model does not match the recorded rollout"
        )
    if animation.metadata.get("geometry") is not None:
        from myosuite_soma.soma_geometry import validate_geometry_provenance

        validate_geometry_provenance(animation.metadata["geometry"], model)
    if animation.metadata.get("conditioning") is not None and any(
        value is not None
        for value in (time, qpos, model, actuator_names, activation, physiology)
    ):
        from myosuite_soma.soma_conditioning import validate_conditioning_source

        features = {} if physiology is None else dict(physiology)
        if activation is not None:
            if "activation" in features and not np.array_equal(
                features["activation"], activation
            ):
                raise ValueError("Conflicting source activation and physiology values")
            features["activation"] = activation
        validate_conditioning_source(
            animation.metadata["conditioning"], features, actuator_names
        )
    if (actuator_names is None) != (activation is None):
        raise ValueError("actuator_names and activation must be supplied together")
    if activation is not None and any(
        mesh.activation is not None for mesh in animation.meshes
    ):
        if animation.metadata.get("source_activation_sha256") != activation_digest(
            actuator_names, activation
        ):
            raise ValueError(
                "SOMA animation source activation does not match the recorded rollout"
            )
        mapping = animation.metadata.get("muscle_map")
        if not isinstance(mapping, dict):
            raise ValueError("Colored SOMA caches require an explicit muscle_map")
        colored = {
            mesh.name: mesh for mesh in animation.meshes if mesh.activation is not None
        }
        if mapping.keys() != colored.keys() or any(
            mesh.kind != "muscle" for mesh in colored.values()
        ):
            raise ValueError("muscle_map must match exactly the colored muscle meshes")
        expected = map_activations(mapping, actuator_names, activation)
        if np.shape(activation)[0] != len(times):
            raise ValueError("activation must have one row per animation frame")
        for name, mesh in colored.items():
            if not np.array_equal(
                np.asarray(mesh.activation, dtype=np.float32),
                np.asarray(expected[name], dtype=np.float32),
            ):
                raise ValueError(
                    f"{name}.activation does not match its mapped activation"
                )


def save_animation(path: str | Path, animation: SomaAnimation) -> None:
    """Write a validated NPZ cache containing only numeric arrays and JSON.

    Args:
        path: Exact output path (no extension is added).
        animation: Fixed-topology world-space mesh samples.
    """
    validate_animation(animation)
    arrays: dict[str, np.ndarray] = {
        "time": np.asarray(animation.time, dtype=np.float64)
    }
    meshes = []
    for index, mesh in enumerate(animation.meshes):
        prefix = f"mesh_{index:04d}"
        arrays[f"{prefix}_points"] = np.asarray(mesh.points)
        arrays[f"{prefix}_faces"] = np.asarray(mesh.faces, dtype=np.int64)
        meshes.append(
            {
                "name": mesh.name,
                "kind": mesh.kind,
                "uv": mesh.texcoord is not None,
                "activation": mesh.activation is not None,
            }
        )
        if mesh.texcoord is not None:
            arrays[f"{prefix}_texcoord"] = np.asarray(mesh.texcoord)
        if mesh.activation is not None:
            arrays[f"{prefix}_activation"] = np.asarray(mesh.activation)
    manifest = {
        "version": _CACHE_VERSION,
        "meshes": meshes,
        "metadata": animation.metadata,
    }
    arrays["manifest"] = np.asarray(json.dumps(manifest, allow_nan=False))
    with Path(path).open("wb") as file:
        np.savez_compressed(file, **arrays)


def load_animation(path: str | Path, *, max_bytes: int = 2 * 1024**3) -> SomaAnimation:
    """Load and validate a portable cache; pickle/object arrays are prohibited.

    Args:
        path: Cache written by :func:`save_animation`.
        max_bytes: Maximum total uncompressed NPZ bytes, checked before NumPy
            reads array payloads. Larger trusted caches need an explicit limit.

    Returns:
        The validated animation, independent of Torch or SOMA asset files.

    Raises:
        ValueError: Unsupported/truncated/malformed cache or invalid animation.
    """
    from myosuite_soma.soma_numeric import inspect_numeric_archive

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    try:
        inspect_numeric_archive(path, max_bytes=max_bytes, allow_text=True)
        with np.load(Path(path), allow_pickle=False) as cache:
            manifest_array = cache["manifest"]
            if manifest_array.shape != () or manifest_array.dtype.kind != "U":
                raise ValueError("SOMA manifest must be a JSON string")
            manifest = json.loads(str(manifest_array.item()))
            if manifest.get("version") != _CACHE_VERSION:
                raise ValueError("Unsupported SOMA animation cache version")
            specs = manifest["meshes"]
            if not isinstance(specs, list):
                raise ValueError("Invalid SOMA mesh manifest")
            expected = {"manifest", "time"}
            meshes = []
            for index, spec in enumerate(specs):
                prefix = f"mesh_{index:04d}"
                expected.update((f"{prefix}_faces", f"{prefix}_points"))
                uv = spec.get("uv", False)
                active = spec.get("activation", False)
                if not isinstance(uv, bool) or not isinstance(active, bool):
                    raise ValueError("Invalid optional mesh array flags")
                for present, suffix in ((uv, "texcoord"), (active, "activation")):
                    if present:
                        expected.add(f"{prefix}_{suffix}")
                meshes.append(
                    AnimatedMesh(
                        spec["name"],
                        cache[f"{prefix}_faces"],
                        cache[f"{prefix}_points"],
                        cache[f"{prefix}_texcoord"] if uv else None,
                        cache[f"{prefix}_activation"] if active else None,
                        spec["kind"],
                    )
                )
            if set(cache.files) != expected or len(cache.files) != len(expected):
                raise ValueError("Unexpected or missing arrays in SOMA cache")
            animation = SomaAnimation(
                cache["time"], tuple(meshes), manifest["metadata"]
            )
    except (
        KeyError,
        TypeError,
        AttributeError,
        OSError,
        EOFError,
        BadZipFile,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(f"Invalid SOMA animation cache: {path}: {exc}") from exc
    validate_animation(animation)
    return animation


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        raise ValueError(f"SOMA retarget source body is missing: {name}")
    return body


def _require_qpos_body(model: mujoco.MjModel, body: int, relative_to: int = 0) -> None:
    source = body
    while body != relative_to:
        if model.body_mocapid[body] >= 0:
            raise ValueError(
                f"SOMA source body {model.body(source).name} depends on mocap state, "
                "which is absent from qpos"
            )
        body = int(model.body_parentid[body])


def _source_poses(model: mujoco.MjModel, qpos: np.ndarray, name: str) -> np.ndarray:
    poses = _finite(qpos, name)
    if poses.ndim != 2 or poses.shape[1] != model.nq or not len(poses):
        raise ValueError(f"{name} must have shape (T, {model.nq})")
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind not in (
            int(mujoco.mjtJoint.mjJNT_FREE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        ):
            continue
        start = int(model.jnt_qposadr[joint])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            start += 3
        quat = poses[:, start : start + 4]
        if np.any(np.abs(quat) > 1.00001) or not np.allclose(
            np.sum(quat * quat, axis=1), 1, rtol=1e-5, atol=1e-8
        ):
            raise ValueError(
                f"{name} must contain a unit quaternion for joint {model.joint(joint).name}"
            )
    return poses


def _mappings(config: SomaConfig) -> dict[str, BodyMapping]:
    mappings = {}
    if config.preset == "myoElbow1D6":
        mappings["RightForeArm"] = BodyMapping("r_ulna_radius_hand", "r_humerus")
    mappings.update(config.body_map)
    if not mappings:
        raise ValueError("SOMA retargeting requires a preset or explicit body_map")
    return mappings


def _reference(config: SomaConfig, model: mujoco.MjModel) -> mujoco.MjData:
    reference = mujoco.MjData(model)
    rest = model.qpos0 if config.rest_qpos is None else config.rest_qpos
    if np.shape(rest) != (model.nq,):
        raise ValueError(f"rest_qpos must have shape ({model.nq},)")
    _source_poses(model, np.asarray(rest)[None], "rest_qpos")
    reference.qpos[:] = rest
    mujoco.mj_forward(model, reference)
    return reference


def retarget_rotations(
    config: SomaConfig,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_names: Sequence[str],
    parents: np.ndarray,
) -> np.ndarray:
    """Retarget calibrated local rotations from FK using isolated scratch data.

    Args:
        config: Source body pairs, rest pose and rotation bases.
        model: Source MuJoCo model; no model/data state is written.
        qpos: Recorded positions, shape ``(T, model.nq)``.
        joint_names: SOMA joint ordering.
        parents: SOMA parent indices, for checking the target hierarchy.

    Returns:
        Local SOMA rotations, shape ``(T, J, 3, 3)``. Unmapped joints are identity.
    """
    config.validate()
    poses = _source_poses(model, qpos, "qpos")
    names = tuple(joint_names)
    parents = np.asarray(parents)
    if not names or len(set(names)) != len(names) or parents.shape != (len(names),):
        raise ValueError("SOMA joint names/parents must have matching unique entries")
    if (
        parents.dtype.kind not in "iu"
        or parents[0] != -1
        or any(p < 0 or p >= j for j, p in enumerate(parents[1:], 1))
    ):
        raise ValueError("SOMA parents must be root-first with root parent -1")
    mappings = _mappings(config)
    unknown = mappings.keys() - set(names)
    if unknown:
        raise ValueError(f"SOMA target joints are missing: {sorted(unknown)}")
    reference = _reference(config, model)
    rotations = np.broadcast_to(np.eye(3), (len(poses), len(names), 3, 3)).copy()
    bindings = []
    for name, spec in mappings.items():
        body, parent = _body_id(model, spec.body), _body_id(model, spec.parent_body)
        ancestor = int(model.body_parentid[body])
        while ancestor != parent and ancestor != 0:
            ancestor = int(model.body_parentid[ancestor])
        if body == parent or ancestor != parent:
            raise ValueError(f"{spec.parent_body} must be an ancestor of {spec.body}")
        _require_qpos_body(model, body, parent)
        rest = reference.xmat[parent].reshape(3, 3).T @ reference.xmat[body].reshape(
            3, 3
        )
        basis = config.basis @ spec.basis
        bindings.append(
            (names.index(name), body, parent, rest, basis, spec.rest_rotation)
        )
    scratch = mujoco.MjData(model)
    for frame, pose in enumerate(poses):
        scratch.qpos[:] = pose
        mujoco.mj_forward(model, scratch)
        for joint, body, parent, rest, basis, target_rest in bindings:
            relative = scratch.xmat[parent].reshape(3, 3).T @ scratch.xmat[
                body
            ].reshape(3, 3)
            rotations[frame, joint] = basis @ relative @ rest.T @ basis.T @ target_rest
    return rotations


def map_activations(
    muscle_map: dict[str, dict[str, float]],
    actuator_names: Sequence[str],
    activation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute explicitly weighted visual activations by exact actuator name.

    Args:
        muscle_map: Mesh name to actuator-name/weight mappings.
        actuator_names: Columns of the recorded activation matrix.
        activation: ``(T, A)`` muscle states in ``[0, 1]`` (not excitations).

    Returns:
        One weighted average time series per configured mesh.
    """
    if not isinstance(muscle_map, dict):
        raise ValueError("muscle_map must be an object")
    names = tuple(actuator_names)
    values = _finite(activation, "activation")
    if (
        not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
        or values.ndim != 2
        or values.shape[1] != len(names)
    ):
        raise ValueError("activation columns must match unique actuator_names")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Recorded activation must be in [0, 1]")
    result = {}
    for mesh, weights in muscle_map.items():
        if not _safe_name(mesh) or not isinstance(weights, dict) or not weights:
            raise ValueError("muscle_map requires safe mesh names and actuator weights")
        if not all(isinstance(name, str) and name for name in weights):
            raise ValueError(f"Invalid actuator names for {mesh}")
        unknown = weights.keys() - set(names)
        if unknown:
            raise ValueError(f"Actuators for {mesh} are missing: {sorted(unknown)}")
        weight = _finite(list(weights.values()), f"weights.{mesh}").astype(float)
        if np.any(weight < 0) or not len(weight) or weight.max() <= 0:
            raise ValueError(f"Weights for {mesh} require positive total weight")
        weight /= weight.max()
        result[mesh] = values[:, [names.index(n) for n in weights]] @ (
            weight / weight.sum()
        )
    return result


def _load_predictor(config: SomaConfig, config_path: str | Path) -> Any:
    if not config.data_root.is_dir():
        raise FileNotFoundError(
            f"SKIM v2 dataset not found: {config.data_root}. Download SKIM v2 and update data_root in {config_path}."
        )
    if not config.checkpoint.is_file():
        raise FileNotFoundError(
            f"SOMA checkpoint not found: {config.checkpoint}. Install RELEASE_V2 checkpoints and update checkpoint in {config_path}."
        )
    from myosuite_soma.soma_model import load_soma_model

    return load_soma_model(
        config.data_root,
        config.subject,
        config.checkpoint,
        device=config.device,
        individual_muscles=config.individual_muscles,
        batch_size=config.batch_size,
    )


def inspect_config(config_path: str | Path) -> dict[str, Any]:
    """List exact rig/mesh names from installed assets for configuring mappings.

    Args:
        config_path: JSON asset configuration.

    Returns:
        A JSON-serializable inventory of joints, meshes and asset provenance.
    """
    config = SomaConfig.from_json(config_path)
    predictor = _load_predictor(config, config_path)
    return {
        "assets": predictor.metadata,
        "joints": [
            {
                "name": name,
                "parent": (
                    None if int(parent) < 0 else predictor.joint_names[int(parent)]
                ),
            }
            for name, parent in zip(predictor.joint_names, predictor.parents)
        ],
        "muscle_meshes": [mesh.name for mesh in predictor.meshes],
        "skin_mesh": predictor.skin_mesh.name,
        "retarget_source_bodies": {
            name: {"body": spec.body, "parent_body": spec.parent_body}
            for name, spec in _mappings(config).items()
        },
        "deformation_driver": "pose_only",
    }


def generate_animation(
    config_path: str | Path,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    time: np.ndarray,
    actuator_names: Sequence[str],
    activation: np.ndarray,
    *,
    physiology: dict[str, np.ndarray] | None = None,
) -> SomaAnimation:
    """Generate SOMA geometry from recorded poses and separate activation colors.

    Args:
        config_path: Explicit JSON assets and calibration.
        model: The source MuJoCo kinematic model.
        qpos: Recorded model positions, ``(T, nq)``.
        time: Strictly increasing recorded times in seconds.
        actuator_names: Names of recorded muscle activation columns.
        activation: Recorded activation states, ``(T, A)``.
        physiology: Optional named native muscle feature matrices.

    Returns:
        A cache-ready, pose-driven animation in MuJoCo world coordinates.

    Raises:
        ValueError: Invalid data, mappings, calibration or generated geometry.
        FileNotFoundError: SOMA assets have not been installed at configured paths.
    """
    return _generate_animation(
        config_path,
        model,
        qpos,
        time,
        actuator_names,
        activation,
        physiology=physiology,
    )


def _generate_animation(
    config_path: str | Path,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    time: np.ndarray,
    actuator_names: Sequence[str],
    activation: np.ndarray,
    *,
    physiology: dict[str, np.ndarray] | None = None,
    predictor: Any = None,
) -> SomaAnimation:
    """Shared generation path; a caller-owned predictor avoids loading twice."""
    config = SomaConfig.from_json(config_path)
    times = _finite(time, "time")
    poses = _source_poses(model, qpos, "qpos")
    if times.ndim != 1 or not len(times) or np.any(times[1:] <= times[:-1]):
        raise ValueError("time must be a nonempty strictly increasing vector")
    if poses.shape != (len(times), model.nq):
        raise ValueError(f"qpos must have shape ({len(times)}, {model.nq})")
    visual_activation = map_activations(config.muscle_map, actuator_names, activation)
    if np.shape(activation) != (len(times), len(actuator_names)):
        raise ValueError("activation must have one row per animation frame")
    if predictor is None:
        predictor = _load_predictor(config, config_path)
    rotations = retarget_rotations(
        config, model, poses, predictor.joint_names, predictor.parents
    )
    roots = np.broadcast_to(predictor.rest_joints[0], (len(times), 3)).copy()
    prediction = predictor.predict(rotations, root_positions=roots)
    conditioning_metadata = None
    if config.conditioning_model is not None:
        from myosuite_soma.soma_conditioning import apply_conditioning

        features = {} if physiology is None else dict(physiology)
        if "activation" in features and not np.array_equal(
            features["activation"], activation
        ):
            raise ValueError("Conflicting activation and physiology values")
        features["activation"] = activation
        prediction, conditioning_metadata = apply_conditioning(
            predictor,
            prediction,
            rotations,
            features,
            actuator_names,
            config.conditioning_model,
            allow_synthetic=config.allow_synthetic_conditioning,
        )
    placement = np.broadcast_to(config.output_translation, (len(times), 3)).copy()
    reference = _reference(config, model)
    anchor_body = config.anchor_body
    anchor_joint = config.anchor_joint
    root_body = config.root_body
    if config.preset == "myoElbow1D6":
        anchor_body = anchor_body or "r_humerus"
        anchor_joint = anchor_joint or "RightArm"
        root_body = root_body or "r_humerus"
    if anchor_body is not None:
        if anchor_joint not in predictor.joint_names:
            raise ValueError(f"SOMA anchor joint is missing: {anchor_joint}")
        anchor_id = _body_id(model, anchor_body)
        _require_qpos_body(model, anchor_id)
        native_anchor = predictor.rest_joints[predictor.joint_names.index(anchor_joint)]
        placement += reference.xpos[anchor_id] - config.scale * (
            config.output_rotation @ native_anchor
        )
    if root_body is not None:
        root_id = _body_id(model, root_body)
        _require_qpos_body(model, root_id)
        scratch = mujoco.MjData(model)
        for frame, pose in enumerate(poses):
            scratch.qpos[:] = pose
            mujoco.mj_forward(model, scratch)
            placement[frame] += (
                scratch.xpos[root_id]
                - reference.xpos[root_id]
                + scratch.xmat[root_id].reshape(3, 3) @ config.root_offset
            )
    meshes = []
    for spec in (*predictor.meshes, predictor.skin_mesh):
        native = (
            prediction.skin if spec.kind == "skin" else prediction.muscles[spec.name]
        )
        world = (
            config.scale * (np.asarray(native) @ config.output_rotation.T)
            + placement[:, None, :]
        )
        meshes.append(
            AnimatedMesh(
                spec.name,
                spec.faces,
                world.astype(np.float32),
                spec.texcoord,
                visual_activation.get(spec.name),
                spec.kind,
            )
        )
    unknown_meshes = config.muscle_map.keys() - {
        mesh.name for mesh in meshes if mesh.kind == "muscle"
    }
    if unknown_meshes:
        raise ValueError(
            f"muscle_map meshes are missing from SOMA output: {sorted(unknown_meshes)}"
        )
    metadata = dict(predictor.metadata)
    metadata.update(
        {
            "coordinate_system": "mujoco_world",
            "units": "m",
            "deformation_driver": "pose_only",
            "source_qpos_sha256": qpos_digest(poses),
            "source_model_sha256": model_digest(model),
            "source_activation_sha256": activation_digest(actuator_names, activation),
            "source_actuator_names": list(actuator_names),
            "environment": config.environment,
            "config_path": str(Path(config_path).resolve()),
            "muscle_map": config.muscle_map,
            "preset": config.preset,
            "retargeting": {
                "body_map": {
                    name: {
                        "body": spec.body,
                        "parent_body": spec.parent_body,
                        "basis": np.asarray(spec.basis).tolist(),
                        "rest_rotation": np.asarray(spec.rest_rotation).tolist(),
                    }
                    for name, spec in _mappings(config).items()
                },
                "basis": config.basis.tolist(),
                "rest_qpos": reference.qpos.tolist(),
                "output_rotation": config.output_rotation.tolist(),
                "output_translation": config.output_translation.tolist(),
                "root_offset": config.root_offset.tolist(),
                "scale": config.scale,
                "anchor_body": anchor_body,
                "anchor_joint": anchor_joint,
                "root_body": root_body,
            },
            "unmapped_joint_behavior": "identity",
        }
    )
    animation = SomaAnimation(times.copy(), tuple(meshes), metadata)
    if conditioning_metadata is not None:
        metadata["conditioning"] = conditioning_metadata
        metadata["deformation_driver"] = "pose_and_physiology"
    if config.geometry is not None:
        from myosuite_soma.soma_geometry import (
            contact_membership_from_predictor,
            deform_animation,
            source_joint_positions,
        )

        native_joints = source_joint_positions(predictor, rotations, roots)
        world_joints = (
            config.scale * (native_joints @ config.output_rotation.T)
            + placement[:, None]
        )
        animation = deform_animation(
            animation,
            source_joints=world_joints,
            joint_names=predictor.joint_names,
            parents=predictor.parents,
            model=model,
            qpos=poses,
            config=config.geometry,
            contact_membership=contact_membership_from_predictor(
                predictor, config.geometry.contacts
            )
            if config.geometry.contacts
            else None,
        ).animation
    validate_animation(
        animation,
        time=times,
        qpos=poses,
        model=model,
        actuator_names=actuator_names,
        activation=activation,
        physiology=physiology,
    )
    return animation


def main(argv: Sequence[str] | None = None) -> None:
    """Print an installed SOMA asset inventory using an explicit JSON config."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect-config",
        type=Path,
        required=True,
        help="List joint and muscle mesh names from configured SOMA assets.",
    )
    args = parser.parse_args(argv)
    try:
        result = inspect_config(args.inspect_config)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
