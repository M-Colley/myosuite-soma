# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0.
# Extracted from the audited MyoSuite SOMA integration; imports adapted.
# Source attribution: THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Experimental physiology-conditioned visual correctives, separate from SOMA.

Training geometry must already have correspondence and be unposed into the
same canonical corrective coordinates. No physiological training data ships
with this module. Synthetic artifacts are rejected during export by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_soma.soma_model import SOMAModel, SOMAPrediction
from myosuite_soma.soma_numeric import inspect_numeric_archive

_VERSION = 1
_UNITS = {
    "activation": "dimensionless",
    "actuator_length": "m_musculotendon",
    "actuator_velocity": "m_s_musculotendon",
    "actuator_force": "N_signed_transmission",
    "fiber_length": "m_fiber",
    "fiber_velocity": "m_s_fiber",
}
_GROUPS = ("subject", "motion", "load")


def _numeric(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.asarray(array, dtype=np.float64)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "US" or not len(array):
        raise ValueError(f"{name} must contain nonempty strings")
    result = tuple(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in array)
    if any(not x or len(x) > 512 for x in result):
        raise ValueError(f"Invalid {name}")
    return result


def _rotations(value: Any) -> np.ndarray:
    array = _numeric(value, "rotations")
    if (
        array.ndim != 4
        or array.shape[2:] != (3, 3)
        or not len(array)
        or not array.shape[1]
        or not np.allclose(array @ array.swapaxes(-1, -2), np.eye(3), atol=1e-4)
        or not np.allclose(np.linalg.det(array), 1, atol=1e-4)
    ):
        raise ValueError("rotations must be proper (frames,joints,3,3) matrices")
    return array


def _read_numeric(path: str | Path) -> dict[str, np.ndarray]:
    try:
        inspect_numeric_archive(
            path, max_bytes=2 * 1024**3, allow_text=True, max_members=80
        )
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, EOFError, zipfile.BadZipFile, TypeError) as error:
        raise ValueError(f"Invalid numeric conditioning archive: {path}") from error


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    raw = arrays.pop("metadata", None)
    if raw is None or raw.shape != () or raw.dtype.kind not in "US":
        raise ValueError("Missing scalar JSON conditioning metadata")
    value = raw.item()
    value = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if len(value) > 1024**2:
        raise ValueError("Conditioning metadata is too large")
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid conditioning JSON metadata") from error
    if (
        not isinstance(result, dict)
        or isinstance(result.get("schema_version"), bool)
        or result.get("schema_version") != _VERSION
    ):
        raise ValueError("Unsupported conditioning schema_version")
    try:
        json.dumps(result, allow_nan=False)
    except ValueError as error:
        raise ValueError("Conditioning metadata must contain finite JSON") from error
    return result


def _write(path: str | Path, metadata: dict[str, Any], **arrays: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".npz":
        raise ValueError("Conditioning archives must have the .npz suffix")
    with output.open("wb") as stream:
        np.savez_compressed(
            stream, metadata=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays
        )


def _anatomy_hash(
    points: np.ndarray, names: tuple[str, ...], counts: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(names).encode())
    digest.update(np.asarray(counts, dtype="<i8").tobytes())
    digest.update(np.asarray(points, dtype="<f4").tobytes())
    return digest.hexdigest()


def anatomy_arrays(model: SOMAModel) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Return canonical points and ordering used by numeric training artifacts."""
    specs = (*model.meshes, model.skin_mesh)
    return (
        np.concatenate([spec.rest_vertices for spec in specs]),
        tuple(spec.name for spec in specs),
        np.asarray([len(spec.rest_vertices) for spec in specs], dtype=np.int64),
    )


def physiology_digest(
    actuator_names: Sequence[str],
    physiology: Mapping[str, np.ndarray],
    required_features: Sequence[str],
) -> str:
    """Hash every required recorded feature with actuator order and physical units."""
    names = _strings(actuator_names, "actuator_names")
    if len(set(names)) != len(names):
        raise ValueError("Physiology digest requires unique actuator names")
    digest = hashlib.sha256(json.dumps(names).encode())
    frames = None
    for key in sorted(required_features):
        if key not in _UNITS or key not in physiology:
            raise ValueError(f"Missing conditioning physiology field: {key}")
        value = _numeric(physiology[key], key)
        if value.ndim != 2 or value.shape[1] != len(names):
            raise ValueError(f"Invalid conditioning physiology shape for {key}")
        if frames is not None and value.shape[0] != frames:
            raise ValueError("Conditioning physiology frames must align")
        frames = len(value)
        digest.update(json.dumps([key, _UNITS[key], list(value.shape)]).encode())
        digest.update(np.asarray(value, dtype="<f8", order="C").tobytes())
    return digest.hexdigest()


def validate_conditioning_source(
    metadata: Mapping[str, Any],
    physiology: Mapping[str, np.ndarray],
    actuator_names: Sequence[str] | None,
) -> None:
    """Verify cached conditioned geometry against all required recorded states."""
    if actuator_names is None:
        raise ValueError("Conditioned caches require named source physiology")
    required = metadata.get("required_features")
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(key, str) for key in required)
        or len(set(required)) != len(required)
        or "activation" not in required
    ):
        raise ValueError("Invalid conditioning feature provenance")
    if list(actuator_names) != metadata.get("source_actuator_names"):
        raise ValueError("Conditioning source actuator order does not match")
    units = metadata.get("feature_units")
    if not isinstance(units, dict) or any(
        key not in _UNITS or units.get(key) != _UNITS[key] for key in required
    ):
        raise ValueError("Invalid conditioning source feature units")
    if physiology_digest(actuator_names, physiology, required) != metadata.get(
        "source_physiology_sha256"
    ):
        raise ValueError("Conditioning source physiology does not match the cache")


@dataclass(frozen=True)
class ConditioningDataset:
    """Corresponded canonical geometry and synchronized native physiology.

    ``baseline`` and ``target`` are unposed metre-valued vertices, not world
    scans. ``reference`` defines the exact zero-corrective physiology. Group
    IDs identify complete recordings and load conditions, never frame IDs.
    """

    rest_points: np.ndarray
    mesh_names: tuple[str, ...]
    mesh_counts: np.ndarray
    rotations: np.ndarray
    actuator_names: tuple[str, ...]
    states: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]
    baseline: np.ndarray
    target: np.ndarray
    subject: tuple[str, ...]
    motion: tuple[str, ...]
    load: tuple[str, ...]
    metadata: dict[str, Any]
    observation_weights: np.ndarray | None = None

    def validate(self) -> None:
        """Reject malformed geometry, unknown units, and ambiguous provenance."""
        points = _numeric(self.rest_points, "rest_points")
        rotations = _rotations(self.rotations)
        frames = len(rotations)
        counts = np.asarray(self.mesh_counts)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not len(points)
            or counts.dtype.kind not in "iu"
            or counts.shape != (len(self.mesh_names),)
            or np.any(counts <= 0)
            or sum(int(count) for count in counts) != len(points)
        ):
            raise ValueError("Invalid canonical points/mesh counts")
        for values, name in (
            (self.mesh_names, "mesh_names"),
            (self.actuator_names, "actuator_names"),
        ):
            names = _strings(values, name)
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate {name}")
        for name in _GROUPS:
            if len(_strings(getattr(self, name), name)) != frames:
                raise ValueError(f"{name} needs one group ID per frame")
        for value, name in ((self.baseline, "baseline"), (self.target, "target")):
            if _numeric(value, name).shape != (frames, len(points), 3):
                raise ValueError(f"Invalid {name} geometry shape")
        if self.observation_weights is not None:
            weights = _numeric(self.observation_weights, "observation_weights")
            if (
                weights.shape != (len(points),)
                or np.any(weights < 0)
                or not np.any(weights > 0)
            ):
                raise ValueError(
                    "observation_weights require nonnegative per-vertex confidence"
                )
        if "activation" not in self.states or set(self.states) != set(self.reference):
            raise ValueError("states/reference require matching activation features")
        units = self.metadata.get("feature_units")
        for name, value in self.states.items():
            if name not in _UNITS or not isinstance(units, dict):
                raise ValueError("Unsupported physiology feature or missing units")
            if units.get(name) != _UNITS[name]:
                raise ValueError(f"Incorrect units for {name}: expected {_UNITS[name]}")
            if _numeric(value, name).shape != (frames, len(self.actuator_names)):
                raise ValueError(f"Invalid physiology shape for {name}")
            if _numeric(self.reference[name], name).shape != (
                len(self.actuator_names),
            ):
                raise ValueError(f"Invalid reference physiology for {name}")
        if np.any(np.asarray(self.states["activation"]) < 0) or np.any(
            np.asarray(self.states["activation"]) > 1
        ):
            raise ValueError("activation must be in [0,1]")
        if self.metadata.get("coordinate_frame") != "SOMA_canonical_corrective_m":
            raise ValueError("Geometry must be in SOMA canonical corrective metres")
        if self.metadata.get("data_kind") not in (
            "synthetic",
            "paired_geometry",
            "pose_only",
        ):
            raise ValueError(
                "data_kind must declare synthetic/paired_geometry/pose_only"
            )
        if (
            not isinstance(self.metadata.get("provenance"), str)
            or not self.metadata["provenance"].strip()
        ):
            raise ValueError("Explicit geometry/physiology provenance is required")


def save_dataset(path: str | Path, data: ConditioningDataset) -> None:
    """Save a validated numeric, pickle-free training dataset."""
    data.validate()
    metadata = dict(data.metadata, schema_version=_VERSION)
    arrays: dict[str, Any] = {
        name: np.asarray(getattr(data, name))
        for name in (
            "rest_points",
            "mesh_names",
            "mesh_counts",
            "rotations",
            "actuator_names",
            "baseline",
            "target",
            *_GROUPS,
        )
    }
    arrays.update({f"state_{key}": value for key, value in data.states.items()})
    arrays.update({f"reference_{key}": value for key, value in data.reference.items()})
    arrays["observation_weights"] = (
        np.ones(len(data.rest_points))
        if data.observation_weights is None
        else data.observation_weights
    )
    _write(path, metadata, **arrays)


def load_dataset(path: str | Path) -> ConditioningDataset:
    """Load numeric arrays with strict units, shape, and provenance checks."""
    arrays = _read_numeric(path)
    metadata = _metadata(arrays)
    fixed = {
        "rest_points",
        "mesh_names",
        "mesh_counts",
        "rotations",
        "actuator_names",
        "baseline",
        "target",
        "observation_weights",
        *_GROUPS,
    }
    state_names = {name[6:] for name in arrays if name.startswith("state_")}
    expected = (
        fixed
        | {f"state_{key}" for key in state_names}
        | {f"reference_{key}" for key in state_names}
    )
    if set(arrays) != expected:
        raise ValueError("Missing or unexpected dataset arrays")
    data = ConditioningDataset(
        rest_points=arrays["rest_points"],
        mesh_names=_strings(arrays["mesh_names"], "mesh_names"),
        mesh_counts=arrays["mesh_counts"],
        rotations=arrays["rotations"],
        actuator_names=_strings(arrays["actuator_names"], "actuator_names"),
        states={key: arrays[f"state_{key}"] for key in sorted(state_names)},
        reference={key: arrays[f"reference_{key}"] for key in sorted(state_names)},
        baseline=arrays["baseline"],
        target=arrays["target"],
        subject=_strings(arrays["subject"], "subject"),
        motion=_strings(arrays["motion"], "motion"),
        load=_strings(arrays["load"], "load"),
        metadata=metadata,
        observation_weights=arrays["observation_weights"],
    )
    data.validate()
    return data


def grouped_split(
    data: ConditioningDataset,
    *,
    holdout_subjects: Sequence[str] = (),
    holdout_motions: Sequence[str] = (),
    holdout_loads: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out complete subjects, recordings, or loads using an OR rule.

    Selected IDs cannot occur in training. No random per-frame split exists.
    """
    data.validate()
    mask = np.zeros(len(data.rotations), dtype=bool)
    for name, selected in zip(
        _GROUPS, (holdout_subjects, holdout_motions, holdout_loads)
    ):
        available = np.asarray(getattr(data, name))
        if not set(selected).issubset(set(available)):
            raise ValueError(f"Unknown holdout {name} ID")
        mask |= np.isin(available, selected)
    if not mask.any() or mask.all():
        raise ValueError("Select nonempty whole-group training and holdout sets")
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def validation_gate(data: ConditioningDataset) -> dict[str, Any]:
    """Assess data eligibility without certifying physiological validity.

    The released S1 pose-only data cannot establish load-conditioned shape.
    Passing structural prerequisites still requires independent measurement
    validation and anatomical review outside this software.
    """
    data.validate()
    kind = data.metadata["data_kind"]
    matched: dict[tuple[str, bytes], list[int]] = {}
    for index, (subject, pose) in enumerate(zip(data.subject, data.rotations)):
        key = (subject, np.round(pose, decimals=5).tobytes())
        matched.setdefault(key, []).append(index)
    pairs = 0
    for indices in matched.values():
        first = indices[0]
        for second in indices[1:]:
            if (
                data.load[first] != data.load[second]
                and not np.allclose(
                    data.states["activation"][first],
                    data.states["activation"][second],
                    atol=1e-5,
                )
                and not np.allclose(data.target[first], data.target[second], atol=1e-6)
            ):
                pairs += 1
    reasons = []
    if kind != "paired_geometry":
        reasons.append(f"{kind} geometry is not independent paired physiology data")
    if not pairs:
        reasons.append("No matched-pose, different-load geometry/activation pairs")
    if not data.metadata.get("measurement_protocol"):
        reasons.append("Missing synchronized geometry/physiology measurement protocol")
    return {
        "data_kind": kind,
        "matched_pose_different_load_pairs": pairs,
        "eligible_for_external_validation": not reasons,
        "physiologically_validated": False,
        "can_claim_physiological_validity": False,
        "reasons": reasons,
        "remaining_gate": "Independent measured holdout geometry, uncertainty and anatomical review",
    }


def _basis(points: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    squared = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * squared / radius**2)


def _centers(points: np.ndarray, count: int) -> np.ndarray:
    chosen = [int(np.argmin(np.sum((points - points.mean(0)) ** 2, axis=1)))]
    distance = np.full(len(points), np.inf)
    for _ in range(1, min(count, len(points))):
        distance = np.minimum(distance, np.sum((points - points[chosen[-1]]) ** 2, 1))
        chosen.append(int(np.argmax(distance)))
    return points[chosen].copy()


def _features(
    rotations: np.ndarray,
    states: Mapping[str, np.ndarray],
    feature_names: tuple[str, ...],
    reference: np.ndarray,
    scale: np.ndarray,
    pose_projection: np.ndarray,
) -> np.ndarray:
    delta = (
        np.concatenate([states[key] for key in feature_names], axis=1) - reference
    ) / scale
    pose = (rotations - np.eye(3)).reshape(len(rotations), -1) @ pose_projection
    return np.concatenate(
        (delta, (delta[:, :, None] * pose[:, None, :]).reshape(len(delta), -1)), axis=1
    )


@dataclass(frozen=True)
class ConditioningFitConfig:
    """Controls for the regularized low-rank shared canonical vector field."""

    centers: int = 24
    radius_m: float = 0.18
    rank: int = 6
    pose_rank: int = 2
    ridge: float = 1e-5
    spatial_ridge: float = 1e-8
    max_displacement_m: float = 0.02


@dataclass(frozen=True)
class ConditioningModel:
    """Numeric research corrective with no intercept or simulator mutations."""

    centers: np.ndarray
    modes: np.ndarray
    weights: np.ndarray
    reference: np.ndarray
    scale: np.ndarray
    pose_projection: np.ndarray
    feature_min: np.ndarray
    feature_max: np.ndarray
    metadata: dict[str, Any]

    def offsets(
        self,
        points: np.ndarray,
        rotations: np.ndarray,
        physiology: Mapping[str, np.ndarray],
        actuator_names: Sequence[str],
    ) -> np.ndarray:
        """Predict shared canonical offsets; reject missing states and extrapolation."""
        poses = _rotations(rotations)
        names = tuple(self.metadata["actuator_names"])
        source = tuple(actuator_names)
        if len(set(source)) != len(source) or not set(names).issubset(source):
            raise ValueError("Conditioning requires the artifact's named actuators")
        ids = [source.index(name) for name in names]
        states = {}
        feature_names = tuple(self.metadata["required_features"])
        for key in feature_names:
            if key not in physiology:
                raise ValueError(
                    f"Conditioning requires recorded physiology field: {key}"
                )
            value = _numeric(physiology[key], key)
            if value.shape != (len(poses), len(source)):
                raise ValueError(f"Invalid conditioning physiology shape for {key}")
            states[key] = value[:, ids]
        if np.any(states["activation"] < 0) or np.any(states["activation"] > 1):
            raise ValueError("Conditioning activation must be in [0,1]")
        if poses.shape[1] != self.metadata["joints"]:
            raise ValueError("Conditioning pose joint count differs from training")
        features = _features(
            poses,
            states,
            feature_names,
            self.reference,
            self.scale,
            self.pose_projection,
        )
        tolerance = 0.05 * np.maximum(1.0, self.feature_max - self.feature_min)
        if np.any(features < self.feature_min - tolerance) or np.any(
            features > self.feature_max + tolerance
        ):
            raise ValueError("Conditioning input exceeds the training feature envelope")
        coefficients = (features @ self.weights) @ self.modes.reshape(
            len(self.modes), -1
        )
        field = np.einsum(
            "vk,tkc->tvc",
            _basis(_numeric(points, "points"), self.centers, self.metadata["radius_m"]),
            coefficients.reshape(len(poses), len(self.centers), 3),
        )
        if (
            not np.isfinite(field).all()
            or np.max(np.linalg.norm(field, axis=-1))
            > self.metadata["max_displacement_m"]
        ):
            raise ValueError("Conditioning correction exceeds its displacement bound")
        return field.astype(np.float32)


def train_conditioning(
    data: ConditioningDataset,
    train: np.ndarray,
    holdout: np.ndarray,
    config: ConditioningFitConfig | None = None,
) -> ConditioningModel:
    """Fit only training groups; reject pose-only geometry and frame leakage."""
    data.validate()
    cfg = config or ConditioningFitConfig()
    if data.metadata["data_kind"] == "pose_only":
        raise ValueError("Pose-only SOMA data cannot train physiological correctives")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not minimum <= value <= maximum
        for value, minimum, maximum in (
            (cfg.centers, 1, 128),
            (cfg.rank, 1, 64),
            (cfg.pose_rank, 0, 8),
        )
    ):
        raise ValueError("Invalid bounded integer centers/rank/pose_rank")
    if any(
        not np.isfinite(value) or value <= 0
        for value in (
            cfg.radius_m,
            cfg.ridge,
            cfg.spatial_ridge,
            cfg.max_displacement_m,
        )
    ):
        raise ValueError("Conditioning fit scales and regularizers must be positive")
    selections = []
    for values in (train, holdout):
        indices = np.asarray(values)
        if (
            indices.dtype.kind not in "iu"
            or indices.ndim != 1
            or not len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(data.rotations))
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Invalid conditioning split indices")
        selections.append(indices)
    train, holdout = selections
    groups = list(zip(data.subject, data.motion, data.load))
    if {groups[index] for index in train} & {groups[index] for index in holdout}:
        raise ValueError("Whole subject/motion/load groups must not cross the split")
    if set(train) & set(holdout) or len(set(train) | set(holdout)) != len(groups):
        raise ValueError("Training and holdout must partition the entire dataset")
    points = _numeric(data.rest_points, "rest_points")
    centers = _centers(points, cfg.centers)
    basis = _basis(points, centers, cfg.radius_m)
    residual = np.asarray(data.target[train] - data.baseline[train], dtype=np.float64)
    observation_weights = (
        np.ones(len(points))
        if data.observation_weights is None
        else data.observation_weights
    )
    lhs = basis.T @ (basis * observation_weights[:, None]) + cfg.spatial_ridge * np.eye(
        len(centers)
    )
    rhs = np.einsum("vk,tvc->ktc", basis * observation_weights[:, None], residual)
    coefficients = (
        np.linalg.solve(lhs, rhs.reshape(len(centers), -1))
        .reshape(len(centers), len(train), 3)
        .transpose(1, 0, 2)
        .reshape(len(train), -1)
    )
    _, _, modes = np.linalg.svd(coefficients, full_matrices=False)
    modes = modes[: min(cfg.rank, len(modes))]
    targets = coefficients @ modes.T
    pose_values = (data.rotations[train] - np.eye(3)).reshape(len(train), -1)
    _, _, pose_modes = np.linalg.svd(pose_values, full_matrices=False)
    pose_projection = pose_modes[: min(cfg.pose_rank, len(pose_modes))].T
    feature_names = tuple(sorted(data.states))
    reference = np.concatenate([data.reference[key] for key in feature_names])
    native = np.concatenate([data.states[key][train] for key in feature_names], axis=1)
    scale = np.maximum(np.std(native, axis=0), 1e-6)
    features = _features(
        data.rotations[train],
        {key: data.states[key][train] for key in feature_names},
        feature_names,
        reference,
        scale,
        pose_projection,
    )
    if len(features) < features.shape[1]:
        weights = features.T @ np.linalg.solve(
            features @ features.T + cfg.ridge * np.eye(len(features)), targets
        )
    else:
        weights = np.linalg.solve(
            features.T @ features + cfg.ridge * np.eye(features.shape[1]),
            features.T @ targets,
        )
    metadata = {
        "schema_version": _VERSION,
        "artifact_kind": "SOMA_shared_field_conditioning",
        "data_kind": data.metadata["data_kind"],
        "validation_status": "synthetic_only"
        if data.metadata["data_kind"] == "synthetic"
        else "research_only_unvalidated",
        "physiologically_validated": False,
        "coordinate_frame": "SOMA_canonical_corrective_m",
        "anatomy_sha256": _anatomy_hash(points, data.mesh_names, data.mesh_counts),
        "required_features": list(feature_names),
        "feature_units": {key: _UNITS[key] for key in feature_names},
        "actuator_names": list(data.actuator_names),
        "joints": data.rotations.shape[1],
        "radius_m": cfg.radius_m,
        "max_displacement_m": cfg.max_displacement_m,
        "provenance": data.metadata["provenance"],
        "baseline_checkpoint_sha256": data.metadata.get("baseline_checkpoint_sha256"),
        "baseline_rigid_checkpoint_sha256": data.metadata.get(
            "baseline_rigid_checkpoint_sha256"
        ),
        "joint_names": data.metadata.get("joint_names"),
        "validation_gate": validation_gate(data),
        "train_groups": [
            list(group) for group in sorted({groups[index] for index in train})
        ],
        "holdout_groups": [
            list(group) for group in sorted({groups[index] for index in holdout})
        ],
        "zero_corrective": "all physiology fields equal the stored reference; no intercept",
        "warning": "Experimental visual correction; no inferred force or contact validity",
        "observed_vertices": int(np.count_nonzero(observation_weights)),
        "unobserved_vertices": int(np.count_nonzero(observation_weights == 0)),
        "observation_scope": "Only positive-weight targets supervise fitting; other surfaces inherit the shared field and are not measured",
    }
    return ConditioningModel(
        centers,
        modes.reshape(len(modes), len(centers), 3),
        weights,
        reference,
        scale,
        pose_projection,
        np.minimum(features.min(0), 0),
        np.maximum(features.max(0), 0),
        metadata,
    )


def save_conditioning_model(path: str | Path, model: ConditioningModel) -> None:
    """Save a numeric corrective artifact without executable serialization."""
    _write(
        path,
        model.metadata,
        **{
            name: getattr(model, name)
            for name in (
                "centers",
                "modes",
                "weights",
                "reference",
                "scale",
                "pose_projection",
                "feature_min",
                "feature_max",
            )
        },
    )


def load_conditioning_model(path: str | Path) -> ConditioningModel:
    """Load a numeric research artifact and reject forged validation claims."""
    arrays = _read_numeric(path)
    metadata = _metadata(arrays)
    expected = {
        "centers",
        "modes",
        "weights",
        "reference",
        "scale",
        "pose_projection",
        "feature_min",
        "feature_max",
    }
    if set(arrays) != expected:
        raise ValueError("Missing or unexpected conditioning model arrays")
    arrays = {key: _numeric(value, key) for key, value in arrays.items()}
    if (
        metadata.get("artifact_kind") != "SOMA_shared_field_conditioning"
        or metadata.get("data_kind") not in ("synthetic", "paired_geometry")
        or metadata.get("physiologically_validated") is not False
        or metadata.get("coordinate_frame") != "SOMA_canonical_corrective_m"
    ):
        raise ValueError(
            "Invalid conditioning artifact or unsupported validation claim"
        )
    names = _strings(metadata.get("actuator_names"), "actuator_names")
    features = _strings(metadata.get("required_features"), "required_features")
    if (
        len(set(names)) != len(names)
        or len(set(features)) != len(features)
        or "activation" not in features
    ):
        raise ValueError("Invalid conditioning actuator/feature names")
    if any(
        key not in _UNITS or metadata.get("feature_units", {}).get(key) != _UNITS[key]
        for key in features
    ):
        raise ValueError("Invalid conditioning physiology units")
    joints = metadata.get("joints")
    if isinstance(joints, bool) or not isinstance(joints, int) or joints <= 0:
        raise ValueError("Invalid conditioning joint count")
    for key in ("radius_m", "max_displacement_m"):
        value = metadata.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid conditioning {key}")
    centers, modes = arrays["centers"], arrays["modes"]
    projection = arrays["pose_projection"]
    count = len(names) * len(features)
    if (
        centers.ndim != 2
        or centers.shape[1:] != (3,)
        or not 1 <= len(centers) <= 128
        or modes.ndim != 3
        or modes.shape[1:] != (len(centers), 3)
        or not 1 <= len(modes) <= 64
        or projection.ndim != 2
        or projection.shape[0] != joints * 9
        or projection.shape[1] > 8
        or arrays["reference"].shape != (count,)
        or arrays["scale"].shape != (count,)
        or np.any(arrays["scale"] <= 0)
        or arrays["weights"].shape != (count * (1 + projection.shape[1]), len(modes))
        or arrays["feature_min"].shape != (len(arrays["weights"]),)
        or arrays["feature_max"].shape != (len(arrays["weights"]),)
        or np.any(arrays["feature_min"] > arrays["feature_max"])
    ):
        raise ValueError("Invalid conditioning model dimensions")
    expected_status = (
        "synthetic_only"
        if metadata["data_kind"] == "synthetic"
        else "research_only_unvalidated"
    )
    signature = metadata.get("anatomy_sha256")
    if (
        metadata.get("validation_status") != expected_status
        or not isinstance(metadata.get("warning"), str)
        or not isinstance(signature, str)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise ValueError("Invalid conditioning status or anatomy fingerprint")
    return ConditioningModel(metadata=metadata, **arrays)


def evaluate_conditioning(
    model: ConditioningModel,
    data: ConditioningDataset,
    indices: np.ndarray,
) -> dict[str, Any]:
    """Measure canonical geometry errors for explicit held-out whole groups."""
    data.validate()
    if (
        _anatomy_hash(data.rest_points, data.mesh_names, data.mesh_counts)
        != model.metadata["anatomy_sha256"]
    ):
        raise ValueError("Evaluation anatomy differs from training artifact")
    values = np.asarray(indices)
    if (
        values.dtype.kind not in "iu"
        or values.ndim != 1
        or not len(values)
        or np.any(values < 0)
        or np.any(values >= len(data.rotations))
    ):
        raise ValueError("Invalid evaluation indices")
    prediction = model.offsets(
        data.rest_points,
        data.rotations[values],
        {key: array[values] for key, array in data.states.items()},
        data.actuator_names,
    )
    residual = data.target[values] - data.baseline[values]
    error = prediction - residual
    observed = (
        np.ones(len(data.rest_points))
        if data.observation_weights is None
        else data.observation_weights
    )

    def mean_squared(value: np.ndarray) -> float:
        return float(
            np.sum(np.sum(value**2, axis=-1) * observed[None])
            / (len(values) * np.sum(observed))
        )

    return {
        "frames": len(values),
        "vertices_per_frame": len(data.rest_points),
        "observed_vertices": int(np.count_nonzero(observed)),
        "baseline_rmse_m": float(np.sqrt(mean_squared(residual))),
        "corrected_rmse_m": float(np.sqrt(mean_squared(error))),
        "maximum_error_m": float(
            np.max(np.linalg.norm(error[:, observed > 0], axis=-1))
        ),
        "maximum_correction_m": float(np.max(np.linalg.norm(prediction, axis=-1))),
        "data_kind": data.metadata["data_kind"],
        "physiologically_validated": False,
    }


def apply_conditioning(
    predictor: SOMAModel,
    prediction: SOMAPrediction,
    rotations: np.ndarray,
    physiology: Mapping[str, np.ndarray],
    actuator_names: Sequence[str],
    model_path: str | Path,
    *,
    allow_synthetic: bool = False,
) -> tuple[SOMAPrediction, dict[str, Any]]:
    """Apply an explicitly selected experimental corrective before world placement.

    Synthetic artifacts require ``allow_synthetic=True``. All geometry uses one
    canonical field transported by the original skinning weights; zero feature
    delta preserves the released output exactly. Native mechanics are untouched.
    """
    model = load_conditioning_model(model_path)
    if model.metadata["data_kind"] == "synthetic" and not allow_synthetic:
        raise ValueError(
            "Synthetic conditioning requires explicit allow_synthetic opt-in"
        )
    points, names, counts = anatomy_arrays(predictor)
    if _anatomy_hash(points, names, counts) != model.metadata["anatomy_sha256"]:
        raise ValueError("Conditioning artifact does not match this canonical anatomy")
    if (
        model.metadata.get("joint_names") is not None
        and tuple(model.metadata["joint_names"]) != predictor.joint_names
    ):
        raise ValueError("Conditioning joint order does not match this SOMA predictor")
    for key, source in (
        ("baseline_checkpoint_sha256", "checkpoint"),
        ("baseline_rigid_checkpoint_sha256", "rigid_checkpoint"),
    ):
        expected = model.metadata.get(key)
        if (
            expected is not None
            and hashlib.sha256(
                Path(predictor.metadata[source]).read_bytes()
            ).hexdigest()
            != expected
        ):
            raise ValueError(
                "Conditioning was trained against a different SOMA checkpoint"
            )
    frames = len(rotations)
    if set(prediction.muscles) != {spec.name for spec in predictor.meshes}:
        raise ValueError("Conditioning prediction does not cover the expected muscles")
    for spec in (*predictor.meshes, predictor.skin_mesh):
        value = (
            prediction.skin if spec.kind == "skin" else prediction.muscles[spec.name]
        )
        if _numeric(value, spec.name).shape != (frames, len(spec.rest_vertices), 3):
            raise ValueError(
                "Conditioning prediction geometry has inconsistent dimensions"
            )
    offsets = model.offsets(points, rotations, physiology, actuator_names)
    sections = np.cumsum(counts)[:-1]
    canonical = dict(zip(names, np.split(offsets, sections, axis=1)))
    posed = predictor.pose_offsets(rotations, canonical)
    muscles = {name: value + posed[name] for name, value in prediction.muscles.items()}
    skin = prediction.skin + posed[predictor.skin_mesh.name]
    metadata = {
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "kind": "experimental_physiology_conditioned_shared_field",
        "data_kind": model.metadata["data_kind"],
        "validation_status": model.metadata["validation_status"],
        "physiologically_validated": False,
        "required_features": model.metadata["required_features"],
        "feature_units": model.metadata["feature_units"],
        "source_actuator_names": list(actuator_names),
        "source_physiology_sha256": physiology_digest(
            actuator_names, physiology, model.metadata["required_features"]
        ),
        "maximum_canonical_offset_m": float(np.max(np.linalg.norm(offsets, axis=-1))),
        "warning": model.metadata["warning"],
    }
    return SOMAPrediction(muscles, skin), metadata


def main(argv: Sequence[str] | None = None) -> None:
    """Train or evaluate numeric research correctives from explicit grouped data."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("dataset", type=Path)
    train_parser.add_argument("output", type=Path)
    train_parser.add_argument("--holdout-subject", action="append", default=[])
    train_parser.add_argument("--holdout-motion", action="append", default=[])
    train_parser.add_argument("--holdout-load", action="append", default=[])
    train_parser.add_argument("--radius-m", type=float, default=0.18)
    train_parser.add_argument("--centers", type=int, default=24)
    train_parser.add_argument("--rank", type=int, default=6)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("dataset", type=Path)
    eval_parser.add_argument("model", type=Path)
    args = parser.parse_args(argv)
    data = load_dataset(args.dataset)
    if args.command == "train":
        train, holdout = grouped_split(
            data,
            holdout_subjects=args.holdout_subject,
            holdout_motions=args.holdout_motion,
            holdout_loads=args.holdout_load,
        )
        model = train_conditioning(
            data,
            train,
            holdout,
            ConditioningFitConfig(
                centers=args.centers, rank=args.rank, radius_m=args.radius_m
            ),
        )
        save_conditioning_model(args.output, model)
        report = {
            "train": evaluate_conditioning(model, data, train),
            "holdout": evaluate_conditioning(model, data, holdout),
            "validation_gate": validation_gate(data),
        }
    else:
        model = load_conditioning_model(args.model)
        groups = {tuple(group) for group in model.metadata["holdout_groups"]}
        holdout = np.array(
            [
                index
                for index, group in enumerate(zip(data.subject, data.motion, data.load))
                if group in groups
            ],
            dtype=np.int64,
        )
        report = {
            "holdout": evaluate_conditioning(model, data, holdout),
            "validation_gate": validation_gate(data),
        }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
