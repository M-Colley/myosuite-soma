# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Bounded native-muscle torque calibration with isolated recording holdouts.

The target is muscle-actuator generalized torque, optionally plus the native
passive generalized force. It is never silently interpreted as inverse-dynamics
torque including gravity, contact or external loads. EMG proxies require an
explicit opt-in and remain labelled as proxies in every result.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.optimize import least_squares

from myosuite_soma.soma import _source_poses, model_digest


@dataclass(frozen=True)
class CalibrationResult:
    """A separate native model, diagnostics and a builder-compatible profile."""

    model: mujoco.MjModel
    report: dict[str, Any]
    profile: dict[str, Any]


@dataclass(frozen=True)
class _Samples:
    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    activation: np.ndarray
    torque: np.ndarray
    actuator_names: tuple[str, ...]
    actuator_ids: np.ndarray
    dof_names: tuple[str, ...]
    joint_ids: np.ndarray
    recording_ids: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    provenance: dict[str, Any]
    activation_source: str


@dataclass(frozen=True)
class _Prepared:
    samples: _Samples
    lengths: np.ndarray
    velocities: np.ndarray
    moments: np.ndarray
    passive: np.ndarray


_CONFIG_FIELDS = {
    "version",
    "muscles",
    "target_type",
    "holdout_recordings",
    "allow_emg_proxy",
    "max_nfev",
    "regularization",
    "dof_weights",
    "rom_samples",
    "rollout_steps",
    "rollout_activation",
    "max_parameter_count",
}
_OBS_FIELDS = {
    "time",
    "qpos",
    "qvel",
    "activation",
    "joint_torque_Nm",
    "actuator_names",
    "dof_names",
    "recording_ids",
    "provenance",
    "activation_source",
    "split",
    "units",
    "input_sha256",
    "target_type",
    "conventions",
    "version",
}
_PARAMETERS = {"force_N": 2, "range0": 0, "range1": 1, "passive_force_scale": 7}


def _array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _names(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be an ordered sequence of names")
    result = tuple(value)
    if (
        not result
        or any(not isinstance(x, str) or not x for x in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain unique nonempty names")
    return result


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) - _CONFIG_FIELDS:
        raise ValueError("Unknown or invalid physical calibration configuration fields")
    result = dict(config)
    if result.get("version") != 1 or isinstance(result.get("version"), bool):
        raise ValueError("Physical calibration configuration requires version 1")
    if result.get("target_type", "actuator") not in ("actuator", "net"):
        raise ValueError("target_type must be actuator or net")
    result.setdefault("target_type", "actuator")
    for name, default, low, high in (
        ("max_nfev", 300, 1, 10000),
        ("rom_samples", 9, 2, 101),
        ("rollout_steps", 100, 0, 10000),
        ("max_parameter_count", 128, 1, 1000),
    ):
        result[name] = _integer(result.get(name, default), name, low, high)
    for name, default in (("regularization", 0.0), ("rollout_activation", 0.05)):
        value = result.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
        result[name] = float(value)
    if result["rollout_activation"] > 1:
        raise ValueError("rollout_activation must be in [0,1]")
    if not isinstance(result.get("allow_emg_proxy", False), bool):
        raise ValueError("allow_emg_proxy must be boolean")
    return result


def _muscles(model: mujoco.MjModel) -> np.ndarray:
    if model.nmocap:
        raise ValueError("Calibration does not infer missing mocap-body states")
    if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):
        raise ValueError("Torque calibration requires enabled muscle actuation")
    if np.any(model.jnt_actfrclimited):
        raise ValueError(
            "Aggregate joint actuator-force clipping is not supported by this calibrator"
        )
    ids = []
    for actuator in range(model.nu):
        if int(model.opt.disableactuator) & (1 << int(model.actuator_group[actuator])):
            raise ValueError(
                "Torque calibration requires enabled muscle actuator groups"
            )
        if int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            raise ValueError(
                "force_N calibration requires spatial tendon transmissions in metres"
            )
        tendon = int(model.actuator_trnid[actuator, 0])
        start, count = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
        if any(
            int(kind) == int(mujoco.mjtWrap.mjWRAP_JOINT)
            for kind in model.wrap_type[start : start + count]
        ):
            raise ValueError(
                "force_N calibration requires spatial rather than fixed-joint tendon paths"
            )
        if (
            int(model.actuator_gaintype[actuator]) != int(mujoco.mjtGain.mjGAIN_MUSCLE)
            or int(model.actuator_biastype[actuator])
            != int(mujoco.mjtBias.mjBIAS_MUSCLE)
            or int(model.actuator_actnum[actuator]) != 1
            or int(model.actuator_actadr[actuator]) < 0
            or model.actuator_actearly[actuator]
        ):
            raise ValueError(
                "Torque calibration currently requires native muscle gain/bias, "
                "one activation state per actuator and actearly=false"
            )
        ids.append(actuator)
    if not ids:
        raise ValueError("Calibration requires native muscle actuators")
    return np.asarray(ids, dtype=int)


def _equalities(model: mujoco.MjModel) -> list[int]:
    pending = {}
    for equality in range(model.neq):
        if not model.eq_active0[equality]:
            continue
        if int(model.eq_type[equality]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        child = int(model.eq_obj1id[equality])
        if child in pending:
            raise ValueError("Multiple active equalities drive one joint")
        pending[child] = equality
    ordered = []
    while pending:
        ready = [
            j for j, e in pending.items() if int(model.eq_obj2id[e]) not in pending
        ]
        if not ready:
            raise ValueError("Cyclic scalar joint equalities are unsupported")
        ordered.extend(pending.pop(j) for j in ready)
    return ordered


def _coordinates(
    model: mujoco.MjModel,
    pose: np.ndarray,
    joints: np.ndarray,
    *,
    project: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Project scalar equality chains and map independent-joint virtual work."""
    pose = pose.copy()
    mapping = np.zeros((model.nv, len(joints)))
    for column, joint in enumerate(joints):
        mapping[int(model.jnt_dofadr[joint]), column] = 1
    dependent = set()
    for equality in _equalities(model):
        child, parent = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
        dependent.add(child)
        if int(model.jnt_type[child]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ) or (
            parent >= 0
            and int(model.jnt_type[parent])
            not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        ):
            raise ValueError("Only scalar joint polynomial equalities are supported")
        c = int(model.jnt_qposadr[child])
        if parent >= 0:
            p = int(model.jnt_qposadr[parent])
            x = pose[p] - model.qpos0[p]
            parent_map = mapping[int(model.jnt_dofadr[parent])]
        else:
            x, parent_map = 0.0, np.zeros(len(joints))
        coefficients = model.eq_data[equality, :5]
        expected = model.qpos0[c] + np.polynomial.polynomial.polyval(x, coefficients)
        if not project and not np.isclose(pose[c], expected, rtol=0, atol=1e-5):
            raise ValueError(
                "Observation poses violate native joint polynomial equalities"
            )
        pose[c] = expected
        derivative = np.polynomial.polynomial.polyval(
            x, np.arange(1, 5) * coefficients[1:]
        )
        mapping[int(model.jnt_dofadr[child])] = derivative * parent_map
    if dependent.intersection(joints.tolist()):
        raise ValueError("dof_names must select independent hinge joints")
    return pose, mapping


def _validate_scalar_derivatives(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray | None = None,
    *,
    conventions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check measured derivatives against active scalar equality polynomials.

    Absolute defaults allow small numerical/soft-constraint residuals. Explicit
    overrides live in ``conventions.equality_kinematics_tolerances`` and use the
    source scalar coordinates' native SI units (rad/s or m/s, and their time
    derivatives). No sample or source coordinate is silently projected here.
    """
    supplied = (conventions or {}).get("equality_kinematics_tolerances", {})
    if not isinstance(supplied, Mapping) or set(supplied) - {
        "velocity_abs",
        "acceleration_abs",
    }:
        raise ValueError(
            "equality_kinematics_tolerances needs velocity_abs/acceleration_abs only"
        )
    tolerances = {"velocity_abs": 1e-3, "acceleration_abs": 1e-2}
    for name, value in supplied.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "Equality derivative tolerances must be finite and nonnegative"
            )
        tolerances[name] = float(value)
    by_joint = {}
    for equality in _equalities(model):
        child, parent = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
        scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        if int(model.jnt_type[child]) not in scalar or (
            parent >= 0 and int(model.jnt_type[parent]) not in scalar
        ):
            raise ValueError(
                "Derivative checks require scalar polynomial joint equalities"
            )
        coefficients = model.eq_data[equality, :5]
        if parent >= 0:
            address = int(model.jnt_dofadr[parent])
            position_address = int(model.jnt_qposadr[parent])
            offset = qpos[:, position_address] - model.qpos0[position_address]
            parent_velocity = qvel[:, address]
            parent_acceleration = qacc[:, address] if qacc is not None else None
        else:
            offset = parent_velocity = np.zeros(len(qpos))
            parent_acceleration = np.zeros(len(qpos)) if qacc is not None else None
        first = np.polynomial.polynomial.polyval(
            offset, np.arange(1, 5) * coefficients[1:]
        )
        second = np.polynomial.polynomial.polyval(
            offset, np.arange(2, 5) * np.arange(1, 4) * coefficients[2:]
        )
        child_address = int(model.jnt_dofadr[child])
        velocity_error = qvel[:, child_address] - first * parent_velocity
        name = model.joint(child).name or f"joint_{child}"
        error = {"maximum_abs_velocity_residual": float(np.max(np.abs(velocity_error)))}
        if error["maximum_abs_velocity_residual"] > tolerances["velocity_abs"]:
            raise ValueError(
                f"Observation velocity violates scalar joint equality for {name}: residual={error['maximum_abs_velocity_residual']:.6g}, tolerance={tolerances['velocity_abs']:.6g}"
            )
        if qacc is not None:
            acceleration_error = qacc[:, child_address] - (
                first * parent_acceleration + second * parent_velocity**2
            )
            error["maximum_abs_acceleration_residual"] = float(
                np.max(np.abs(acceleration_error))
            )
            if (
                error["maximum_abs_acceleration_residual"]
                > tolerances["acceleration_abs"]
            ):
                raise ValueError(
                    f"Observation acceleration violates scalar joint equality for {name}: residual={error['maximum_abs_acceleration_residual']:.6g}, tolerance={tolerances['acceleration_abs']:.6g}"
                )
        by_joint[name] = error
    return {
        "checked_equalities": len(by_joint),
        "velocity_abs_tolerance": tolerances["velocity_abs"],
        "acceleration_abs_tolerance": tolerances["acceleration_abs"]
        if qacc is not None
        else None,
        "by_joint": by_joint,
        "within_declared_tolerance": True,
        "scope": "Local active scalar-polynomial derivative consistency; native SI coordinate units, not measured physiology",
    }


def _samples(
    model: mujoco.MjModel,
    observations: Mapping[str, Any],
    config: dict[str, Any],
    *,
    allow_synthetic: bool,
    require_holdout: bool,
) -> _Samples:
    if not isinstance(observations, Mapping) or set(observations) - _OBS_FIELDS:
        raise ValueError("Unknown or invalid observation fields")
    if "version" in observations and observations["version"] != 1:
        raise ValueError("Observation schema requires version 1")
    if (
        "target_type" in observations
        and observations["target_type"] != config["target_type"]
    ):
        raise ValueError(
            "Observation target_type disagrees with calibration target_type"
        )
    expected_units = {
        "time": "s",
        "qpos": "SI",
        "qvel": "SI/s",
        "joint_torque_Nm": "Nm",
        "activation": "unitless",
    }
    if "units" in observations and observations["units"] != expected_units:
        raise ValueError(
            "Observation units must explicitly use seconds, SI kinematics and Nm"
        )
    provenance = observations.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("status") not in ("measured", "estimated", "synthetic")
        or not isinstance(provenance.get("source"), str)
        or not provenance["source"]
    ):
        raise ValueError("Observation provenance needs status and nonempty source")
    if provenance["status"] == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic observations require allow_synthetic=True")
    provenance = copy.deepcopy(provenance)
    if "input_sha256" in observations:
        digest = observations["input_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(x not in "0123456789abcdef" for x in digest.lower())
        ):
            raise ValueError("input_sha256 must be a SHA-256 hex string")
        provenance["observation_file_sha256"] = digest
    if "conventions" in observations:
        provenance["conventions"] = copy.deepcopy(observations["conventions"])
        if not isinstance(provenance["conventions"], dict):
            raise ValueError("Observation conventions must be a JSON object")
        conventions = provenance["conventions"]
        from myosuite_soma.physical_model import physical_digest

        for key in ("source_model_sha256", "source_kinematic_sha256"):
            if key in conventions:
                digest = conventions[key]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest.lower())
                ):
                    raise ValueError(f"conventions.{key} must be a SHA-256 hex string")
        if "source_kinematic_sha256" in conventions and conventions[
            "source_kinematic_sha256"
        ].lower() != model_digest(model):
            raise ValueError(
                "Observation source kinematic digest does not match the evaluated model"
            )
        if "source_model_sha256" in conventions:
            same = conventions["source_model_sha256"].lower() == physical_digest(model)
            if require_holdout and not same:
                raise ValueError(
                    "Observation source physical digest does not match the initial calibration model"
                )
            provenance["source_physics_comparison"] = {
                "observation_source_model_sha256": conventions[
                    "source_model_sha256"
                ].lower(),
                "candidate_model_sha256": physical_digest(model),
                "same_input_physics": same,
                "validated_initial_physical_source": bool(require_holdout and same),
            }
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Observation provenance must contain finite JSON values"
        ) from exc
    activation_source = observations.get("activation_source")
    if activation_source not in ("measured_activation", "emg_proxy", "synthetic"):
        raise ValueError("activation_source must explicitly describe activation inputs")
    if activation_source == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic activation requires allow_synthetic=True")
    if activation_source == "emg_proxy" and not config.get("allow_emg_proxy", False):
        raise ValueError(
            "EMG is not activation; emg_proxy requires allow_emg_proxy=true"
        )
    qpos = _source_poses(model, observations.get("qpos"), "qpos").copy()
    count = len(qpos)
    time = _array(observations.get("time"), "time", (count,))
    qvel = _array(observations.get("qvel"), "qvel", (count, model.nv))
    actuator_names = _names(observations.get("actuator_names"), "actuator_names")
    native_ids = _muscles(model)
    native_names = tuple(model.actuator(int(i)).name for i in native_ids)
    if set(actuator_names) != set(native_names):
        raise ValueError(
            "activation columns must cover all native muscle actuators exactly"
        )
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names], int)
    activation = _array(
        observations.get("activation"), "activation", (count, len(actuator_names))
    )
    if np.any(activation < 0) or np.any(activation > 1):
        raise ValueError("Observed activation or explicit EMG proxy must lie in [0,1]")
    dof_names = _names(observations.get("dof_names"), "dof_names")
    try:
        joint_ids = np.array([model.joint(name).id for name in dof_names], int)
    except KeyError as exc:
        raise ValueError("dof_names must name source hinge joints") from exc
    if any(
        int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE) for j in joint_ids
    ):
        raise ValueError(
            "Torque observations currently support named hinge joints in Nm"
        )
    torque = _array(
        observations.get("joint_torque_Nm"), "joint_torque_Nm", (count, len(dof_names))
    )
    recording_ids = np.asarray(observations.get("recording_ids"))
    if (
        recording_ids.shape != (count,)
        or recording_ids.dtype.kind not in "US"
        or any(not str(x) for x in recording_ids)
    ):
        raise ValueError("recording_ids must name one nonempty recording per sample")
    recording_ids = recording_ids.astype(str)
    for recording in np.unique(recording_ids):
        if np.any(np.diff(time[recording_ids == recording]) <= 0):
            raise ValueError("time must increase strictly within each recording")
    split = observations.get("split")
    holdout = config.get("holdout_recordings")
    if split is not None and holdout is not None:
        raise ValueError("Choose observation split or holdout_recordings, not both")
    if split is not None:
        split = np.asarray(split)
        if split.shape != (count,) or not np.isin(split, ["train", "validation"]).all():
            raise ValueError("split must contain train or validation for every sample")
        for recording in np.unique(recording_ids):
            if len(set(split[recording_ids == recording])) != 1:
                raise ValueError("A recording cannot cross train and validation splits")
        train, validation = split == "train", split == "validation"
    elif holdout is not None:
        holdout = _names(holdout, "holdout_recordings")
        if set(holdout) - set(recording_ids):
            raise ValueError("holdout_recordings includes an absent recording")
        validation = np.isin(recording_ids, holdout)
        train = ~validation
    else:
        train, validation = np.ones(count, bool), np.zeros(count, bool)
    if require_holdout and (not train.any() or not validation.any()):
        raise ValueError(
            "Fitting requires independent train and whole-recording validation data"
        )
    for pose in qpos:
        _coordinates(model, pose, joint_ids)
    provenance["joint_equality_kinematics"] = _validate_scalar_derivatives(
        model, qpos, qvel, conventions=observations.get("conventions")
    )
    return _Samples(
        time,
        qpos,
        qvel,
        activation,
        torque,
        actuator_names,
        actuator_ids,
        dof_names,
        joint_ids,
        recording_ids,
        train,
        validation,
        dict(provenance),
        activation_source,
    )


def _dense_moment(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    if data.actuator_moment.ndim == 2:
        return np.asarray(data.actuator_moment).copy()
    result = np.zeros((model.nu, model.nv))
    for actuator in range(model.nu):
        start, count = (
            int(data.moment_rowadr[actuator]),
            int(data.moment_rownnz[actuator]),
        )
        result[actuator, data.moment_colind[start : start + count]] = (
            data.actuator_moment[start : start + count]
        )
    return result


def _prepare(model: mujoco.MjModel, samples: _Samples) -> _Prepared:
    count, muscles, dofs = (
        len(samples.time),
        len(samples.actuator_ids),
        len(samples.dof_names),
    )
    lengths, velocities = np.empty((count, muscles)), np.empty((count, muscles))
    moments, passive = np.empty((count, muscles, dofs)), np.empty((count, dofs))
    data = mujoco.MjData(model)
    for frame, pose in enumerate(samples.qpos):
        mujoco.mj_resetData(model, data)
        data.time = samples.time[frame]
        data.qpos[:] = pose
        data.qvel[:] = samples.qvel[frame]
        data.act[model.actuator_actadr[samples.actuator_ids]] = samples.activation[
            frame
        ]
        mujoco.mj_forward(model, data)
        _, mapping = _coordinates(model, pose, samples.joint_ids)
        lengths[frame] = data.actuator_length[samples.actuator_ids]
        velocities[frame] = data.actuator_velocity[samples.actuator_ids]
        moments[frame] = _dense_moment(model, data)[samples.actuator_ids] @ mapping
        passive[frame] = data.qfrc_passive @ mapping
    return _Prepared(samples, lengths, velocities, moments, passive)


def _predict(
    model: mujoco.MjModel, prepared: _Prepared, target_type: str
) -> np.ndarray:
    samples = prepared.samples
    forces = np.empty_like(prepared.lengths)
    for column, actuator in enumerate(samples.actuator_ids):
        for frame in range(len(samples.time)):
            gain = mujoco.mju_muscleGain(
                prepared.lengths[frame, column],
                prepared.velocities[frame, column],
                model.actuator_lengthrange[actuator],
                model.actuator_acc0[actuator],
                model.actuator_gainprm[actuator, :9],
            )
            bias = mujoco.mju_muscleBias(
                prepared.lengths[frame, column],
                model.actuator_lengthrange[actuator],
                model.actuator_acc0[actuator],
                model.actuator_biasprm[actuator, :9],
            )
            # mj_forward evaluates the supplied activation state directly;
            # actrange limits integration, not this instantaneous observation.
            force = gain * samples.activation[frame, column] + bias
            if model.actuator_forcelimited[actuator]:
                force = np.clip(force, *model.actuator_forcerange[actuator])
            forces[frame, column] = force
    result = np.einsum("ta,tad->td", forces, prepared.moments)
    if target_type == "net":
        result += prepared.passive
    if not np.isfinite(result).all():
        raise ValueError("Native muscle prediction produced nonfinite torques")
    return result


def _freeze_auto_force(model: mujoco.MjModel) -> None:
    for actuator in _muscles(model):
        for parameters in (model.actuator_gainprm, model.actuator_biasprm):
            if parameters[actuator, 2] < 0:
                value = parameters[actuator, 3] / model.actuator_acc0[actuator]
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("Automatic native muscle strength is undefined")
                parameters[actuator, 2] = value


def _metrics(samples: _Samples, predicted: np.ndarray) -> dict[str, Any]:
    result = {}
    for name, mask in (
        ("all", np.ones(len(samples.time), bool)),
        ("train", samples.train),
        ("validation", samples.validation),
    ):
        if not mask.any():
            continue
        error = predicted[mask] - samples.torque[mask]
        result[name] = {
            "samples": int(mask.sum()),
            "recordings": sorted(set(samples.recording_ids[mask])),
            "rmse_Nm": float(np.sqrt(np.mean(error**2))),
            "max_abs_error_Nm": float(np.max(np.abs(error))),
            "by_dof_rmse_Nm": {
                key: float(value)
                for key, value in zip(
                    samples.dof_names, np.sqrt(np.mean(error**2, axis=0))
                )
            },
        }
    return result


def _native_lengths(model: mujoco.MjModel, actuator: int) -> dict[str, Any]:
    low, high = model.actuator_lengthrange[actuator]
    r0, r1 = model.actuator_gainprm[actuator, :2]
    l0 = (high - low) / (r1 - r0)
    lt = low - r0 * l0
    if not np.isfinite(l0 + lt) or l0 <= 0:
        raise ValueError(
            "Muscle operating ranges imply invalid effective fibre/tendon lengths"
        )
    return {
        "effective_optimal_length_m": float(l0),
        "effective_inelastic_tendon_length_m": float(lt),
        "anatomical_length_interpretation_valid": bool(lt >= -1e-9),
        "scope": "Effective constants of the native force-length fit, not independently measured tissue lengths",
    }


def _physical_checks(
    model: mujoco.MjModel, samples: _Samples, config: dict[str, Any]
) -> dict[str, Any]:
    if (
        not np.isfinite(model.body_mass).all()
        or np.any(model.body_mass < 0)
        or not np.isfinite(model.body_inertia).all()
        or np.any(model.body_inertia < 0)
    ):
        raise ValueError("Model contains invalid segment mass or inertia")
    massive = model.body_mass > 0
    inertia = model.body_inertia[massive]
    if np.any(inertia <= 0) or np.any(
        2 * inertia > inertia.sum(axis=1, keepdims=True) + 1e-12
    ):
        raise ValueError(
            "Model principal inertias must be positive and satisfy triangle inequalities"
        )
    curves = {}
    data = mujoco.MjData(model)
    native_ids = _muscles(model)
    for column, (name, joint) in enumerate(zip(samples.dof_names, samples.joint_ids)):
        address = int(model.jnt_qposadr[joint])
        limits = (
            model.jnt_range[joint]
            if model.jnt_limited[joint]
            else np.array(
                [samples.qpos[:, address].min(), samples.qpos[:, address].max()]
            )
        )
        angles = np.linspace(*limits, config["rom_samples"])
        passive_curve, lower_curve, upper_curve, moment_values = [], [], [], []
        for angle in angles:
            pose = model.qpos0.copy()
            pose[address] = angle
            pose, mapping = _coordinates(model, pose, samples.joint_ids, project=True)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            moments = (_dense_moment(model, data) @ mapping)[:, column]
            gain = np.array(
                [
                    mujoco.mju_muscleGain(
                        data.actuator_length[a],
                        0.0,
                        model.actuator_lengthrange[a],
                        model.actuator_acc0[a],
                        model.actuator_gainprm[a, :9],
                    )
                    for a in native_ids
                ]
            )
            bias = np.array(
                [
                    mujoco.mju_muscleBias(
                        data.actuator_length[a],
                        model.actuator_lengthrange[a],
                        model.actuator_acc0[a],
                        model.actuator_biasprm[a, :9],
                    )
                    for a in native_ids
                ]
            )
            minimum = np.where(
                model.actuator_actlimited, model.actuator_actrange[:, 0], 0.0
            )
            maximum = np.where(
                model.actuator_actlimited, model.actuator_actrange[:, 1], 1.0
            )
            minimum, maximum = np.maximum(minimum, 0.0), np.minimum(maximum, 1.0)
            zero, one = bias + gain * minimum, bias + gain * maximum
            passive_force = bias.copy()
            for a in native_ids:
                if model.actuator_forcelimited[a]:
                    passive_force[a] = np.clip(
                        passive_force[a], *model.actuator_forcerange[a]
                    )
                    zero[a] = np.clip(zero[a], *model.actuator_forcerange[a])
                    one[a] = np.clip(one[a], *model.actuator_forcerange[a])
            additional = (
                float(data.qfrc_passive @ mapping[:, column])
                if config["target_type"] == "net"
                else 0.0
            )
            base = float(zero @ moments) + additional
            increments = (one - zero) * moments
            passive_curve.append(float(passive_force @ moments) + additional)
            lower_curve.append(base + float(np.minimum(increments, 0).sum()))
            upper_curve.append(base + float(np.maximum(increments, 0).sum()))
            moment_values.append(moments.tolist())
        curves[name] = {
            "angles_rad": angles.tolist(),
            "passive_torque_Nm": passive_curve,
            "minimum_torque_Nm": lower_curve,
            "maximum_torque_Nm": upper_curve,
            "moment_arms_m": moment_values,
        }
    mujoco.mj_resetData(model, data)
    data.qpos[:] = samples.qpos[0]
    data.qvel[:] = samples.qvel[0]
    data.ctrl[:] = config["rollout_activation"]
    data.act[:] = config["rollout_activation"]
    maximum_speed, contacts, penetration = 0.0, 0, 0.0
    initial_pose = data.qpos.copy()
    warning_start = np.array([x.number for x in data.warning])
    for _ in range(config["rollout_steps"]):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Forward rollout produced nonfinite state")
        maximum_speed = max(maximum_speed, float(np.max(np.abs(data.qvel), initial=0)))
        contacts = max(contacts, int(data.ncon))
        if data.ncon:
            penetration = max(
                penetration, max(0.0, -float(min(c.dist for c in data.contact)))
            )
    warning_delta = np.array([x.number for x in data.warning]) - warning_start
    return {
        "segment_mass_kg": float(model.body_mass.sum()),
        "native_muscle_lengths": {
            model.actuator(int(a)).name: _native_lengths(model, int(a))
            for a in native_ids
        },
        "rom": curves,
        "torque_envelope_scope": "Independent activation bounds on each tested coordinate; not a simultaneous control policy",
        "moment_arm_coordinate_order": [
            model.actuator(int(a)).name for a in native_ids
        ],
        "rollout": {
            "steps": config["rollout_steps"],
            "timestep_s": float(model.opt.timestep),
            "finite": True,
            "warning_counts": warning_delta.tolist(),
            "stable_without_warnings": bool(not warning_delta.any()),
            "maximum_speed": maximum_speed,
            "maximum_contacts": contacts,
            "maximum_contact_penetration_m": penetration,
            "endpoint_qpos_max_change": float(
                np.max(np.abs(data.qpos - initial_pose), initial=0)
            ),
        },
    }


def validate_physical_model(
    model: mujoco.MjModel,
    observations: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Evaluate local torque observations, ROM and a bounded forward rollout."""
    if not isinstance(allow_synthetic, bool):
        raise ValueError("allow_synthetic must be boolean")
    config = _configuration(config)
    samples = _samples(
        model,
        observations,
        config,
        allow_synthetic=allow_synthetic,
        require_holdout=False,
    )
    prepared = _prepare(model, samples)
    predicted = _predict(model, prepared, config["target_type"])
    from myosuite_soma.physical_model import physical_digest

    origin = samples.provenance.get("source_physics_comparison", {})
    return {
        "candidate_model_sha256": physical_digest(model),
        "observation_source_model_sha256": origin.get(
            "observation_source_model_sha256"
        ),
        "same_input_physics": origin.get("same_input_physics"),
        "joint_equality_kinematics": samples.provenance["joint_equality_kinematics"],
        "target_type": config["target_type"],
        "target_definition": "qfrc_actuator"
        + (" + qfrc_passive" if config["target_type"] == "net" else ""),
        "polynomial_joint_coupling_projection": True,
        "metrics": _metrics(samples, predicted),
        "provenance": samples.provenance,
        "activation_source": samples.activation_source,
        "checks": _physical_checks(model, samples, config),
        "limitations": [
            "Joint torque and EMG do not uniquely identify individual muscle parameters.",
            "Native muscle tendons are inelastic and pennation is not explicitly represented.",
            "A finite short rollout does not establish long-term stability or a valid contact policy.",
        ],
    }


def fit_physical_model(
    model: mujoco.MjModel,
    observations: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    allow_synthetic: bool = False,
) -> CalibrationResult:
    """Fit bounded native parameters on training recordings, then score holdouts.

    ``muscles`` maps exact actuator names to two-element bounds for ``force_N``,
    ``range0``, ``range1`` and/or ``passive_force_scale``. The last parameter is
    MuJoCo's dimensionless ``fpmax``. All changed gain/bias parameters remain
    consistent. No observation or caller-owned model is modified.
    """
    if not isinstance(allow_synthetic, bool):
        raise ValueError("allow_synthetic must be boolean")
    config = _configuration(config)
    samples = _samples(
        model,
        observations,
        config,
        allow_synthetic=allow_synthetic,
        require_holdout=True,
    )
    muscles = config.get("muscles")
    if not isinstance(muscles, dict) or not muscles:
        raise ValueError("Fitting requires nonempty named muscle parameter bounds")
    variant = copy.copy(model)
    _freeze_auto_force(variant)
    prepared = _prepare(variant, samples)
    baseline = _predict(variant, prepared, config["target_type"])
    parameters, low, high, initial, labels = [], [], [], [], []
    for name, bounds in muscles.items():
        if (
            name not in samples.actuator_names
            or not isinstance(bounds, dict)
            or not bounds
            or set(bounds) - _PARAMETERS.keys()
        ):
            raise ValueError(f"Unknown muscle or parameter bounds for {name}")
        actuator = model.actuator(name).id
        extrema = {}
        for key, interval in bounds.items():
            pair = _array(interval, f"{name}.{key}", (2,))
            if (
                pair[0] >= pair[1]
                or pair[0] < 0
                or (key in ("force_N", "range1") and pair[0] <= 0)
            ):
                raise ValueError(f"{name}.{key} needs increasing nonnegative bounds")
            slot = _PARAMETERS[key]
            value = float(
                variant.actuator_gainprm[actuator, slot]
                if key != "passive_force_scale"
                else variant.actuator_biasprm[actuator, slot]
            )
            parameters.append((actuator, slot))
            low.append(pair[0])
            high.append(pair[1])
            initial.append(float(np.clip(value, *pair)))
            labels.append(f"{name}.{key}")
            extrema[key] = pair
        r0 = extrema.get("range0", variant.actuator_gainprm[actuator, :2][[0, 0]])
        r1 = extrema.get("range1", variant.actuator_gainprm[actuator, :2][[1, 1]])
        ell0, ell1 = variant.actuator_lengthrange[actuator]
        if (
            r0[1] >= r1[0]
            or ell0 < 0
            or ell1 <= ell0
            or (
                set(bounds).intersection({"range0", "range1"})
                and ell0 - r0[1] * (ell1 - ell0) / (r1[0] - r0[1]) < -1e-9
            )
        ):
            raise ValueError(
                f"{name} bounds must preserve positive effective fibre length and nonnegative tendon length throughout"
            )
        if set(bounds).intersection({"range0", "range1"}) and not np.array_equal(
            variant.actuator_gainprm[actuator, :2],
            variant.actuator_biasprm[actuator, :2],
        ):
            raise ValueError(
                "Fitting operating ranges requires matching initial native gain/bias ranges"
            )
    if len(parameters) > config["max_parameter_count"]:
        raise ValueError("Requested calibration exceeds max_parameter_count")
    initial = np.asarray(initial)
    scaling = np.maximum(np.abs(initial), 0.1)
    dof_weights = config.get("dof_weights", {})
    if not isinstance(dof_weights, dict) or set(dof_weights) - set(samples.dof_names):
        raise ValueError("dof_weights must use observed joint names")
    weight = np.array([dof_weights.get(n, 1.0) for n in samples.dof_names], dtype=float)
    if not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("dof_weights must be finite and positive")
    train = samples.train

    def assign(values):
        for (actuator, slot), value in zip(parameters, values):
            variant.actuator_gainprm[actuator, slot] = value
            variant.actuator_biasprm[actuator, slot] = value

    def residual(values, regularized=True):
        assign(values)
        error = (
            _predict(variant, prepared, config["target_type"])[train]
            - samples.torque[train]
        ) * np.sqrt(weight)
        if regularized and config["regularization"]:
            return np.r_[
                error.ravel(),
                np.sqrt(config["regularization"]) * (values - initial) / scaling,
            ]
        return error.ravel()

    result = least_squares(
        residual,
        initial,
        bounds=(low, high),
        x_scale=scaling,
        max_nfev=config["max_nfev"],
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    assign(result.x)
    predicted = _predict(variant, prepared, config["target_type"])
    # Numerical sensitivities exclude the regularization rows, which otherwise
    # falsely make underdetermined physical parameters look identifiable.
    jacobian = np.empty((int(train.sum()) * len(samples.dof_names), len(parameters)))
    for column, value in enumerate(result.x):
        step = 1e-5 * max(abs(value), 0.1)
        plus, minus = result.x.copy(), result.x.copy()
        plus[column] = min(high[column], value + step)
        minus[column] = max(low[column], value - step)
        jacobian[:, column] = (
            (residual(plus, False) - residual(minus, False))
            / (plus[column] - minus[column])
            * scaling[column]
        )
    assign(result.x)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = (
        max(jacobian.shape)
        * np.finfo(float).eps
        * (singular[0] if len(singular) else 0)
    )
    rank = int(np.sum(singular > max(tolerance, 1e-9)))
    from myosuite_soma.physical_model import physical_digest

    source_hash = physical_digest(model)
    status = (
        "synthetic"
        if samples.provenance["status"] == "synthetic"
        or samples.activation_source == "synthetic"
        else "estimated"
    )
    profile_muscles = {}
    for name, requested in muscles.items():
        actuator = model.actuator(name).id
        profile_muscles[name] = {}
        if "force_N" in requested:
            profile_muscles[name]["force_N"] = float(
                variant.actuator_gainprm[actuator, 2]
            )
        if "range0" in requested or "range1" in requested:
            profile_muscles[name]["range"] = variant.actuator_gainprm[
                actuator, :2
            ].tolist()
        if "passive_force_scale" in requested:
            profile_muscles[name]["passive_force_max"] = float(
                variant.actuator_biasprm[actuator, 7]
            )
    profile = {
        "version": 1,
        "source_model_sha256": source_hash,
        "provenance": {
            "status": status,
            "source": "Bounded native torque fit: " + samples.provenance["source"],
        },
        "muscles": profile_muscles,
    }
    report = {
        "version": 1,
        "source_model_sha256": source_hash,
        "fitted_model_sha256": physical_digest(variant),
        "validated_initial_physical_source": samples.provenance.get(
            "source_physics_comparison", {}
        ).get("validated_initial_physical_source", False),
        "joint_equality_kinematics": samples.provenance["joint_equality_kinematics"],
        "optimizer": {
            "success": bool(result.success),
            "message": result.message,
            "evaluations": int(result.nfev),
            "parameters": dict(zip(labels, result.x.tolist())),
        },
        "target_type": config["target_type"],
        "target_definition": "qfrc_actuator"
        + (" + qfrc_passive" if config["target_type"] == "net" else ""),
        "baseline": _metrics(samples, baseline),
        "fitted": _metrics(samples, predicted),
        "provenance": samples.provenance,
        "activation_source": samples.activation_source,
        "synthetic_only": status == "synthetic",
        "identifiability": {
            "parameter_count": len(parameters),
            "training_data_rank": rank,
            "full_column_rank": rank == len(parameters),
            "scaled_singular_values": singular.tolist(),
            "condition_number": float(singular[0] / singular[-1])
            if len(singular) and singular[-1] > tolerance and rank == len(parameters)
            else None,
            "scope": "Local training-torque sensitivity; full rank does not establish unique physiological truth",
        },
        "checks": _physical_checks(variant, samples, config),
        "limitations": [
            "Torque fits do not independently validate individual muscle forces or fibre properties.",
            "EMG proxies remain proxies; no unobserved activation is inferred.",
            "Independent recording holdouts are mandatory, but not equivalent to independent subject validation.",
            "Native tendons are inelastic and pennation is not explicitly modeled.",
        ],
    }
    return CalibrationResult(variant, report, profile)
