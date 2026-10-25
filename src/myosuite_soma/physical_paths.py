# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Offline fitting of reviewed attachment sites to local MTU-length observations.

This estimates body-local path-site positions, not biological fibre lengths or
tendon slack lengths. Native MuJoCo geometry evaluates every candidate path.
Whole recordings are held out and original source strength is preserved when
the final model's path operating ranges are recomputed.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.optimize import least_squares

from myosuite_soma.physical_calibration import (
    CalibrationResult,
    _array,
    _coordinates,
    _integer,
    _names,
)
from myosuite_soma.physical_model import build_physical_model, physical_digest
from myosuite_soma.soma import _source_poses, model_digest


def _actuators(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    indices = []
    for name in names:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if index < 0:
            raise ValueError(f"Unknown MTU actuator: {name}")
        if int(model.actuator_trntype[index]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            raise ValueError(
                "MTU lengths require a spatial tendon transmission in metres"
            )
        tendon = int(model.actuator_trnid[index, 0])
        first, count = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
        types = model.wrap_type[first : first + count]
        if (
            np.any(types == int(mujoco.mjtWrap.mjWRAP_JOINT))
            or np.any(types == int(mujoco.mjtWrap.mjWRAP_PULLEY))
            or not np.array_equal(model.actuator_gear[index], [1, 0, 0, 0, 0, 0])
        ):
            raise ValueError(
                "MTU observations require unit tendon gear and no fixed-joint or pulley path"
            )
        indices.append(int(index))
    return np.asarray(indices, int)


def _observations(
    model: mujoco.MjModel, observations: Mapping[str, Any], allow_synthetic: bool
):
    allowed = {
        "version",
        "time",
        "qpos",
        "actuator_names",
        "mtu_length_m",
        "recording_ids",
        "provenance",
        "units",
        "input_sha256",
        "conventions",
    }
    if not isinstance(observations, Mapping) or set(observations) - allowed:
        raise ValueError("Unknown or invalid MTU path observation fields")
    if "version" in observations and (
        observations["version"] != 1 or isinstance(observations["version"], bool)
    ):
        raise ValueError("MTU observations require schema version 1")
    provenance = observations.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("status") not in ("measured", "estimated", "synthetic")
        or not isinstance(provenance.get("source"), str)
        or not provenance["source"]
    ):
        raise ValueError(
            "MTU observations require explicit provenance status and source"
        )
    if provenance["status"] == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic MTU observations require allow_synthetic=True")
    provenance = copy.deepcopy(provenance)
    conventions = observations.get("conventions", {})
    if not isinstance(conventions, dict):
        raise ValueError("MTU observation conventions must be a JSON object")
    for key, actual in (
        ("source_model_sha256", physical_digest(model)),
        ("source_kinematic_sha256", model_digest(model)),
    ):
        expected = conventions.get(key)
        if expected is not None and expected != actual:
            raise ValueError(
                f"MTU observation {key} disagrees with the initial source model"
            )
    provenance["conventions"] = copy.deepcopy(conventions)
    if "input_sha256" in observations:
        digest = observations["input_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest.lower())
        ):
            raise ValueError("input_sha256 must be a SHA-256 hex string")
        provenance["observation_file_sha256"] = digest
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("MTU provenance must contain finite JSON values") from exc
    if "units" in observations and observations["units"] != {
        "time": "s",
        "qpos": "SI",
        "mtu_length_m": "m",
    }:
        raise ValueError(
            "MTU observation units must explicitly be seconds, SI qpos and metres"
        )
    qpos = _source_poses(model, observations.get("qpos"), "qpos").copy()
    count = len(qpos)
    times = _array(observations.get("time"), "time", (count,))
    names = _names(observations.get("actuator_names"), "actuator_names")
    indices = _actuators(model, names)
    lengths = _array(
        observations.get("mtu_length_m"), "mtu_length_m", (count, len(names))
    )
    if np.any(lengths <= 0):
        raise ValueError("MTU lengths must be positive in metres")
    recordings = np.asarray(observations.get("recording_ids"))
    if (
        recordings.shape != (count,)
        or recordings.dtype.kind not in "US"
        or any(not str(x) for x in recordings)
    ):
        raise ValueError("recording_ids must name every MTU observation")
    recordings = recordings.astype(str)
    for recording in np.unique(recordings):
        if np.any(np.diff(times[recordings == recording]) <= 0):
            raise ValueError("MTU observation time must increase within each recording")
    for pose in qpos:
        _coordinates(model, pose, np.array([], int))
    return qpos, names, indices, lengths, recordings, provenance


def _metrics(predicted, target, names, recordings, train, validation):
    result = {}
    for label, mask in (("train", train), ("validation", validation)):
        error = predicted[mask] - target[mask]
        result[label] = {
            "samples": int(mask.sum()),
            "recordings": sorted(set(recordings[mask])),
            "rmse_m": float(np.sqrt(np.mean(error**2))),
            "max_abs_error_m": float(np.max(np.abs(error))),
            "by_actuator_rmse_m": {
                name: float(value)
                for name, value in zip(names, np.sqrt(np.mean(error**2, axis=0)))
            },
        }
    return result


def fit_path_lengths(
    model: mujoco.MjModel,
    observations: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    allow_synthetic: bool = False,
) -> CalibrationResult:
    """Fit explicitly bounded body-local attachment positions to MTU lengths.

    Configuration requires ``version=1``, ``holdout_recordings`` and
    ``sites={name: {lower_m: [x,y,z], upper_m: [x,y,z]}}``. Equal coordinate
    bounds fix that coordinate rather than adding an unidentifiable variable.
    ``regularization`` penalizes deviations in metres from clipped source sites.
    Native source force scaling is frozen before the final geometry rebuild.
    No trained SOMA surface is treated as a measured attachment automatically.
    """
    if not isinstance(allow_synthetic, bool):
        raise ValueError("allow_synthetic must be an explicit boolean")
    if not isinstance(config, Mapping) or set(config) - {
        "version",
        "sites",
        "holdout_recordings",
        "max_nfev",
        "regularization",
        "max_parameter_count",
    }:
        raise ValueError("Unknown or invalid MTU path calibration configuration")
    if config.get("version") != 1 or isinstance(config.get("version"), bool):
        raise ValueError("MTU path calibration requires version 1")
    if model.nmocap:
        raise ValueError("MTU path calibration cannot infer missing mocap states")
    max_nfev = _integer(config.get("max_nfev", 300), "max_nfev", 1, 10000)
    max_parameters = _integer(
        config.get("max_parameter_count", 128), "max_parameter_count", 1, 1000
    )
    regularization = config.get("regularization", 0.0)
    if (
        isinstance(regularization, bool)
        or not isinstance(regularization, (int, float))
        or not np.isfinite(regularization)
        or regularization < 0
    ):
        raise ValueError("regularization must be finite and nonnegative")
    poses, names, actuator_ids, target, recordings, provenance = _observations(
        model, observations, allow_synthetic
    )
    holdouts = _names(config.get("holdout_recordings"), "holdout_recordings")
    if set(holdouts) - set(recordings):
        raise ValueError("MTU holdout_recordings names an absent recording")
    validation = np.isin(recordings, holdouts)
    train = ~validation
    if not train.any() or not validation.any():
        raise ValueError(
            "MTU path fitting requires independent train and whole-recording validation"
        )
    sites = config.get("sites")
    if not isinstance(sites, Mapping) or not sites:
        raise ValueError(
            "MTU path fitting requires explicitly reviewed named site bounds"
        )
    trial = copy.copy(model)
    variables, low, high, initial, labels = [], [], [], [], []
    site_ids = {}
    fixed = []
    for name, limits in sites.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(limits, Mapping)
            or set(limits) != {"lower_m", "upper_m"}
        ):
            raise ValueError(
                "Each site needs exact lower_m and upper_m coordinate bounds"
            )
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            raise ValueError(f"Unknown source attachment site: {name}")
        site_ids[name] = int(site)
        lower = _array(limits["lower_m"], f"{name}.lower_m", (3,))
        upper = _array(limits["upper_m"], f"{name}.upper_m", (3,))
        if np.any(lower > upper):
            raise ValueError("Site lower_m cannot exceed upper_m")
        for axis in range(3):
            if lower[axis] == upper[axis]:
                fixed.append((site, axis, float(lower[axis])))
            else:
                variables.append((site, axis))
                low.append(lower[axis])
                high.append(upper[axis])
                initial.append(
                    float(np.clip(model.site_pos[site, axis], lower[axis], upper[axis]))
                )
                labels.append(f"{name}.{'xyz'[axis]}")
    if not variables or len(variables) > max_parameters:
        raise ValueError(
            "MTU calibration needs 1..max_parameter_count non-fixed site coordinates"
        )
    data = mujoco.MjData(trial)

    def prediction(values=None):
        if values is not None:
            for (site, axis), value in zip(variables, values):
                trial.site_pos[site, axis] = value
            for site, axis, value in fixed:
                trial.site_pos[site, axis] = value
        result = np.empty_like(target)
        for frame, pose in enumerate(poses):
            data.qpos[:] = pose
            mujoco.mj_fwdPosition(trial, data)
            result[frame] = data.actuator_length[actuator_ids]
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError(
                "Candidate native paths produced nonfinite or nonpositive MTU lengths"
            )
        return result

    baseline = prediction()
    initial = np.asarray(initial)
    widths = np.asarray(high) - np.asarray(low)

    def residual(values, regularized=True):
        error = (prediction(values)[train] - target[train]).ravel()
        if regularized and regularization:
            return np.r_[error, np.sqrt(regularization) * (values - initial)]
        return error

    result = least_squares(
        residual,
        initial,
        bounds=(low, high),
        x_scale=widths,
        max_nfev=max_nfev,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    fitted = prediction(result.x)
    jacobian = np.empty((int(train.sum()) * len(names), len(variables)))
    for column, value in enumerate(result.x):
        step = max(widths[column] * 1e-5, 1e-8)
        plus, minus = result.x.copy(), result.x.copy()
        plus[column] = min(high[column], value + step)
        minus[column] = max(low[column], value - step)
        jacobian[:, column] = (
            (residual(plus, False) - residual(minus, False))
            / (plus[column] - minus[column])
            * widths[column]
        )
    prediction(result.x)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > max(tolerance, 1e-10)))
    status = "synthetic" if provenance["status"] == "synthetic" else "estimated"
    profile = {
        "version": 1,
        "source_model_sha256": physical_digest(model),
        "provenance": {
            "status": status,
            "source": "Bounded MTU-length path fit: " + provenance["source"],
        },
        "sites": {
            name: {"pos_m": trial.site_pos[index].tolist()}
            for name, index in site_ids.items()
        },
        "recompute_lengthrange": True,
    }
    built = build_physical_model(model, profile, allow_synthetic=allow_synthetic)
    # Rebuild must preserve the fitted geometry while recomputing operating
    # ranges and constants; evaluate it independently of the optimizer's model.
    check = mujoco.MjData(built.model)
    recompiled = np.empty_like(target)
    for frame, pose in enumerate(poses):
        check.qpos[:] = pose
        mujoco.mj_fwdPosition(built.model, check)
        recompiled[frame] = check.actuator_length[actuator_ids]
    if not np.allclose(recompiled, fitted, rtol=0, atol=1e-10):
        raise RuntimeError("Physical rebuild changed the fitted MTU path lengths")
    report = {
        "version": 1,
        "kind": "reviewed_native_path_length_calibration",
        "source_model_sha256": physical_digest(model),
        "fitted_model_sha256": physical_digest(built.model),
        "provenance": provenance,
        "synthetic_only": status == "synthetic",
        "optimizer": {
            "success": bool(result.success),
            "message": result.message,
            "evaluations": int(result.nfev),
            "parameters_m": dict(zip(labels, result.x.tolist())),
        },
        "baseline": _metrics(baseline, target, names, recordings, train, validation),
        "fitted": _metrics(recompiled, target, names, recordings, train, validation),
        "identifiability": {
            "parameter_count": len(variables),
            "training_data_rank": rank,
            "full_column_rank": rank == len(variables),
            "scaled_singular_values": singular.tolist(),
            "scope": "Local training-length sensitivity, excluding regularization and all holdout observations",
        },
        "physical_rebuild": built.report,
        "limitations": [
            "Measured or estimated MTU path length does not identify biological tendon slack or optimal fibre length.",
            "Site bounds and anatomical correspondences are supplied explicitly by the caller, not verified automatically.",
            "SOMA surfaces are not treated as measured attachments or fibres.",
            "Length fits alone do not validate joint torque, tissue mechanics or contact behaviour.",
        ],
    }
    return CalibrationResult(built.model, report, profile)
