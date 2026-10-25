# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Rigid-body dynamics validation from native kinematics and measured loads.

This does not infer activation or individual muscle force. Measured world-space
forces and moments are applied with native Jacobians; simulated contact forces
are omitted to avoid counting the same ground reaction twice.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

import mujoco
import numpy as np

from myosuite_soma.physical_calibration import (
    _array,
    _coordinates,
    _dense_moment,
    _equalities,
    _names,
    _validate_scalar_derivatives,
)
from myosuite_soma.physical_model import physical_digest
from myosuite_soma.soma import _source_poses, model_digest


def _coordinates_and_units(model, pose):
    _coordinates(model, pose, np.array([], int))
    projection = np.eye(model.nv)
    dependent = set()
    for equality in _equalities(model):
        child, parent = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
        dependent.add(child)
        if parent < 0:
            row = np.zeros(model.nv)
        else:
            x = (
                pose[int(model.jnt_qposadr[parent])]
                - model.qpos0[int(model.jnt_qposadr[parent])]
            )
            derivative = np.polynomial.polynomial.polyval(
                x, np.arange(1, 5) * model.eq_data[equality, 1:5]
            )
            row = derivative * projection[int(model.jnt_dofadr[parent])]
        projection[int(model.jnt_dofadr[child])] = row
    labels, units, free = [None] * model.nv, [None] * model.nv, {}
    independent = []
    for joint in range(model.njnt):
        name = model.joint(joint).name or f"joint_{joint}"
        address = int(model.jnt_dofadr[joint])
        kind = int(model.jnt_type[joint])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            names = [name + "." + axis for axis in ("tx", "ty", "tz", "rx", "ry", "rz")]
            labels[address : address + 6] = names
            units[address : address + 6] = ["N"] * 3 + ["Nm"] * 3
            free[name] = address
            independent.extend(range(address, address + 6))
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            labels[address : address + 3] = [
                name + "." + axis for axis in ("rx", "ry", "rz")
            ]
            units[address : address + 3] = ["Nm"] * 3
            independent.extend(range(address, address + 3))
        else:
            labels[address] = name
            units[address] = "Nm" if kind == int(mujoco.mjtJoint.mjJNT_HINGE) else "N"
            if joint not in dependent:
                independent.append(address)
    return projection, labels, units, free, independent, dependent


def _mass(model, data):
    result = np.empty((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, result)
    except TypeError:
        # MuJoCo before the MjData-based overload takes the sparse qM buffer.
        mujoco.mj_fullM(model, result, data.qM)
    return result


def validate_inverse_dynamics(
    model: mujoco.MjModel,
    observations: Mapping[str, Any],
    *,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Compare rigid-body force balance with explicit native-coordinate inputs.

    Required arrays: ``time``, ``qpos``, ``qvel``, ``qacc``. Each entry in
    ``external_forces`` needs an exact ``body`` plus ``point_world_m``,
    ``force_world_N`` and ``torque_world_Nm`` arrays of shape ``(T,3)``.
    An explicit empty list declares that there are no measured external loads.
    ``reference_generalized_force`` optionally compares independent hinge/slide
    joints listed in ``dof_names``; units are N for slides and Nm for hinges.

    The required force is ``M*qacc + bias - passive - J_external.T*wrench``.
    Active scalar joint equalities use virtual-work projection. Other equality
    types are rejected rather than silently omitting their constraint loads.
    Free-joint translations use the world basis and rotations the native local
    basis. This is a dynamics consistency test, not individual muscle validation.
    """
    if not isinstance(allow_synthetic, bool):
        raise ValueError("allow_synthetic must be an explicit boolean")
    allowed = {
        "version",
        "time",
        "qpos",
        "qvel",
        "qacc",
        "recording_ids",
        "external_forces",
        "provenance",
        "units",
        "conventions",
        "input_sha256",
        "dof_names",
        "reference_generalized_force",
    }
    if not isinstance(observations, Mapping) or set(observations) - allowed:
        raise ValueError("Unknown inverse-dynamics observation fields")
    if "version" in observations and (
        observations["version"] != 1 or isinstance(observations["version"], bool)
    ):
        raise ValueError("Inverse-dynamics observations require schema version 1")
    if model.nmocap:
        raise ValueError("Inverse dynamics cannot infer omitted mocap states")
    if not model.nv:
        raise ValueError(
            "Inverse dynamics requires at least one native degree of freedom"
        )
    massive = model.body_mass > 0
    inertia = model.body_inertia[massive]
    if (
        not np.isfinite(model.body_mass).all()
        or np.any(model.body_mass < 0)
        or not np.isfinite(inertia).all()
        or np.any(inertia <= 0)
        or np.any(2 * inertia > inertia.sum(axis=1, keepdims=True) + 1e-12)
    ):
        raise ValueError(
            "Inverse dynamics requires valid segment masses and physical principal inertias"
        )
    for equality in range(model.neq):
        if model.eq_active0[equality] and int(model.eq_type[equality]) != int(
            mujoco.mjtEq.mjEQ_JOINT
        ):
            raise ValueError(
                "Inverse dynamics currently supports only scalar polynomial joint equalities"
            )
    provenance = observations.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("status") not in ("measured", "estimated", "synthetic")
        or not isinstance(provenance.get("source"), str)
        or not provenance["source"]
    ):
        raise ValueError(
            "Inverse-dynamics observations require provenance status and source"
        )
    if provenance["status"] == "synthetic" and not allow_synthetic:
        raise ValueError(
            "Synthetic inverse-dynamics observations require explicit opt-in"
        )
    provenance = copy.deepcopy(provenance)
    if "input_sha256" in observations:
        digest = observations["input_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest.lower())
        ):
            raise ValueError("input_sha256 must be a SHA-256 hex string")
        provenance["observation_file_sha256"] = digest
    conventions = observations.get("conventions", {})
    if not isinstance(conventions, dict):
        raise ValueError("Inverse-dynamics conventions must be a JSON object")
    try:
        json.dumps(
            {"provenance": provenance, "conventions": conventions}, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Inverse-dynamics provenance requires finite JSON values"
        ) from exc
    if "source_kinematic_sha256" in conventions and conventions[
        "source_kinematic_sha256"
    ] != model_digest(model):
        raise ValueError("Inverse-dynamics source kinematic digest does not match")
    origin = conventions.get("source_model_sha256")
    if origin is not None and (
        not isinstance(origin, str)
        or len(origin) != 64
        or any(c not in "0123456789abcdef" for c in origin.lower())
    ):
        raise ValueError("source_model_sha256 must be a SHA-256 hex string")
    expected_units = {
        "time": "s",
        "qpos": "SI",
        "qvel": "SI/s",
        "qacc": "SI/s2",
        "external_forces": "world_SI",
        "reference_generalized_force": "native_SI",
    }
    if "units" in observations and observations["units"] != expected_units:
        raise ValueError(
            "Inverse-dynamics units must explicitly describe native SI and world-space external loads"
        )
    qpos = _source_poses(model, observations.get("qpos"), "qpos").copy()
    count = len(qpos)
    times = _array(observations.get("time"), "time", (count,))
    qvel = _array(observations.get("qvel"), "qvel", (count, model.nv))
    qacc = _array(observations.get("qacc"), "qacc", (count, model.nv))
    derivative_checks = _validate_scalar_derivatives(
        model, qpos, qvel, qacc, conventions=conventions
    )
    recordings = np.asarray(observations.get("recording_ids", ["recording"] * count))
    if (
        recordings.shape != (count,)
        or recordings.dtype.kind not in "US"
        or any(not str(x) for x in recordings)
    ):
        raise ValueError("recording_ids must identify every inverse-dynamics sample")
    for recording in set(recordings):
        if np.any(np.diff(times[recordings == recording]) <= 0):
            raise ValueError(
                "Inverse-dynamics times must increase within each recording"
            )
    external = observations.get("external_forces")
    if not isinstance(external, list):
        raise ValueError("Declare external_forces explicitly, including [] when absent")
    loads = []
    for entry in external:
        if not isinstance(entry, Mapping) or set(entry) != {
            "body",
            "point_world_m",
            "force_world_N",
            "torque_world_Nm",
        }:
            raise ValueError(
                "External loads need exact body, world point, force and torque fields"
            )
        body = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entry["body"])
            if isinstance(entry["body"], str)
            else -1
        )
        if body <= 0:
            raise ValueError(
                "External loads must name an existing moving/attached body, not world"
            )
        point = _array(entry["point_world_m"], "point_world_m", (count, 3))
        force = _array(entry["force_world_N"], "force_world_N", (count, 3))
        torque = _array(entry["torque_world_Nm"], "torque_world_Nm", (count, 3))
        loads.append((body, point, force, torque))
    reference = observations.get("reference_generalized_force")
    dof_names = observations.get("dof_names")
    if (reference is None) != (dof_names is None):
        raise ValueError(
            "Reference generalized forces and dof_names must be supplied together"
        )
    columns = []
    if reference is not None:
        dof_names = _names(dof_names, "dof_names")
        reference = _array(
            reference, "reference_generalized_force", (count, len(dof_names))
        )
        for name in dof_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or int(model.jnt_type[joint]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(
                    "Reference forces must select exact hinge or slide joint names"
                )
            columns.append(int(model.jnt_dofadr[joint]))
    candidate_hash = physical_digest(model)
    data = mujoco.MjData(model)
    required, projected, external_generalized, actuator_moments = [], [], [], []
    native_contacts = []
    for frame, pose in enumerate(qpos):
        projection, labels, units, free, independent, dependent = (
            _coordinates_and_units(model, pose)
        )
        if reference is not None and any(
            model.joint(name).id in dependent for name in dof_names
        ):
            raise ValueError(
                "Reference forces must select independent joints after equality projection"
            )
        data.qpos[:] = pose
        data.qvel[:] = qvel[frame]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        applied = np.zeros(model.nv)
        for body, point, force, torque in loads:
            mujoco.mj_applyFT(
                model, data, force[frame], torque[frame], point[frame], body, applied
            )
        force = (
            _mass(model, data) @ qacc[frame]
            + data.qfrc_bias
            - data.qfrc_passive
            - applied
        )
        required.append(force)
        projected.append(projection.T @ force)
        external_generalized.append(applied)
        moment = _dense_moment(model, data) @ projection
        if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):
            moment[:] = 0
        else:
            for actuator in range(model.nu):
                if int(model.opt.disableactuator) & (
                    1 << int(model.actuator_group[actuator])
                ):
                    moment[actuator] = 0
        actuator_moments.append(moment)
        native_contacts.append(int(data.ncon))
    required, projected = np.asarray(required), np.asarray(projected)
    if not np.isfinite(required).all():
        raise ValueError("Inverse dynamics produced nonfinite generalized force")
    actuated = np.any(np.abs(np.asarray(actuator_moments)) > 1e-10, axis=(0, 1))
    unactuated = [i for i in independent if not actuated[i]]
    free_residual = {}
    for name, address in free.items():
        force = projected[:, address : address + 3]
        torque = projected[:, address + 3 : address + 6]
        free_residual[name] = {
            "force_N": force.tolist(),
            "torque_Nm": torque.tolist(),
            "force_rms_norm_N": float(np.sqrt(np.mean(np.sum(force**2, axis=1)))),
            "torque_rms_norm_Nm": float(np.sqrt(np.mean(np.sum(torque**2, axis=1)))),
            "basis": "Translation: world XYZ; rotation: native free-joint body-local XYZ",
        }
    comparison = None
    if reference is not None:
        error = projected[:, columns] - reference
        comparison = {
            "by_dof": {
                name: {
                    "unit": units[column],
                    "rmse": float(np.sqrt(np.mean(error[:, i] ** 2))),
                    "max_abs_error": float(np.max(np.abs(error[:, i]))),
                }
                for i, (name, column) in enumerate(zip(dof_names, columns))
            }
        }
    return {
        "version": 1,
        "kind": "rigid_body_inverse_dynamics_consistency",
        "candidate_model_sha256": candidate_hash,
        "observation_source_model_sha256": origin,
        "same_input_physics": candidate_hash == origin if origin is not None else None,
        "provenance": copy.deepcopy(provenance),
        "conventions": copy.deepcopy(conventions),
        "formula": "M*qacc + qfrc_bias - qfrc_passive - external_generalized_force",
        "native_contact_forces_used": False,
        "maximum_native_contacts_ignored": max(native_contacts),
        "muscle_activation_required": False,
        "joint_equality_kinematics": derivative_checks,
        "unactuated_definition": "No enabled nonzero actuator transmission moment in any observed pose; activation and available force are not inferred",
        "external_load_count": len(loads),
        "coordinate_names": labels,
        "coordinate_units": units,
        "required_generalized_force": required.tolist(),
        "projected_required_generalized_force": projected.tolist(),
        "external_generalized_force": np.asarray(external_generalized).tolist(),
        "free_base_residuals": free_residual,
        "unactuated_residuals": {
            labels[i]: {
                "unit": units[i],
                "rmse": float(np.sqrt(np.mean(projected[:, i] ** 2))),
                "max_abs": float(np.max(np.abs(projected[:, i]))),
            }
            for i in unactuated
        },
        "reference_comparison": comparison,
        "limitations": [
            "Required generalized joint force is not a measurement of individual muscle force or activation.",
            "External force points, signs, body assignments and kinematics must already match the native source model.",
            "Native contact constraint forces are intentionally omitted because external measured loads are supplied separately.",
            "Only scalar polynomial joint equalities are projected; other equality constraints are rejected.",
        ],
    }
