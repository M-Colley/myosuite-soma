# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Explicit offline physical model variants, separate from visual SOMA export.

All observations/profiles and generated models stay local. Architecture inputs
are independent measurements or declared priors; SOMA surfaces are not silently
treated as measured physiological cross-sectional areas or fibre observations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np


@dataclass(frozen=True)
class BuiltPhysicalModel:
    """A private compiled model variant and, when available, its editable spec."""

    model: mujoco.MjModel
    spec: Any
    report: dict[str, Any]


def physical_digest(model: mujoco.MjModel) -> str:
    """Hash all exposed native numeric arrays, dimensions, names and options.

    Model arrays are constants, not an ``MjData`` state. Including derived
    constants makes this a fingerprint for a particular compiled representation,
    not a promise of equality across MuJoCo releases or compiler settings.
    """
    digest = hashlib.sha256(bytes(model.names))
    for prefix, obj in (("model", model), ("option", model.opt)):
        for name in sorted(n for n in dir(obj) if not n.startswith("_")):
            # Native compilation identity is reset when loading an MJB; it is
            # not a physical parameter or part of its serialized constants.
            if name == "signature":
                continue
            try:
                value = getattr(obj, name)
            except (AttributeError, RuntimeError):
                continue
            if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
                array = np.asarray(value)
                canonical = np.ascontiguousarray(
                    array.astype(array.dtype.newbyteorder("<"), copy=False)
                )
                digest.update(
                    f"{prefix}.{name}:{array.shape}:{array.dtype.kind}".encode()
                )
                digest.update(canonical.tobytes())
            elif isinstance(value, (int, float, bool, np.integer, np.floating)):
                digest.update(f"{prefix}.{name}:{value}".encode())
    return digest.hexdigest()


def _refresh_legacy_collision_bounds(model: mujoco.MjModel) -> None:
    """Refit existing collision hierarchies on engines before MuJoCo 3.17.

    Those engines' mj_setConst does not refresh geom bounds or body BVHs after
    numeric size/pose/inertial-frame edits. AABB half-extents transform by the
    absolute rotation matrix; parent boxes are unions of their children. The
    compiler's hierarchy and nonprimitive geometry bounds remain unchanged.
    See MuJoCo's setBound (added in 3.17) for native frame/flat-box conventions:
    https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_setconst.c
    The corresponding 3.15 implementation has no bound-refit pass:
    https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_setconst.c
    """
    if mujoco.mj_version() >= 3017000:
        return
    for geom in range(model.ngeom):
        kind = int(model.geom_type[geom])
        size = model.geom_size[geom]
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            half = np.full(3, size[0])
            radius = size[0]
        elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            half = np.array([size[0], size[0], size[0] + size[1]])
            radius = size[0] + size[1]
        elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            half = np.array([size[0], size[0], size[1]])
            radius = np.hypot(size[0], size[1])
        elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            half = size
            radius = np.max(size)
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            half = size
            radius = np.linalg.norm(size)
        else:
            continue
        model.geom_aabb[geom, :3] = 0
        model.geom_aabb[geom, 3:] = half
        model.geom_rbound[geom] = radius
    for body in range(model.nbody):
        first = int(model.body_bvhadr[body])
        count = int(model.body_bvhnum[body])
        if first < 0 or count == 0:
            continue
        inertial_rotation = np.empty(9)
        mujoco.mju_quat2Mat(inertial_rotation, model.body_iquat[body])
        inverse_rotation = inertial_rotation.reshape(3, 3).T
        lower = np.empty((count, 3))
        upper = np.empty((count, 3))
        for local in range(count - 1, -1, -1):
            node = first + local
            geom = int(model.bvh_nodeid[node])
            if geom >= 0:
                geom_rotation = np.empty(9)
                mujoco.mju_quat2Mat(geom_rotation, model.geom_quat[geom])
                rotation = inverse_rotation @ geom_rotation.reshape(3, 3)
                aabb = model.geom_aabb[geom]
                center = rotation @ aabb[:3] + inverse_rotation @ (
                    model.geom_pos[geom] - model.body_ipos[body]
                )
                half = np.abs(rotation) @ aabb[3:]
                lower[local], upper[local] = center - half, center + half
            else:
                children = model.bvh_child[node]
                if np.any(children <= local) or np.any(children >= count):
                    raise ValueError("Unsupported native collision hierarchy layout")
                lower[local] = np.min(lower[children], axis=0)
                upper[local] = np.max(upper[children], axis=0)
        flat = np.abs(upper - lower) < 1e-14
        lower[flat] -= 1e-14
        upper[flat] += 1e-14
        model.bvh_aabb[first : first + count, :3] = (upper + lower) / 2
        model.bvh_aabb[first : first + count, 3:] = (upper - lower) / 2


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"{name} must contain finite numbers with shape {shape}")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _keys(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    return value


def _provenance(value: Any, name: str, allow_synthetic: bool) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("status") not in ("measured", "estimated", "synthetic")
        or not isinstance(value.get("source"), str)
        or not value["source"].strip()
    ):
        raise ValueError(f"{name} needs status=measured/estimated/synthetic and source")
    if value["status"] == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic physical profiles require allow_synthetic=True")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    return value


def architecture_force(architecture: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """Estimate line-of-action strength with one explicit pennation convention.

    ``pcsa_kind='projected'`` already includes the pennation cosine; it is never
    applied again. Perpendicular PCSA and ``volume/fibre_length`` are projected
    exactly once. Specific tension is an input prior, not measured by SOMA.
    """
    value = _keys(
        architecture,
        {
            "volume_m3",
            "fibre_length_m",
            "specific_tension_pa",
            "pennation_deg",
            "pcsa_m2",
            "pcsa_kind",
        },
        "architecture",
    )
    sigma = _number(
        value.get("specific_tension_pa"), "specific_tension_pa", positive=True
    )
    angle = _number(value.get("pennation_deg", 0), "pennation_deg")
    if not 0 <= angle < 90:
        raise ValueError("pennation_deg must be in [0,90)")
    if "pcsa_m2" in value:
        if "volume_m3" in value or "fibre_length_m" in value:
            raise ValueError("Choose PCSA or volume/fibre length, not both")
        pcsa = _number(value["pcsa_m2"], "pcsa_m2", positive=True)
        kind = value.get("pcsa_kind")
        if kind not in ("perpendicular", "projected"):
            raise ValueError("pcsa_kind must explicitly be perpendicular or projected")
    else:
        if "pcsa_kind" in value:
            raise ValueError("pcsa_kind only applies to supplied pcsa_m2")
        pcsa = _number(value.get("volume_m3"), "volume_m3", positive=True) / _number(
            value.get("fibre_length_m"), "fibre_length_m", positive=True
        )
        kind = "perpendicular"
    projected = pcsa if kind == "projected" else pcsa * np.cos(np.deg2rad(angle))
    force = float(sigma * projected)
    if not np.isfinite(force) or force <= 0:
        raise ValueError("Architecture estimate must produce positive finite force")
    return force, {
        "method": "specific_tension_times_projected_pcsa",
        "input_pcsa_kind": kind,
        "projected_pcsa_m2": float(projected),
        "specific_tension_pa": sigma,
        "pennation_deg": angle,
        "force_N": force,
        "status": "estimated",
        "note": "Architecture-based prior; maximum force was not measured by this calculation",
    }


def fit_capsule(points: Any, axis: Any = None) -> dict[str, np.ndarray]:
    """Fit a conservative capsule around body-local reference surface samples.

    PCA selects only the axis when none is supplied. The cylinder covers the
    entire projected extent and the radius covers the greatest radial distance;
    its hemispheres therefore extend beyond the sampled axial endpoints. This
    is a geometric enclosure, not soft-tissue material or pressure calibration.
    """
    raw = np.asarray(points)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or len(raw) < 3:
        raise ValueError(
            "capsule_points_m must contain at least three body-local points"
        )
    points = _array(raw, raw.shape, "capsule_points_m")
    center = points.mean(0)
    centered = points - center
    if axis is None:
        _, directions = np.linalg.eigh(centered.T @ centered)
        direction = directions[:, -1]
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
    else:
        direction = _array(axis, (3,), "capsule_axis")
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError("capsule_axis must have nonzero length")
    direction /= length
    projection = centered @ direction
    low, high = float(projection.min()), float(projection.max())
    radial = centered - projection[:, None] * direction
    radius = float(np.linalg.norm(radial, axis=1).max())
    if radius <= 1e-10 or high - low <= 1e-10:
        raise ValueError("Capsule fit needs nonzero radial and axial extent")
    quat = np.empty(4)
    mujoco.mju_quatZ2Vec(quat, direction)
    return {
        "pos_m": center + direction * ((low + high) / 2),
        "quat_wxyz": quat,
        "size_m": np.array([radius, (high - low) / 2, 0.0]),
        "axis": direction,
    }


def _id(model: mujoco.MjModel, kind: Any, name: str) -> int:
    if not isinstance(name, str) or not name:
        raise ValueError("Model element names must be nonempty strings")
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"Model element missing: {name}")
    return int(value)


def _inertia(value: Any, label: str) -> np.ndarray:
    result = _array(value, (3,), label)
    # Normalize before summing: physical admissibility must not depend on an
    # arbitrary absolute unit tolerance or overflow for large finite moments.
    if np.any(result <= 0) or 1 > (result / result.max()).sum() - 1 + 1e-10:
        raise ValueError(
            f"{label} needs positive principal moments satisfying triangle inequalities"
        )
    return result


def _set_inertia(model: mujoco.MjModel, body: int, value: Any, label: str) -> None:
    raw = np.asarray(value)
    if raw.shape == (3,):
        model.body_inertia[body] = _inertia(raw, label)
        return
    tensor = _array(raw, (3, 3), label)
    if np.max(np.abs(tensor - tensor.T)) > 1e-10 * np.max(np.abs(tensor)):
        raise ValueError(f"{label} must be a symmetric body-local inertia tensor")
    tensor = 0.5 * (tensor + tensor.T)
    principal, rotation = np.linalg.eigh(tensor)
    _inertia(principal, label)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).ravel())
    model.body_inertia[body] = principal
    model.body_iquat[body] = quat


def _native_muscle(model: mujoco.MjModel, index: int) -> bool:
    return int(model.actuator_gaintype[index]) == int(
        mujoco.mjtGain.mjGAIN_MUSCLE
    ) and int(model.actuator_biastype[index]) == int(mujoco.mjtBias.mjBIAS_MUSCLE)


def _source(source: Any) -> tuple[mujoco.MjModel, Any, str]:
    if isinstance(source, mujoco.MjModel):
        return copy.copy(source), None, "compiled_model"
    if isinstance(source, mujoco.MjSpec):
        spec = source.copy()
        return spec.compile(), spec, "copied_spec"
    path = Path(source).resolve()
    if path.suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(path)), None, str(path)
    spec = (
        mujoco.MjSpec.from_zip(str(path))
        if path.suffix.lower() == ".zip"
        else mujoco.MjSpec.from_file(str(path))
    )
    return spec.compile(), spec, str(path)


def _json_profile(profile: Any, allow_synthetic: bool) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("Physical profile must be a JSON object")
    try:
        result = json.loads(json.dumps(profile, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Physical profiles must contain only finite JSON values"
        ) from exc
    _keys(
        result,
        {
            "version",
            "provenance",
            "source_model_sha256",
            "bodies",
            "sites",
            "geoms",
            "muscles",
            "recompute_lengthrange",
        },
        "physical profile",
    )
    if result.get("version") != 1 or isinstance(result.get("version"), bool):
        raise ValueError("Physical profile version must be 1")
    _provenance(result.get("provenance"), "profile.provenance", allow_synthetic)
    for section in ("bodies", "sites", "geoms", "muscles"):
        entries = result.setdefault(section, {})
        if not isinstance(entries, dict):
            raise ValueError(f"{section} must map exact native element names to edits")
        for name, entry in entries.items():
            if not isinstance(name, str) or not name or not isinstance(entry, dict):
                raise ValueError(f"{section} must map exact names to JSON edit objects")
            if "provenance" in entry:
                _provenance(
                    entry["provenance"], f"{section}.{name}.provenance", allow_synthetic
                )
    if "recompute_lengthrange" in result and not isinstance(
        result["recompute_lengthrange"], bool
    ):
        raise ValueError("recompute_lengthrange must be boolean")
    return result


def build_physical_model(
    source: mujoco.MjSpec | mujoco.MjModel | Path | str,
    profile: dict[str, Any],
    *,
    allow_synthetic: bool = False,
) -> BuiltPhysicalModel:
    """Build an independent physical variant from explicit, provenance-labelled edits.

    ``bodies.<name>.scale`` is an isotropic segment-local operation: it scales
    attached primitive geoms, sites, joint positions and immediate child body
    offsets. It does not scale the body's own attachment offset, descendants'
    shapes, mass or inertia. Bodies with mesh/hfield/SDF geoms need externally
    rescaled source assets and are rejected for this shortcut. Mass changes are
    explicit; a new mass alone preserves source COM/orientation and multiplies
    existing principal inertia by the mass ratio, without double-counting tissue.

    Native muscle ``force=-1`` is resolved from the ORIGINAL source acc0 before
    any mass/geometry edits, preventing unrequested mass-dependent strength
    changes. Active and passive F0 values are preserved separately unless an
    explicit architecture/force edit intentionally sets both to a common value.
    """
    profile = _json_profile(profile, allow_synthetic)
    model, spec, origin = _source(source)
    original = copy.copy(model)
    before = physical_digest(model)
    expected = profile.get("source_model_sha256")
    if expected is not None and expected != before:
        raise ValueError(
            "Physical profile source_model_sha256 does not match the source model"
        )
    recompile_bodies = []
    for name, edit in profile["bodies"].items():
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        com_changes = "com_m" in edit and not np.array_equal(
            _array(edit["com_m"], (3,), f"{name}.com_m"), model.body_ipos[body]
        )
        tensor_changes = "inertia_kg_m2" in edit and np.shape(
            edit["inertia_kg_m2"]
        ) == (3, 3)
        if model.body_simple[body] and (com_changes or tensor_changes):
            recompile_bodies.append(name)
    if recompile_bodies:
        if spec is None:
            raise ValueError(
                "COM/tensor edits invalidate this compiled simple-body mass-matrix layout. "
                "Supply the original MjSpec or XML so native sparsity can be recompiled."
            )
        # Sparse matrix layouts and allocation sizes are compiler products.
        # Clearing body_simple/dof_simplenum in-place is insufficient and can
        # produce incorrect inertial coupling even when mj_setConst succeeds.
        # Request the general layout on the private spec before its next compile.
        for body_spec in spec.bodies:
            if body_spec.id > 0:
                body_spec.simple = False
        model = spec.compile()
        if (model.nq, model.nv, model.nu, model.nbody) != (
            original.nq,
            original.nv,
            original.nu,
            original.nbody,
        ):
            raise ValueError(
                "Inertial recompilation unexpectedly changed model topology"
            )
    report: dict[str, Any] = {
        "version": 1,
        "source": origin,
        "source_model_sha256": before,
        "provenance": profile["provenance"],
        "profile": profile,
        "force_estimates": {},
        "auto_force_frozen": [],
        "capsule_fits": {},
        "lengthrange_recomputed": [],
        "recompiled_for_inertial_edits": recompile_bodies,
        "limitations": [
            "Native muscle tendons are inelastic and pennation is not a dynamic state",
            "Architecture-based strength is a prior, not a direct force measurement",
            "Collision fitting supplies geometry, not a soft-tissue contact material model",
            "Existing control policies require independent evaluation on the changed dynamics",
        ],
    }
    for index in range(model.nu):
        if not _native_muscle(model, index):
            continue
        for field in ("actuator_gainprm", "actuator_biasprm"):
            parameters = getattr(model, field)
            if parameters[index, 2] < 0:
                acceleration = float(original.actuator_acc0[index])
                if acceleration <= 0 or not np.isfinite(acceleration):
                    raise ValueError(
                        f"Cannot resolve automatic force for {model.actuator(index).name}"
                    )
                parameters[index, 2] = parameters[index, 3] / acceleration
                report["auto_force_frozen"].append(
                    {
                        "actuator": model.actuator(index).name,
                        "field": field,
                        "force_N": float(parameters[index, 2]),
                    }
                )
    geometry_changed = False
    # Native bodies are parent-first. This gives segment-local scaling and
    # explicit child offsets deterministic precedence regardless of JSON order.
    for name, edit in sorted(
        profile["bodies"].items(),
        key=lambda pair: _id(model, mujoco.mjtObj.mjOBJ_BODY, pair[0]),
    ):
        _keys(
            edit,
            {"pos_m", "scale", "mass_kg", "com_m", "inertia_kg_m2", "provenance"},
            f"bodies.{name}",
        )
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body == 0:
            raise ValueError("The world body cannot be physically personalized")
        if "pos_m" in edit:
            # A free body's reference translation belongs to qpos0, not to
            # body_pos. Updating only the latter would silently leave its
            # native forward pose unchanged and make export state ambiguous.
            # Pose placement belongs to an explicit initial-state operation.
            if np.any(
                (model.jnt_bodyid == body)
                & (model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
            ):
                raise ValueError(
                    f"{name}.pos_m cannot relocate a free-joint body; "
                    "set its explicit initial qpos instead"
                )
            model.body_pos[body] = _array(edit["pos_m"], (3,), f"{name}.pos_m")
            geometry_changed = True
        if "scale" in edit:
            scale = _number(edit["scale"], f"{name}.scale", positive=True)
            geoms = np.flatnonzero(model.geom_bodyid == body)
            primitives = {
                int(mujoco.mjtGeom.mjGEOM_SPHERE),
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                int(mujoco.mjtGeom.mjGEOM_BOX),
            }
            if any(int(model.geom_type[g]) not in primitives for g in geoms):
                raise ValueError(
                    f"{name}.scale supports primitive geoms only; rescale mesh source assets explicitly"
                )
            model.geom_pos[geoms] *= scale
            model.geom_size[geoms] *= scale
            model.site_pos[model.site_bodyid == body] *= scale
            model.jnt_pos[model.jnt_bodyid == body] *= scale
            children = np.flatnonzero(model.body_parentid == body)
            model.body_pos[children] *= scale
            geometry_changed = True
        if "mass_kg" in edit:
            mass = _number(edit["mass_kg"], f"{name}.mass_kg", positive=True)
            old_mass = float(original.body_mass[body])
            if old_mass <= 0:
                raise ValueError(f"{name} needs an existing positive segment mass")
            model.body_mass[body] = mass
            model.body_inertia[body] = original.body_inertia[body] * mass / old_mass
        if "com_m" in edit:
            model.body_ipos[body] = _array(edit["com_m"], (3,), f"{name}.com_m")
        if "inertia_kg_m2" in edit:
            _set_inertia(model, body, edit["inertia_kg_m2"], f"{name}.inertia_kg_m2")
    for name, edit in profile["sites"].items():
        _keys(edit, {"pos_m", "provenance"}, f"sites.{name}")
        site = _id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if "pos_m" not in edit:
            raise ValueError(f"sites.{name} requires pos_m")
        model.site_pos[site] = _array(edit["pos_m"], (3,), f"{name}.pos_m")
        geometry_changed = True
    for name, edit in profile["geoms"].items():
        _keys(
            edit,
            {"pos_m", "size_m", "capsule_points_m", "capsule_axis", "provenance"},
            f"geoms.{name}",
        )
        geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if "capsule_points_m" in edit:
            wrapping = np.isin(
                model.wrap_type,
                [
                    int(mujoco.mjtWrap.mjWRAP_SPHERE),
                    int(mujoco.mjtWrap.mjWRAP_CYLINDER),
                ],
            )
            if np.any(wrapping & (model.wrap_objid == geom)):
                raise ValueError(
                    f"{name} is a tendon wrapping geom and cannot become a capsule; "
                    "edit its supported wrapping position/size or use a separate collision geom"
                )
            if "pos_m" in edit or "size_m" in edit:
                raise ValueError(
                    "Capsule fitting cannot be mixed with explicit pos_m/size_m"
                )
            fitted = fit_capsule(edit["capsule_points_m"], edit.get("capsule_axis"))
            if int(model.geom_type[geom]) not in (
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(mujoco.mjtGeom.mjGEOM_SPHERE),
                int(mujoco.mjtGeom.mjGEOM_BOX),
                int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            ):
                raise ValueError(
                    "Capsule fitting can replace existing primitive geoms only"
                )
            model.geom_type[geom] = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            model.geom_pos[geom] = fitted["pos_m"]
            model.geom_quat[geom] = fitted["quat_wxyz"]
            model.geom_size[geom] = fitted["size_m"]
            report["capsule_fits"][name] = {k: v.tolist() for k, v in fitted.items()}
        else:
            if "capsule_axis" in edit:
                raise ValueError("capsule_axis requires capsule_points_m")
            if "pos_m" in edit:
                model.geom_pos[geom] = _array(edit["pos_m"], (3,), f"{name}.pos_m")
            if "size_m" in edit:
                sizes = _array(edit["size_m"], (3,), f"{name}.size_m")
                kind = int(model.geom_type[geom])
                required = {
                    int(mujoco.mjtGeom.mjGEOM_SPHERE): 1,
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2,
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER): 2,
                    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 3,
                    int(mujoco.mjtGeom.mjGEOM_BOX): 3,
                }.get(kind)
                if (
                    required is None
                    or np.any(sizes[:required] <= 0)
                    or np.any(sizes < 0)
                ):
                    raise ValueError(
                        "size_m must be valid for an existing primitive geom"
                    )
                model.geom_size[geom] = sizes
        geometry_changed = True
    explicit_ranges: dict[int, np.ndarray] = {}
    effective_targets: dict[int, tuple[float, float]] = {}
    for name, edit in profile["muscles"].items():
        _keys(
            edit,
            {
                "force_N",
                "architecture",
                "range",
                "passive_force_max",
                "lengthrange_m",
                "optimal_fibre_length_m",
                "tendon_slack_length_m",
                "provenance",
            },
            f"muscles.{name}",
        )
        actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if not _native_muscle(model, actuator):
            raise ValueError(f"{name} must use native muscle gain and bias types")
        if "force_N" in edit and "architecture" in edit:
            raise ValueError("Choose explicit force_N or architecture, not both")
        fibre_target = "optimal_fibre_length_m" in edit
        tendon_target = "tendon_slack_length_m" in edit
        if fibre_target != tendon_target:
            raise ValueError(
                "Provide optimal_fibre_length_m and tendon_slack_length_m together"
            )
        if fibre_target:
            if "range" in edit:
                raise ValueError(
                    "Choose effective fibre/slack lengths or normalized range, not both"
                )
            fibre = _number(
                edit["optimal_fibre_length_m"],
                f"{name}.optimal_fibre_length_m",
                positive=True,
            )
            tendon = _number(
                edit["tendon_slack_length_m"], f"{name}.tendon_slack_length_m"
            )
            if tendon < 0:
                raise ValueError("tendon_slack_length_m must be nonnegative")
            effective_targets[actuator] = fibre, tendon
        if fibre_target or "lengthrange_m" in edit:
            transmission = int(model.actuator_trntype[actuator])
            tendon_id = int(model.actuator_trnid[actuator, 0])
            if transmission != int(mujoco.mjtTrn.mjTRN_TENDON):
                raise ValueError(
                    "Metre-valued lengths require a spatial tendon transmission"
                )
            start = int(model.tendon_adr[tendon_id])
            count = int(model.tendon_num[tendon_id])
            if any(
                int(t) == int(mujoco.mjtWrap.mjWRAP_JOINT)
                for t in model.wrap_type[start : start + count]
            ):
                raise ValueError(
                    "Metre-valued lengths require a spatial, not fixed joint tendon"
                )
        if "architecture" in edit:
            force, estimate = architecture_force(edit["architecture"])
            report["force_estimates"][name] = estimate
        elif "force_N" in edit:
            force = _number(edit["force_N"], f"{name}.force_N")
            if force < 0:
                raise ValueError(
                    "force_N must be nonnegative; negative native auto-force is not an explicit estimate"
                )
        else:
            force = None
        for field in ("actuator_gainprm", "actuator_biasprm"):
            parameters = getattr(model, field)
            if force is not None:
                parameters[actuator, 2] = force
            if "range" in edit:
                operating = _array(edit["range"], (2,), f"{name}.range")
                if operating[0] < 0 or operating[1] <= operating[0]:
                    raise ValueError(
                        "Muscle range must be nonnegative and strictly increasing"
                    )
                parameters[actuator, :2] = operating
            if "passive_force_max" in edit:
                passive = _number(
                    edit["passive_force_max"], f"{name}.passive_force_max"
                )
                if passive < 0:
                    raise ValueError("passive_force_max must be nonnegative")
                parameters[actuator, 7] = passive
        if "lengthrange_m" in edit:
            lengths = _array(edit["lengthrange_m"], (2,), f"{name}.lengthrange_m")
            if lengths[0] < 0 or lengths[1] <= lengths[0]:
                raise ValueError(
                    "lengthrange_m must be nonnegative and strictly increasing"
                )
            explicit_ranges[actuator] = lengths
    recompute = geometry_changed or profile.get("recompute_lengthrange", False)
    if geometry_changed and profile.get("recompute_lengthrange") is False:
        raise ValueError(
            "Geometry edits require recomputed actuator lengths; stale ranges cannot be retained"
        )
    # MuJoCo 3.15/3.16 leave collision bounds stale even after mj_setConst.
    # COM/inertia edits also change the frame in which body BVHs are stored.
    _refresh_legacy_collision_bounds(model)
    # Geometry sizes/bounds, body inertias and acc0 all need fresh constants.
    mujoco.mj_setConst(model, mujoco.MjData(model))
    if recompute:
        options = mujoco.MjLROpt()
        mujoco.mj_defaultLROpt(options)
        options.useexisting = False
        options.uselimit = True
        # Match MuJoCo's compiler LengthRange preparation: its native helper
        # assumes actuation/contact/gravity/passive terms were disabled by its
        # caller. Leaving them active can make recomputation appear constant
        # or converge to strength/contact-dependent rather than kinematic ROM.
        old_flags, old_timestep = int(model.opt.disableflags), float(model.opt.timestep)
        disabled = 0
        for label in (
            "FRICTIONLOSS",
            "CONTACT",
            "SPRING",
            "DAMPER",
            "GRAVITY",
            "ACTUATION",
            "PASSIVE",
        ):
            disabled |= int(getattr(mujoco.mjtDisableBit, "mjDSBL_" + label, 0))
        try:
            model.opt.disableflags = disabled
            if options.timestep > 0:
                model.opt.timestep = options.timestep
            data = mujoco.MjData(model)
            for index in range(model.nu):
                if _native_muscle(model, index) and index not in explicit_ranges:
                    try:
                        mujoco.mj_setLengthRange(model, data, index, options)
                    except (mujoco.FatalError, RuntimeError, ValueError) as exc:
                        raise ValueError(
                            f"Length-range recomputation failed for {model.actuator(index).name}: {exc}. "
                            "Review paths/limits or supply independently calibrated lengthrange_m."
                        ) from exc
                    report["lengthrange_recomputed"].append(model.actuator(index).name)
        finally:
            model.opt.disableflags = old_flags
            model.opt.timestep = old_timestep
    for index, lengths in explicit_ranges.items():
        model.actuator_lengthrange[index] = lengths
    for index, (fibre, tendon) in effective_targets.items():
        operating = (model.actuator_lengthrange[index] - tendon) / fibre
        if (
            not np.isfinite(operating).all()
            or operating[0] < 0
            or operating[1] <= operating[0]
        ):
            raise ValueError(
                f"Effective fibre/slack lengths for {model.actuator(index).name} do not fit its path length range"
            )
        model.actuator_gainprm[index, :2] = operating
        model.actuator_biasprm[index, :2] = operating
    report["effective_lengths"] = {}
    for index in range(model.nu):
        if not _native_muscle(model, index):
            continue
        r0, r1 = model.actuator_gainprm[index, :2]
        low, high = model.actuator_lengthrange[index]
        if not np.isfinite([r0, r1, low, high]).all() or r1 <= r0 or high <= low:
            raise ValueError(
                f"Invalid final muscle operating ranges for {model.actuator(index).name}"
            )
        fibre = float((high - low) / (r1 - r0))
        tendon = float(low - r0 * fibre)
        spatial = int(model.actuator_trntype[index]) == int(mujoco.mjtTrn.mjTRN_TENDON)
        if spatial:
            tendon_id = int(model.actuator_trnid[index, 0])
            start, count = (
                int(model.tendon_adr[tendon_id]),
                int(model.tendon_num[tendon_id]),
            )
            spatial = not any(
                int(t) == int(mujoco.mjtWrap.mjWRAP_JOINT)
                for t in model.wrap_type[start : start + count]
            )
        effective = {
            "optimal_fibre_length": fibre,
            "inelastic_tendon_length": tendon,
            "units": "m" if spatial else "native_transmission_units",
            "note": "Effective native model constants, not independent tissue measurements",
        }
        if spatial:
            effective.update(
                optimal_fibre_length_m=fibre, inelastic_tendon_length_m=tendon
            )
        report["effective_lengths"][model.actuator(index).name] = effective
    dynamic = model.body_mass > 0
    if np.any(~np.isfinite(model.body_mass)) or np.any(model.body_mass < 0):
        raise ValueError("Final body masses must be finite and nonnegative")
    for body in np.flatnonzero(dynamic):
        _inertia(model.body_inertia[body], f"{model.body(body).name}.inertia")
    check = mujoco.MjData(model)
    mujoco.mj_forward(model, check)
    if not all(
        np.isfinite(getattr(check, n)).all()
        for n in ("qacc", "qfrc_bias", "qfrc_passive", "actuator_force")
    ):
        raise ValueError("Physical variant produced nonfinite native forward dynamics")
    report["total_mass_kg"] = float(model.body_mass.sum())
    report["source_total_mass_kg"] = float(original.body_mass.sum())
    report["target_model_sha256"] = physical_digest(model)
    if spec is not None and hasattr(spec, "copy_back"):
        try:
            spec.copy_back(model)
            # A path edit can change the default spring rest length on a
            # later compile, even when its compiled baseline value itself did
            # not change and copy_back therefore leaves the authored default.
            # Preserve the actual native spring interval explicitly, including
            # inactive springs, so active passive-tendon mechanics roundtrip.
            for tendon_spec in spec.tendons:
                if 0 <= tendon_spec.id < model.ntendon:
                    tendon_spec.springlength = model.tendon_lengthspring[
                        tendon_spec.id
                    ].copy()
        except ValueError as exc:
            report["mjcf_export_note"] = (
                f"Native edits cannot be represented in this spec; use MJB: {exc}"
            )
            spec = None
    elif spec is not None:
        report["mjcf_export_note"] = (
            "Editable MJCF requires MjSpec.copy_back (MuJoCo 3.17+); MJB remains available"
        )
        spec = None
    report["editable_spec_available"] = spec is not None
    return BuiltPhysicalModel(model, spec, report)


def _portable_spec(source: Any) -> Any:
    """Copy referenced external files into a private portable asset namespace."""
    spec = source.copy()
    original_assets = dict(spec.assets)
    portable: dict[str, bytes] = {}
    base = Path(spec.modelfiledir or ".")

    def capture(filename: str, directory: str, label: str) -> str:
        if not filename:
            return filename
        candidates = [filename, str(Path(directory) / filename), Path(filename).name]
        payload = next(
            (original_assets[key] for key in candidates if key in original_assets), None
        )
        if payload is None:
            paths = (
                [Path(filename)]
                if Path(filename).is_absolute()
                else [base / directory / filename, base / filename]
            )
            path = next((p for p in paths if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(
                    f"Referenced model asset is unavailable for portable export: {filename}"
                )
            payload = path.read_bytes()
        suffix = Path(filename).suffix
        name = f"assets/{label}{suffix}"
        portable[name] = bytes(payload)
        return name

    for kind, values, directory in (
        ("mesh", spec.meshes, spec.meshdir),
        ("texture", spec.textures, spec.texturedir),
        ("hfield", spec.hfields, spec.meshdir),
        ("skin", spec.skins, spec.meshdir),
    ):
        for index, asset in enumerate(values):
            asset.file = capture(asset.file, directory, f"{kind}_{index:05d}")
            if kind == "texture":
                cubes = list(asset.cubefiles)
                for face, name in enumerate(cubes):
                    if name:
                        cubes[face] = capture(
                            name, directory, f"{kind}_{index:05d}_face_{face}"
                        )
                asset.cubefiles = cubes
    spec.meshdir = ""
    spec.texturedir = ""
    # Programmatic specs can retain both legacy njmax/nstack and memory;
    # native compilation accepts that state, but the XML parser rejects it.
    # An explicit memory budget supersedes those legacy allocation hints.
    if spec.memory > 0:
        spec.njmax = -1
        spec.nstack = -1
    spec.assets = portable
    return spec


def _check_portable(
    original: mujoco.MjModel, restored: mujoco.MjModel
) -> dict[str, Any]:
    # Asset paths/addresses and compile signatures deliberately differ. Compare
    # actual physical/geometry arrays, not resource bookkeeping or global state.
    names = (
        "body_pos",
        "body_quat",
        "body_mass",
        "body_ipos",
        "body_iquat",
        "body_inertia",
        "jnt_type",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "geom_type",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "site_pos",
        "site_quat",
        "mesh_vert",
        "mesh_face",
        "hfield_data",
        "tex_data",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_dynprm",
        "actuator_lengthrange",
        "tendon_stiffness",
        "tendon_damping",
        "tendon_lengthspring",
        "qpos0",
    )
    if (original.nq, original.nv, original.nu, original.nbody) != (
        restored.nq,
        restored.nv,
        restored.nu,
        restored.nbody,
    ):
        raise ValueError("Portable model changed native model dimensions")
    differences = {}

    def quaternion_error(left, right):
        left, right = np.atleast_2d(left), np.atleast_2d(right)
        sign = np.where((left * right).sum(1) < 0, -1.0, 1.0)[:, None]
        # Stable near identity; q and -q represent the same rigid rotation.
        return 4 * np.arctan2(
            np.linalg.norm(left - sign * right, axis=1),
            np.linalg.norm(left + sign * right, axis=1),
        )

    vector_fields = {"body_pos", "body_ipos", "geom_pos", "site_pos", "jnt_pos"}
    quaternion_fields = {"body_quat", "body_iquat", "geom_quat", "site_quat"}
    for name in names:
        if not hasattr(original, name):
            continue
        left, right = getattr(original, name), getattr(restored, name)
        # MuJoCo's native XML writer uses six significant digits. For example,
        # 1.534315 kg is emitted as 1.53432 kg. MJB remains exact; the editable
        # form is checked against this explicit native serialization bound.
        equivalent = left.shape == right.shape
        angle_error = None
        if equivalent and left.dtype.kind in "biu":
            equivalent = np.array_equal(left, right)
        elif equivalent and name in quaternion_fields:
            angle_error = quaternion_error(left, right)
            equivalent = np.all(angle_error <= 2e-5)
        elif equivalent and name in vector_fields:
            equivalent = np.all(
                np.linalg.norm(left - right, axis=1)
                <= 1e-9 + 6e-6 * np.linalg.norm(left, axis=1)
            )
        elif equivalent and name == "qpos0":
            for joint in range(original.njnt):
                address = int(original.jnt_qposadr[joint])
                kind = int(original.jnt_type[joint])
                if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                    a, b = left[address : address + 3], right[address : address + 3]
                    equivalent &= np.linalg.norm(a - b) <= 1e-9 + 6e-6 * np.linalg.norm(
                        a
                    )
                    address += 3
                if kind in (
                    int(mujoco.mjtJoint.mjJNT_FREE),
                    int(mujoco.mjtJoint.mjJNT_BALL),
                ):
                    equivalent &= bool(
                        np.all(
                            quaternion_error(
                                left[address : address + 4],
                                right[address : address + 4],
                            )
                            <= 2e-5
                        )
                    )
                else:
                    equivalent &= bool(
                        np.isclose(left[address], right[address], atol=1e-9, rtol=6e-6)
                    )
        elif equivalent:
            equivalent = np.allclose(left, right, atol=1e-9, rtol=6e-6)
        if not equivalent:
            raise ValueError(
                f"Portable editable export changed {name}; use the verified MJB"
            )
        if left.size:
            difference = np.abs(left.astype(float) - right.astype(float))
            if np.any(difference):
                differences[name] = {
                    "maximum_absolute_difference": float(difference.max()),
                    "maximum_relative_difference": float(
                        (difference / np.maximum(np.abs(left), 1e-9)).max()
                    ),
                }
                if angle_error is not None:
                    differences[name]["maximum_orientation_difference_rad"] = float(
                        angle_error.max()
                    )
    return {
        "native_xml_significant_digits": 6,
        "relative_tolerance": 6e-6,
        "absolute_tolerance": 1e-9,
        "orientation_tolerance_rad": 2e-5,
        "nonzero_roundtrip_differences": differences,
    }


def _discard_temporary(path: Path, parent: Path) -> None:
    # Recursive cleanup is confined to the exact private directory created by
    # this exporter, never an existing caller directory or an external path.
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not path.name.startswith(
        ".physical-model-"
    ):
        raise RuntimeError("Refusing to clean an unexpected temporary export path")
    shutil.rmtree(resolved)


def save_physical_model(result: BuiltPhysicalModel, output_dir: Path | str) -> Path:
    """Save a private new directory containing MJB, report and optional MJCF zip.

    ``model.mjb`` is self-contained for this MuJoCo version. ``model.zip`` from
    MjSpec retains external mesh/texture assets, when editable-spec support is
    available; its XML can be inspected or loaded with ``MjSpec.from_zip``.
    Generated model/asset archives are local outputs and must not be published
    without independent permission for their source assets.
    """
    if not isinstance(result, BuiltPhysicalModel):
        raise ValueError("Expected BuiltPhysicalModel")
    target = Path(output_dir).resolve()
    if target.exists():
        raise ValueError(
            "Choose a new output directory; existing files are never overwritten"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".physical-model-", dir=target.parent))
    export_report = copy.deepcopy(result.report)
    try:
        mujoco.mj_saveModel(result.model, str(temporary / "model.mjb"))
        reloaded = mujoco.MjModel.from_binary_path(str(temporary / "model.mjb"))
        if physical_digest(reloaded) != physical_digest(result.model):
            raise ValueError("Saved MJB did not preserve physical model constants")
        if result.spec is not None:
            try:
                portable = _portable_spec(result.spec)
                portable.to_zip(str(temporary / "model.zip"))
                export_report["editable_roundtrip"] = _check_portable(
                    result.model,
                    mujoco.MjSpec.from_zip(str(temporary / "model.zip")).compile(),
                )
                (temporary / "model.xml").write_text(
                    portable.to_xml(), encoding="utf-8"
                )
                for name, payload in portable.assets.items():
                    asset = temporary / name
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    asset.write_bytes(payload)
            except (ValueError, OSError, mujoco.FatalError) as exc:
                # The exact self-contained binary was verified above. Optional
                # editable resources must not cause a valid binary to be lost.
                export_report.pop("editable_roundtrip", None)
                export_report["editable_spec_available"] = False
                export_report["editable_export_note"] = (
                    f"Verified MJB retained; editable export failed: {exc}"
                )
                for name in ("model.xml", "model.zip"):
                    (temporary / name).unlink(missing_ok=True)
                assets = temporary / "assets"
                if assets.is_dir() and assets.resolve().parent == temporary.resolve():
                    shutil.rmtree(assets)
        (temporary / "report.json").write_text(
            json.dumps(export_report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Windows and POSIX rename both reject an existing nonempty directory.
        # mkdir reservation avoids replacing an existing empty directory as well.
        target.mkdir(exist_ok=False)
        try:
            for file in temporary.iterdir():
                os.replace(file, target / file.name)
        except BaseException:
            for file in target.iterdir():
                if file.is_dir():
                    # The directory was created above and contains only copied
                    # local assets; reject any replaced/symlinked target.
                    if file.resolve().parent != target:
                        raise RuntimeError("Refusing unexpected asset cleanup target")
                    shutil.rmtree(file)
                else:
                    file.unlink()
            target.rmdir()
            raise
    finally:
        _discard_temporary(temporary, target.parent)
    return target
