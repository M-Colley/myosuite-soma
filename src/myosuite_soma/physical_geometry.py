# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Explicit geometry and anthropometric priors for physical model variants."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from myosuite_soma.physical_model import physical_digest
from myosuite_soma.soma import _source_poses, _require_qpos_body, validate_animation


def anthropometric_profile(
    model: mujoco.MjModel,
    segment_scales: Mapping[str, float],
    *,
    provenance: Mapping[str, Any],
    mass_policy: str = "preserve",
    target_mass_kg: float | None = None,
) -> dict[str, Any]:
    """Form an explicit, geometrically similar segment prior.

    Segment lengths and local COM scale by s, inertia by mass_ratio*s**2.
    Mass is preserved, multiplied by s**3 under constant density, or normalized
    to a measured target total using source mass*s**3 proportions. This is an
    initialization assumption, not evidence that strength follows body size.
    Strength is deliberately supplied by independent architecture/torque data.
    The builder's segment geometry shortcut supports primitive geoms only.
    """
    if not isinstance(segment_scales, Mapping):
        raise ValueError("segment_scales must map exact body names to factors")
    if not isinstance(mass_policy, str) or mass_policy not in {
        "preserve",
        "constant_density",
        "target_total",
    }:
        raise ValueError("Unknown anthropometric mass policy")
    if (target_mass_kg is not None) != (mass_policy == "target_total"):
        raise ValueError("target_mass_kg is required only for target_total")
    scales = np.ones(model.nbody)
    for name, value in segment_scales.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Segment scales require exact body names")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if (
            body <= 0
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid named segment scale: {name}")
        scales[body] = float(value)
    masses = model.body_mass.copy()
    if mass_policy != "preserve":
        masses *= scales**3
    if mass_policy == "target_total":
        if (
            isinstance(target_mass_kg, bool)
            or not isinstance(target_mass_kg, (int, float))
            or not np.isfinite(target_mass_kg)
            or target_mass_kg <= 0
            or masses.sum() <= 0
        ):
            raise ValueError("Target total mass must be finite and positive")
        masses *= target_mass_kg / masses.sum()
    edits = {}
    for body in range(1, model.nbody):
        name = model.body(body).name
        scale = scales[body]
        changed_mass = masses[body] != model.body_mass[body]
        if scale == 1 and not changed_mass:
            continue
        if not name:
            raise ValueError("Anthropometric edits require named source bodies")
        edit = {}
        if scale != 1:
            edit["scale"] = float(scale)
            edit["com_m"] = (model.body_ipos[body] * scale).tolist()
        if model.body_mass[body] > 0:
            ratio = masses[body] / model.body_mass[body]
            edit["mass_kg"] = float(masses[body])
            edit["inertia_kg_m2"] = (
                model.body_inertia[body] * ratio * scale**2
            ).tolist()
        edits[name] = edit
    return {
        "version": 1,
        "source_model_sha256": physical_digest(model),
        "provenance": {
            **dict(provenance),
            "anthropometric_prior": {
                "mass_policy": mass_policy,
                "target_mass_kg": target_mass_kg,
                "segment_scales": dict(segment_scales),
                "inertia_rule": "source inertia * mass_ratio * segment_scale**2",
                "status": "geometric similarity assumption, not subject validation",
            },
        },
        "bodies": edits,
    }


def capsule_profile_from_animation(
    animation,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    assignments: Mapping[str, Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit existing collision primitives from explicitly selected anatomy vertices.

    The complete recorded qpos trajectory must match the cache's provenance.
    Each native geom assignment supplies surface, vertex_ids and frame. Points
    are moved from the cache's world frame into that geom's native body frame.
    The caller reviews anatomical membership; no nearest-mesh body guess occurs.
    The resulting profile contains local geometry and must remain local.
    """
    poses = _source_poses(model, qpos, "collision source qpos")
    validate_animation(animation, model=model, qpos=poses)
    if len(poses) != len(animation.time):
        raise ValueError("Collision trajectory must match every animation sample")
    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError("Explicit collision geom/surface assignments are required")
    surfaces = {surface.name: surface for surface in animation.meshes}
    edits = {}
    scratch = mujoco.MjData(model)
    for name, spec in assignments.items():
        if not isinstance(spec, Mapping) or set(spec) - {
            "surface",
            "vertex_ids",
            "frame",
            "axis",
        }:
            raise ValueError(
                "Collision assignments use surface, vertex_ids, frame and optional axis"
            )
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0:
            raise ValueError(f"Unknown collision geom: {name}")
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            raise ValueError(f"{name} is not enabled as a collision geom")
        frame = spec.get("frame", 0)
        if (
            isinstance(frame, bool)
            or not isinstance(frame, int)
            or not 0 <= frame < len(poses)
        ):
            raise ValueError(
                "Collision reference frame is outside the source recording"
            )
        surface = surfaces.get(spec.get("surface"))
        if surface is None:
            raise ValueError("Collision surface must name an existing cached mesh")
        ids = np.asarray(spec.get("vertex_ids"))
        if (
            ids.dtype.kind not in "iu"
            or ids.ndim != 1
            or len(ids) < 4
            or np.any(ids < 0)
            or np.any(ids >= surface.points.shape[1])
            or len(set(ids.tolist())) != len(ids)
        ):
            raise ValueError("Select at least four unique valid collision vertex_ids")
        scratch.qpos[:] = poses[frame]
        mujoco.mj_forward(model, scratch)
        body = int(model.geom_bodyid[geom])
        _require_qpos_body(model, body)
        points = (surface.points[frame, ids] - scratch.xpos[body]) @ scratch.xmat[
            body
        ].reshape(3, 3)
        entry = {"capsule_points_m": points.tolist()}
        entry["provenance"] = {
            **dict(provenance),
            "surface": surface.name,
            "frame": frame,
            "vertex_ids": ids.tolist(),
            "source_qpos_sha256": animation.metadata.get("source_qpos_sha256"),
            "source_time_s": float(animation.time[frame]),
            "selection": "explicit caller-selected body membership; anatomical correspondence is not verified automatically",
        }
        if "axis" in spec:
            axis = np.asarray(spec["axis"])
            if (
                axis.shape != (3,)
                or axis.dtype.kind not in "fiu"
                or not np.isfinite(axis).all()
            ):
                raise ValueError(
                    "Collision capsule axis must be a finite body-local 3-vector"
                )
            entry["capsule_axis"] = axis.tolist()
        edits[name] = entry
    return {
        "version": 1,
        "source_model_sha256": physical_digest(model),
        "provenance": dict(provenance),
        "geoms": edits,
        "recompute_lengthrange": True,
    }
