# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0.
# Extracted from the audited MyoSuite SOMA integration; imports adapted.
# Source attribution: THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Optional coherent visual registration and native-clearance foot morphology."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from pathlib import Path

import mujoco
import numpy as np

if TYPE_CHECKING:
    from myosuite_soma.soma import SomaAnimation

_VERSION = "joint-rbf-contact-3"
_IMPLEMENTATION_SHA256 = hashlib.sha256(
    Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
).hexdigest()


@dataclass(frozen=True)
class JointTarget:
    """An exact source body point or named joint anchor in world metres."""

    body: str | None = None
    joint: str | None = None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FootContactConfig:
    """Bounded local foot morph driven by source collision-plane clearance."""

    ankle: str
    toe: str
    collision_geoms: tuple[str, ...]
    floor_geom: str = "floor"
    radius_m: float = 0.10
    blend_m: float = 0.06
    height_m: float = 0.16
    min_scale: float = 0.20
    clearance_m: float = 0.001
    contact_distance_m: float = 0.025
    release_distance_m: float = 0.09
    temporal_weight: float = 2.0
    smoothing_seconds: float = 0.06


@dataclass(frozen=True)
class GeometryConfig:
    """Explicit bounded postprocessing; omission preserves native SOMA output."""

    landmarks: dict[str, JointTarget] = field(default_factory=dict)
    radius_m: float = 0.35
    max_displacement_m: float = 0.12
    min_segment_scale: float = 0.60
    max_segment_scale: float = 1.50
    min_jacobian: float = 0.05
    regularization: float = 0.001
    max_gradient_norm: float = 0.60
    contacts: tuple[FootContactConfig, ...] = ()
    heldout_landmarks: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GeometryConfig:
        """Parse an explicit JSON geometry section without optional model assets."""
        if not isinstance(raw, dict) or raw.keys() - cls.__dataclass_fields__.keys():
            raise ValueError("Unknown or invalid geometry configuration")
        values = dict(raw)
        mapping = values.get("landmarks", {})
        if not isinstance(mapping, dict):
            raise ValueError("geometry.landmarks must be an object")
        targets = {}
        for name, spec in mapping.items():
            if (
                not isinstance(spec, dict)
                or spec.keys() - JointTarget.__dataclass_fields__.keys()
            ):
                raise ValueError(f"Invalid geometry target: {name}")
            spec = dict(spec)
            if "offset" in spec:
                spec["offset"] = tuple(spec["offset"])
            targets[name] = JointTarget(**spec)
        values["landmarks"] = targets
        if "heldout_landmarks" in values:
            if not isinstance(values["heldout_landmarks"], list):
                raise ValueError("geometry.heldout_landmarks must be a list")
            values["heldout_landmarks"] = tuple(values["heldout_landmarks"])
        contacts = values.get("contacts", [])
        if not isinstance(contacts, list):
            raise ValueError("geometry.contacts must be a list")
        parsed = []
        for spec in contacts:
            if (
                not isinstance(spec, dict)
                or spec.keys() - FootContactConfig.__dataclass_fields__.keys()
            ):
                raise ValueError("Invalid foot contact configuration")
            spec = dict(spec)
            if "collision_geoms" in spec:
                spec["collision_geoms"] = tuple(spec["collision_geoms"])
            try:
                parsed.append(FootContactConfig(**spec))
            except TypeError as exc:
                raise ValueError("Incomplete foot contact configuration") from exc
        values["contacts"] = tuple(parsed)
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject unsafe bounds and ambiguous source targets."""
        if not isinstance(self.landmarks, dict) or len(self.landmarks) < 2:
            raise ValueError("geometry requires at least two named landmark targets")
        if (
            not all(isinstance(name, str) for name in self.heldout_landmarks)
            or len(set(self.heldout_landmarks)) != len(self.heldout_landmarks)
            or any(name not in self.landmarks for name in self.heldout_landmarks)
            or len(self.landmarks) - len(self.heldout_landmarks) < 2
        ):
            raise ValueError(
                "Heldout landmarks must be unique mapped names, leaving at least two controls"
            )
        for name, target in self.landmarks.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(target, JointTarget)
            ):
                raise ValueError("Invalid geometry landmark")
            if (target.body is None) == (target.joint is None):
                raise ValueError(
                    "Each geometry target requires exactly one body or joint"
                )
            for value in (target.body, target.joint):
                if value is not None and (not isinstance(value, str) or not value):
                    raise ValueError("Geometry source names must be nonempty strings")
            offset = np.asarray(target.offset)
            if (
                offset.shape != (3,)
                or offset.dtype.kind not in "fiu"
                or not np.isfinite(offset).all()
            ):
                raise ValueError("Geometry target offset must be a finite three-vector")
            if target.joint is not None and np.any(offset):
                raise ValueError(
                    "Geometry joint-anchor targets cannot have body offsets"
                )
        numbers = (
            self.radius_m,
            self.max_displacement_m,
            self.min_segment_scale,
            self.max_segment_scale,
            self.min_jacobian,
            self.regularization,
            self.max_gradient_norm,
        )
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not np.isfinite(x)
            or x <= 0
            for x in numbers
        ):
            raise ValueError("Geometry bounds must be positive and finite")
        if (
            self.min_segment_scale > 1
            or self.max_segment_scale < 1
            or self.min_jacobian >= 1
            or self.max_gradient_norm >= 1
        ):
            raise ValueError("Geometry bounds must include the identity deformation")
        seen = set()
        for contact in self.contacts:
            if not isinstance(contact, FootContactConfig):
                raise ValueError("Invalid contact config")
            if (
                contact.ankle not in self.landmarks
                or contact.toe not in self.landmarks
                or contact.ankle in seen
            ):
                raise ValueError("Contact requires distinct mapped ankle/toe pairs")
            seen.add(contact.ankle)
            if not contact.collision_geoms or len(set(contact.collision_geoms)) != len(
                contact.collision_geoms
            ):
                raise ValueError("Contact requires unique collision geom names")
            if not all(
                isinstance(n, str) and n
                for n in (*contact.collision_geoms, contact.floor_geom)
            ):
                raise ValueError("Contact geom names must be nonempty")
            positive = (
                contact.radius_m,
                contact.blend_m,
                contact.height_m,
                contact.min_scale,
                contact.contact_distance_m,
                contact.release_distance_m,
            )
            if any(
                isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in positive
            ):
                raise ValueError("Contact dimensions must be positive and finite")
            if (
                not 0 < contact.min_scale <= 1
                or contact.release_distance_m <= contact.contact_distance_m
            ):
                raise ValueError("Invalid contact scale or stance/release distances")
            if (
                not np.isfinite(contact.clearance_m)
                or contact.clearance_m < 0
                or not np.isfinite(contact.temporal_weight)
                or contact.temporal_weight < 0
            ):
                raise ValueError(
                    "Contact margin/temporal weight must be finite and nonnegative"
                )
            if (
                not np.isfinite(contact.smoothing_seconds)
                or contact.smoothing_seconds < 0
            ):
                raise ValueError("Contact smoothing_seconds must be nonnegative")


@dataclass(frozen=True)
class GeometryResult:
    """Deformed animation and corresponding registration diagnostics."""

    animation: SomaAnimation
    joints: np.ndarray
    targets: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ContactMembership:
    """Static anatomical weights, indexed by contact ankle and mesh name."""

    weights: dict[str, dict[str, np.ndarray]]
    metadata: dict[str, Any] = field(default_factory=dict)


def contact_membership_from_predictor(
    predictor: Any, contacts: tuple[FootContactConfig, ...]
) -> ContactMembership:
    """Bind contact to the released Foot/ToeBase LBS groups on every surface.

    Individual muscles use exactly the interpolated weights used by native
    SOMA posing. These static material memberships do not follow world-space
    proximity when a different limb enters a foot's spatial region.
    """
    from myosuite_soma.soma import qpos_digest

    def array(value: Any) -> np.ndarray:
        return np.asarray(
            value.detach().cpu().numpy() if hasattr(value, "detach") else value
        )

    specs = (*predictor.meshes, predictor.skin_mesh)
    matrices = [array(binding[3]) for binding in predictor._bindings] + [
        array(predictor._weights)
    ]
    digest = hashlib.sha256(json.dumps(predictor.joint_names).encode())
    for spec, matrix in zip(specs, matrices):
        if (
            matrix.shape != (len(spec.rest_vertices), len(predictor.joint_names))
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=1), 1, atol=1e-4)
        ):
            raise ValueError("Invalid native LBS contact membership source")
        digest.update(spec.name.encode())
        digest.update(qpos_digest(matrix).encode())
    weights = {}
    groups = {}
    for contact in contacts:
        if (
            contact.ankle not in predictor.joint_names
            or contact.toe not in predictor.joint_names
        ):
            raise ValueError("Contact membership joints are missing from native rig")
        ids = [
            predictor.joint_names.index(contact.ankle),
            predictor.joint_names.index(contact.toe),
        ]
        groups[contact.ankle] = [contact.ankle, contact.toe]
        weights[contact.ankle] = {
            spec.name: np.clip(matrix[:, ids].sum(axis=1), 0, 1).astype(np.float32)
            for spec, matrix in zip(specs, matrices)
        }
    return ContactMembership(
        weights,
        {
            "kind": "native_lbs_group_sum",
            "joint_groups": groups,
            "source_lbs_sha256": digest.hexdigest(),
            "individual_muscle_binding": "native SOMA interpolated LBS weights",
        },
    )


def _validated_membership(
    membership: ContactMembership | None,
    animation: SomaAnimation,
    contacts: tuple[FootContactConfig, ...],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any] | None]:
    from myosuite_soma.soma import qpos_digest

    if not contacts:
        return {}, None
    if not isinstance(membership, ContactMembership):
        raise ValueError(
            "Contact morphology requires explicit static anatomical membership"
        )
    if set(membership.weights) != {contact.ankle for contact in contacts}:
        raise ValueError("Contact membership groups must match configured ankles")
    mesh_names = {mesh.name for mesh in animation.meshes}
    weights = {}
    digest = hashlib.sha256()
    totals = {mesh.name: np.zeros(mesh.points.shape[1]) for mesh in animation.meshes}
    for contact in contacts:
        group = membership.weights[contact.ankle]
        if not isinstance(group, dict) or set(group) != mesh_names:
            raise ValueError(
                "Contact membership must cover exactly every animated mesh"
            )
        weights[contact.ankle] = {}
        digest.update(contact.ankle.encode())
        for mesh in animation.meshes:
            values = np.asarray(group[mesh.name])
            if (
                values.shape != (mesh.points.shape[1],)
                or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values > 1)
            ):
                raise ValueError(
                    "Contact memberships must be finite static per-vertex weights in [0,1]"
                )
            values = np.array(values, dtype=np.float32, copy=True)
            weights[contact.ankle][mesh.name] = values
            totals[mesh.name] += values
            digest.update(mesh.name.encode())
            digest.update(qpos_digest(values).encode())
        skins = [m for m in animation.meshes if m.kind == "skin"]
        if not skins or not np.any(weights[contact.ankle][skins[0].name] > 0):
            raise ValueError("Each contact needs a nonempty anatomical skin membership")
    if any(np.any(total > 1 + 1e-5) for total in totals.values()):
        raise ValueError(
            "Contact anatomical memberships overlap beyond unit total weight"
        )
    metadata = json.loads(json.dumps(membership.metadata, allow_nan=False))
    metadata["weights_sha256"] = digest.hexdigest()
    metadata["scope"] = (
        "Static reference material memberships; no proximity-based anatomical reassignment"
    )
    return weights, metadata


def geometry_model_digest(model: mujoco.MjModel) -> str:
    """Hash source kinematics and geometry inputs used by visual contact fitting."""
    from myosuite_soma.soma import model_digest, qpos_digest

    digest = hashlib.sha256(model_digest(model).encode())
    for value in (
        model.geom_type,
        model.geom_bodyid,
        model.geom_size,
        model.geom_pos,
        model.geom_quat,
        model.geom_contype,
        model.geom_conaffinity,
        model.geom_margin,
        model.geom_gap,
        model.geom_solref,
        model.geom_solimp,
        model.exclude_signature,
        model.pair_geom1,
        model.pair_geom2,
        model.pair_margin,
        model.pair_gap,
    ):
        digest.update(qpos_digest(value).encode())
    return digest.hexdigest()


def deformer_digest() -> str:
    """Hash normalized implementation source for reproducible visual provenance."""
    return _IMPLEMENTATION_SHA256


def validate_geometry_provenance(
    metadata: dict[str, Any], model: mujoco.MjModel | None = None
) -> None:
    """Check deformer/config version and optional source collision fingerprint."""
    if not isinstance(metadata, dict) or metadata.get("version") not in (
        _VERSION,
        "joint-rbf-contact-2",
    ):
        raise ValueError("Unsupported SOMA geometry deformer provenance")
    specification = metadata.get("config")
    if not isinstance(specification, dict):
        raise ValueError("Missing SOMA geometry configuration provenance")
    canonical = json.dumps(specification, sort_keys=True, allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != metadata.get("config_sha256"):
        raise ValueError("SOMA geometry configuration hash does not match")
    config = GeometryConfig.from_dict(json.loads(canonical))
    if metadata["version"] == _VERSION and config.contacts:
        membership = metadata.get("contact_membership")
        if (
            not isinstance(membership, dict)
            or not isinstance(membership.get("weights_sha256"), str)
            or len(membership["weights_sha256"]) != 64
        ):
            raise ValueError("Missing anatomical contact membership provenance")
    if model is not None and metadata.get(
        "source_geometry_model_sha256"
    ) != geometry_model_digest(model):
        raise ValueError(
            "SOMA geometry collision model does not match the recorded rollout"
        )


def source_joint_positions(
    predictor: Any, rotations: np.ndarray, roots: np.ndarray
) -> np.ndarray:
    """Evaluate learned-corrected native joints without changing native predict."""
    from myosuite_soma.soma_model import _skinning_transforms

    torch = predictor._torch
    result = np.empty((len(rotations), len(predictor.joint_names), 3), np.float32)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=predictor._device.type, enabled=False),
    ):
        for start in range(0, len(rotations), predictor._batch_size):
            stop = start + predictor._batch_size
            pose = torch.tensor(
                np.array(rotations[start:stop], dtype=np.float32, order="C"),
                device=predictor._device,
            )
            root = torch.tensor(
                np.array(roots[start:stop], dtype=np.float32, order="C"),
                device=predictor._device,
            )
            transforms = _skinning_transforms(
                torch,
                pose,
                root,
                predictor._rest,
                predictor.parents,
                predictor._rigid(pose),
            )
            joints = (transforms[:, :, :3, :3] @ predictor._rest[..., None])[
                ..., 0
            ] + transforms[:, :, :3, 3]
            result[start:stop] = joints.cpu().numpy()
    return result


def target_joint_positions(
    config: GeometryConfig,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_names: tuple[str, ...],
    source_joints: np.ndarray,
) -> np.ndarray:
    """Read target world points from isolated source-model forward kinematics."""
    from myosuite_soma.soma import _require_qpos_body, _source_poses

    poses = _source_poses(model, qpos, "qpos")
    targets = np.array(source_joints, dtype=float, copy=True)
    if (
        targets.shape != (len(poses), len(joint_names), 3)
        or not np.isfinite(targets).all()
    ):
        raise ValueError(
            "Geometry source joints must match source frames and named rig"
        )
    bindings = []
    for name, target in config.landmarks.items():
        if name not in joint_names:
            raise ValueError(f"Missing geometry target joint {name}")
        kind = (
            mujoco.mjtObj.mjOBJ_BODY
            if target.body is not None
            else mujoco.mjtObj.mjOBJ_JOINT
        )
        index = mujoco.mj_name2id(model, kind, target.body or target.joint)
        if index < 0:
            raise ValueError(
                f"Missing geometry source point {target.body or target.joint}"
            )
        body = index if target.body is not None else int(model.jnt_bodyid[index])
        _require_qpos_body(model, body)
        bindings.append((joint_names.index(name), target, index))
    scratch = mujoco.MjData(model)
    for frame, pose in enumerate(poses):
        scratch.qpos[:] = pose
        mujoco.mj_forward(model, scratch)
        for joint, target, index in bindings:
            targets[frame, joint] = (
                scratch.xpos[index] + scratch.xmat[index].reshape(3, 3) @ target.offset
                if target.body is not None
                else scratch.xanchor[index]
            )
    return targets


def _kernel(points: np.ndarray, nodes: np.ndarray, radius: float) -> np.ndarray:
    from scipy.spatial.distance import cdist

    distance = cdist(points, nodes) / radius
    return np.maximum(1 - distance, 0) ** 4 * (4 * distance + 1)


def _field(
    points: np.ndarray,
    nodes: np.ndarray,
    coefficients: np.ndarray,
    radius: float,
    jacobian: bool = False,
) -> tuple[np.ndarray, float, float]:
    output = np.empty_like(points, dtype=float)
    minimum = 1.0
    maximum_gradient = 0.0
    for begin in range(0, len(points), 4096):
        values = points[begin : begin + 4096]
        output[begin : begin + 4096] = (
            values + _kernel(values, nodes, radius) @ coefficients
        )
        if jacobian:
            delta = values[:, None] - nodes
            distance = np.linalg.norm(delta, axis=2) / radius
            gradient = (
                -20 * np.maximum(1 - distance, 0)[..., None] ** 3 * delta / radius**2
            )
            derivative = np.eye(3)[None] + np.einsum(
                "ni,vnj->vij", coefficients, gradient
            )
            minimum = min(minimum, float(np.linalg.det(derivative).min()))
            maximum_gradient = max(
                maximum_gradient,
                float(np.linalg.norm(derivative - np.eye(3)[None], axis=(1, 2)).max()),
            )
    return output, minimum, maximum_gradient


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def _contact_region(
    points: np.ndarray,
    ankle: np.ndarray,
    toe: np.ndarray,
    normal: np.ndarray,
    config: FootContactConfig,
) -> tuple[np.ndarray, np.ndarray]:
    axis = toe - ankle
    length = float(axis @ axis)
    if length < 1e-8:
        raise ValueError("Contact ankle/toe axis has zero length")
    along = np.clip((points - ankle) @ axis / length, 0, 1)
    centers = ankle + along[:, None] * (toe - ankle)
    delta = points - centers
    height = delta @ normal
    planar = delta - height[:, None] * normal
    radial = 1 - _smoothstep(
        (np.linalg.norm(planar, axis=1) - config.radius_m) / config.blend_m
    )
    vertical = 1 - _smoothstep(np.maximum(height, 0) / config.height_m)
    return radial * vertical, height


def _geom_extent(
    model: mujoco.MjModel, data: mujoco.MjData, index: int, normal: np.ndarray
) -> float:
    axis = data.geom_xmat[index].reshape(3, 3).T @ normal
    size, kind = model.geom_size[index], model.geom_type[index]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(size[0] + abs(axis[2]) * size[1])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return float(np.linalg.norm(axis * size))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(axis) @ size)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(np.linalg.norm(axis[:2]) * size[0] + abs(axis[2]) * size[1])
    raise ValueError("Contact collision bounds support primitive source geoms only")


def _contact_inputs(
    model: mujoco.MjModel, qpos: np.ndarray, config: FootContactConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from myosuite_soma.soma import _require_qpos_body

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config.floor_geom)
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in config.collision_geoms
    ]
    if floor < 0 or any(index < 0 for index in ids):
        raise ValueError("Missing configured contact geometry")
    if (
        model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE
        or model.geom_bodyid[floor] != 0
    ):
        raise ValueError("Visual contact requires an explicit static source plane")
    if any(
        not (model.geom_contype[index] or model.geom_conaffinity[index])
        for index in ids
    ):
        raise ValueError(
            "Configured contact geoms must participate in source collisions"
        )
    if any(
        not (
            (model.geom_contype[index] & model.geom_conaffinity[floor])
            or (model.geom_contype[floor] & model.geom_conaffinity[index])
        )
        for index in ids
    ):
        raise ValueError("Configured foot/floor collision masks do not interact")
    excluded = set(model.exclude_signature.tolist())
    if any(int(model.geom_bodyid[index]) in excluded for index in ids):
        raise ValueError(
            "Configured foot body is explicitly excluded from the world floor"
        )
    for index in ids:
        _require_qpos_body(model, int(model.geom_bodyid[index]))
    scratch = mujoco.MjData(model)
    normals, origins, clearance = [], [], []
    for pose in qpos:
        scratch.qpos[:] = pose
        mujoco.mj_forward(model, scratch)
        normal = scratch.geom_xmat[floor].reshape(3, 3)[:, 2].copy()
        origin = scratch.geom_xpos[floor].copy()
        lower = min(
            float((scratch.geom_xpos[index] - origin) @ normal)
            - _geom_extent(model, scratch, index, normal)
            for index in ids
        )
        normals.append(normal)
        origins.append(origin)
        clearance.append(lower)
    return np.asarray(normals), np.asarray(origins), np.asarray(clearance)


def _temporal_scales(
    upper: np.ndarray,
    weight: float,
    times: np.ndarray,
    smoothing_seconds: float,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    from scipy.optimize import lsq_linear
    from scipy.sparse import diags, eye, vstack

    if (
        upper.ndim != 1
        or np.shape(times) != upper.shape
        or not np.isfinite(upper).all()
        or np.any(upper < 0)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(weight)
        or weight < 0
        or not np.isfinite(smoothing_seconds)
        or smoothing_seconds < 0
    ):
        raise ValueError(
            "Temporal bounds/times must be finite nonnegative increasing samples"
        )

    if weight == 0 or len(upper) < 3:
        if diagnostics is not None:
            diagnostics.update(method="identity", converged=True, attempts=[])
        return upper.copy()
    count = len(upper)
    before, after = np.diff(times)[:-1], np.diff(times)[1:]
    second = diags(
        [
            2 / (before * (before + after)),
            -2 / (before * after),
            2 / (after * (before + after)),
        ],
        offsets=[0, 1, 2],
        shape=(count - 2, count),
        format="csr",
    )
    matrix = vstack(
        (eye(count, format="csr"), np.sqrt(weight) * smoothing_seconds**2 * second),
        format="csr",
    )
    values = np.r_[np.ones(count), np.zeros(count - 2)]
    design = matrix @ diags(upper, format="csr")
    result = lsq_linear(
        design,
        values,
        bounds=(0.0, 1.0),
        tol=1e-10,
    )
    attempts = [
        {
            "method": "sparse_trf",
            "converged": bool(result.success),
            "status": int(result.status),
        }
    ]
    method = "sparse_trf"
    if not result.success and count < 1000:
        result = lsq_linear(
            design.toarray(), values, bounds=(0.0, 1.0), method="bvls", tol=1e-10
        )
        method = "dense_bvls"
        attempts.append(
            {
                "method": method,
                "converged": bool(result.success),
                "status": int(result.status),
            }
        )
    if not result.success:
        import osqp

        solver = osqp.OSQP()
        solver.setup(
            P=(design.T @ design).tocsc(),
            q=np.asarray(-design.T @ values),
            A=eye(count, format="csc"),
            l=np.zeros(count),
            u=np.ones(count),
            verbose=False,
            polishing=True,
            eps_abs=1e-9,
            eps_rel=1e-9,
            max_iter=50000,
        )
        solver.warm_start(x=np.clip(result.x, 0.0, 1.0))
        fitted = solver.solve(raise_error=False)
        result = SimpleNamespace(
            x=fitted.x,
            success=fitted.info.status_val == 1,
            status=fitted.info.status_val,
            nit=fitted.info.iter,
        )
        method = "sparse_osqp"
        attempts.append(
            {
                "method": method,
                "converged": bool(result.success),
                "status": int(result.status),
            }
        )
    if not np.isfinite(result.x).all():
        raise RuntimeError("Temporal fit produced nonfinite scales")
    solution = np.minimum(upper, np.maximum(0.0, result.x) * upper)
    if diagnostics is not None:
        residual = matrix @ solution - values
        diagnostics.update(
            method=method,
            converged=bool(result.success),
            status=int(result.status),
            iterations=int(result.nit),
            objective=0.5 * float(residual @ residual),
            attempts=attempts,
            scope="Box-feasible temporal regularization; convergence status does not certify anatomical fit",
        )
    return solution


def deform_animation(
    animation: SomaAnimation,
    *,
    source_joints: np.ndarray,
    joint_names: tuple[str, ...],
    parents: np.ndarray,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    config: GeometryConfig,
    contact_membership: ContactMembership | None = None,
) -> GeometryResult:
    """Apply the same smooth registration/contact field to all visual surfaces.

    Args:
        animation: Native world-space SOMA output; never mutated.
        source_joints: Actual learned-corrected native world joints, ``(T,J,3)``.
        joint_names: Ordering of the native rig.
        parents: Native root-first parent indices.
        model: Original source model, read only.
        qpos: Positions corresponding exactly to the animation times.
        config: Explicit geometry targets and bounds.
        contact_membership: Static reference anatomical weights, required when
            contacts are enabled; generated from native SOMA LBS in production.

    Returns:
        New animation and native/target joint diagnostics. Contact is driven by
        geometric source clearance, not estimated force or simulated tissue.
    """
    from myosuite_soma.soma import SomaAnimation, qpos_digest, validate_animation

    config.validate()
    validate_animation(animation)
    memberships, membership_metadata = _validated_membership(
        contact_membership, animation, config.contacts
    )
    source = np.asarray(source_joints, dtype=float)
    if (
        not joint_names
        or len(set(joint_names)) != len(joint_names)
        or not all(isinstance(name, str) and name for name in joint_names)
    ):
        raise ValueError("Geometry joint names must be unique nonempty strings")
    if (
        source.shape != (len(animation.time), len(joint_names), 3)
        or not np.isfinite(source).all()
    ):
        raise ValueError("source_joints must be finite (T,J,3) positions")
    parents = np.asarray(parents)
    if (
        parents.shape != (len(joint_names),)
        or parents.dtype.kind not in "iu"
        or parents[0] != -1
        or any(p < 0 or p >= j for j, p in enumerate(parents[1:], 1))
    ):
        raise ValueError("Geometry rig must have root-first integer parents")
    if len(qpos) != len(source):
        raise ValueError("Geometry qpos/frame counts differ")
    targets = target_joint_positions(config, model, qpos, tuple(joint_names), source)
    evaluated = [joint_names.index(name) for name in config.landmarks]
    active = [
        joint_names.index(name)
        for name in config.landmarks
        if name not in config.heldout_landmarks
    ]
    pairs = []
    for child in active:
        parent = int(parents[child])
        while parent >= 0 and parent not in active:
            parent = int(parents[parent])
        if parent >= 0:
            pairs.append((parent, child))
    ratios = []
    for parent, child in pairs:
        length = np.linalg.norm(source[:, child] - source[:, parent], axis=1)
        if np.any(length < 1e-6):
            raise ValueError("Native geometry segment has zero length")
        ratios.extend(
            (
                np.linalg.norm(targets[:, child] - targets[:, parent], axis=1) / length
            ).tolist()
        )
    if ratios and (
        min(ratios) < config.min_segment_scale or max(ratios) > config.max_segment_scale
    ):
        raise ValueError("Requested registration exceeds bounded segment morphology")
    meshes = [
        replace(mesh, points=np.empty_like(mesh.points, dtype=np.float32))
        for mesh in animation.meshes
    ]
    sizes = [len(mesh.points[0]) for mesh in meshes]
    offsets = np.r_[0, np.cumsum(sizes)]
    moved_joints = np.empty_like(source)
    minimum_jacobian = 1.0
    maximum_displacement = 0.0
    strengths = []
    gradient_norms = []
    for frame in range(len(source)):
        nodes = list(source[frame, active])
        moved = list(targets[frame, active])
        for parent, child in pairs:
            nodes.append((source[frame, parent] + source[frame, child]) / 2)
            moved.append((targets[frame, parent] + targets[frame, child]) / 2)
        nodes, moved = np.asarray(nodes), np.asarray(moved)
        if len(np.unique(np.round(nodes, decimals=8), axis=0)) != len(nodes):
            raise ValueError("Geometry interpolation nodes coincide")
        coefficient = np.linalg.solve(
            _kernel(nodes, nodes, config.radius_m)
            + np.eye(len(nodes)) * config.regularization,
            moved - nodes,
        )
        points = np.concatenate([mesh.points[frame] for mesh in animation.meshes])
        mapped, jacobian, gradient_norm = _field(
            points, nodes, coefficient, config.radius_m, True
        )
        displacement = float(np.linalg.norm(mapped - points, axis=1).max())
        safe_gradient = min(
            config.max_gradient_norm, 1 - config.min_jacobian ** (1 / 3) - 1e-6
        )
        strength = min(
            1.0,
            config.max_displacement_m / max(displacement, 1e-12),
            safe_gradient / max(gradient_norm, 1e-12),
        )
        strengths.append(strength)
        gradient_norms.append(gradient_norm)
        if strength < 1:
            mapped = points + strength * (mapped - points)
            coefficient *= strength
            # The Frobenius norm bounds every sampled singular displacement.
            jacobian = (1 - strength * gradient_norm) ** 3
        minimum_jacobian = min(minimum_jacobian, jacobian)
        maximum_displacement = max(maximum_displacement, displacement * strength)
        for mesh, begin, end in zip(meshes, offsets[:-1], offsets[1:]):
            mesh.points[frame] = mapped[begin:end]
        moved_joints[frame], _, _ = _field(
            source[frame], nodes, coefficient, config.radius_m
        )
    registration_temporal = {}
    smoothed_strengths = _temporal_scales(
        np.asarray(strengths),
        1.0,
        np.asarray(animation.time, dtype=float),
        0.06,
        diagnostics=registration_temporal,
    )
    if np.any(smoothed_strengths < 0) or np.any(
        smoothed_strengths > np.asarray(strengths)
    ):
        raise RuntimeError("Registration smoothing exceeded actual field bounds")
    effective_gradient = smoothed_strengths * np.asarray(gradient_norms)
    minimum_jacobian = float(np.min((1 - effective_gradient) ** 3))
    if minimum_jacobian < config.min_jacobian:
        raise ValueError("Smoothed registration exceeds its Jacobian bound")
    maximum_displacement = 0.0
    for frame, (strength, smoothed) in enumerate(zip(strengths, smoothed_strengths)):
        ratio = smoothed / strength
        moved_joints[frame] = source[frame] + ratio * (
            moved_joints[frame] - source[frame]
        )
        for mesh, native in zip(meshes, animation.meshes):
            mesh.points[frame] = native.points[frame] + ratio * (
                mesh.points[frame] - native.points[frame]
            )
            maximum_displacement = max(
                maximum_displacement,
                float(
                    np.linalg.norm(
                        mesh.points[frame].astype(float) - native.points[frame], axis=1
                    ).max()
                ),
            )
        if maximum_displacement > config.max_displacement_m + 1e-7:
            raise ValueError(
                "Smoothed registration exceeds its actual displacement bound"
            )
    contact_reports = []
    contact_context = []
    skin = next((mesh for mesh in meshes if mesh.kind == "skin"), None)
    skin_before_contact = (
        None if skin is None else skin.points[..., 2].min(axis=1).copy()
    )
    if config.contacts and skin is None:
        raise ValueError("Contact morphology requires the native skin surface")
    for contact in config.contacts:
        ankle, toe = joint_names.index(contact.ankle), joint_names.index(contact.toe)
        normals, origins, native = _contact_inputs(model, qpos, contact)
        upper = np.ones(len(source))
        before = []
        support_before = []
        for frame in range(len(source)):
            points = skin.points[frame].astype(float)
            weight, height = _contact_region(
                points,
                moved_joints[frame, ankle],
                moved_joints[frame, toe],
                normals[frame],
                contact,
            )
            selected = (weight > 0.95) & (memberships[contact.ankle][skin.name] > 1e-6)
            weight *= memberships[contact.ankle][skin.name]
            distance = (points - origins[frame]) @ normals[frame]
            support_before.append(
                float(distance[weight > 1e-6].min()) if np.any(weight > 1e-6) else None
            )
            before.append(
                float(distance[selected].min())
                if np.any(selected)
                else float(distance.min())
            )
            need = (native[frame] + contact.clearance_m - distance) / (
                np.maximum(-height * weight, 1e-12)
            )
            required = max(0.0, float(need[(height < 0) & selected].max(initial=0)))
            stance = 1 - float(
                _smoothstep(
                    np.array(
                        (native[frame] - contact.contact_distance_m)
                        / (contact.release_distance_m - contact.contact_distance_m)
                    )
                )
            )
            upper[frame] = 1 - stance * min(required, 1 - contact.min_scale)
        contact_temporal = {}
        scales = np.maximum(
            contact.min_scale,
            _temporal_scales(
                upper,
                contact.temporal_weight,
                np.asarray(animation.time, dtype=float),
                contact.smoothing_seconds,
                diagnostics=contact_temporal,
            ),
        )
        after = []
        for frame in range(len(source)):
            for mesh in meshes:
                points = mesh.points[frame].astype(float)
                weight, height = _contact_region(
                    points,
                    moved_joints[frame, ankle],
                    moved_joints[frame, toe],
                    normals[frame],
                    contact,
                )
                weight *= memberships[contact.ankle][mesh.name]
                mesh.points[frame] = (
                    points
                    + ((scales[frame] - 1) * weight * height)[:, None] * normals[frame]
                )
            weight, _ = _contact_region(
                skin.points[frame],
                moved_joints[frame, ankle],
                moved_joints[frame, toe],
                normals[frame],
                contact,
            )
            selected = (weight > 0.95) & (memberships[contact.ankle][skin.name] > 1e-6)
            weight *= memberships[contact.ankle][skin.name]
            distance = (skin.points[frame] - origins[frame]) @ normals[frame]
            after.append(
                float(distance[selected].min())
                if np.any(selected)
                else float(distance.min())
            )
        contact_reports.append(
            {
                "ankle": contact.ankle,
                "source_clearance_m": native.tolist(),
                "visual_before_m": before,
                "full_support_before_m": support_before,
                "visual_after_m": after,
                "compression_scale": scales.tolist(),
                "temporal_solver": contact_temporal,
                "maximum_added_penetration_before_m": float(
                    np.maximum(native - before, 0).max()
                ),
                "maximum_added_penetration_after_m": float(
                    np.maximum(native - after, 0).max()
                ),
                "contact_driver": "source_primitive_plane_clearance",
                "minimum_normal_scale_with_membership_held_fixed": float(scales.min()),
                "contact_jacobian_note": "Static material masks vary over the surface; this scalar is not a full spatial Jacobian or injectivity certificate. Check sampled triangle quality separately.",
            }
        )
        contact_context.append((contact, ankle, toe, normals, origins, native))
    for report, (contact, ankle, toe, normals, origins, native) in zip(
        contact_reports, contact_context
    ):
        support_after = []
        for frame in range(len(source)):
            weight, _ = _contact_region(
                skin.points[frame],
                moved_joints[frame, ankle],
                moved_joints[frame, toe],
                normals[frame],
                contact,
            )
            weight *= memberships[contact.ankle][skin.name]
            distance = (skin.points[frame] - origins[frame]) @ normals[frame]
            support_after.append(
                float(distance[weight > 1e-6].min()) if np.any(weight > 1e-6) else None
            )
        report["full_support_after_m"] = support_after
        usable = np.array([value is not None for value in support_after])
        report["maximum_added_full_support_penetration_after_m"] = float(
            np.maximum(
                native[usable] - np.asarray(support_after, dtype=float)[usable], 0
            ).max(initial=0)
        )
    error_before = np.linalg.norm(source[:, evaluated] - targets[:, evaluated], axis=2)
    error_after = np.linalg.norm(
        moved_joints[:, evaluated] - targets[:, evaluated], axis=2
    )
    final_displacement = max(
        float(np.linalg.norm(mesh.points.astype(float) - native.points, axis=2).max())
        for mesh, native in zip(meshes, animation.meshes)
    )
    if final_displacement > config.max_displacement_m:
        raise ValueError("Combined geometry morphology exceeds max_displacement_m")
    specification = asdict(config)
    diagnostics = {
        "version": _VERSION,
        "deformer_sha256": deformer_digest(),
        "config": specification,
        "config_sha256": hashlib.sha256(
            json.dumps(specification, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "source_geometry_model_sha256": geometry_model_digest(model),
        "source_joints_sha256": qpos_digest(source),
        "contact_membership": membership_metadata,
        "landmark_rmse_before_m": float(np.sqrt(np.mean(error_before**2))),
        "landmark_rmse_after_m": float(np.sqrt(np.mean(error_after**2))),
        "per_landmark_rmse_before_m": {
            joint_names[index]: float(np.sqrt(np.mean(error_before[:, column] ** 2)))
            for column, index in enumerate(evaluated)
        },
        "per_landmark_rmse_after_m": {
            joint_names[index]: float(np.sqrt(np.mean(error_after[:, column] ** 2)))
            for column, index in enumerate(evaluated)
        },
        "heldout_landmark_names": list(config.heldout_landmarks),
        "fit_control_names": [joint_names[index] for index in active],
        "landmark_metric_scope": "source anchor agreement; fit controls are interpolation constraints, not independent anatomy measurements",
        "maximum_displacement_m": maximum_displacement,
        "registration_strength": smoothed_strengths.tolist(),
        "registration_temporal_solver": registration_temporal,
        "registration_strength_range": [
            float(smoothed_strengths.min()),
            float(smoothed_strengths.max()),
        ],
        "bounded_field_note": "Regularized controls; globally reduced field amplitude where sampled displacement/Jacobian bounds require it. Positive sampled Jacobians are not a proof of global injectivity.",
        "maximum_combined_displacement_m": final_displacement,
        "minimum_sampled_registration_jacobian": minimum_jacobian,
        "registration_jacobian_statistic": "Conservative lower bound from sampled displacement-gradient Frobenius norms, rechecked after temporal regularization",
        "segment_scale_range": [min(ratios), max(ratios)] if ratios else [1.0, 1.0],
        "contacts": contact_reports,
        "skin_world_z_min_before_contact_m": None
        if skin_before_contact is None
        else skin_before_contact.tolist(),
        "skin_world_z_min_after_contact_m": None
        if skin is None
        else skin.points[..., 2].min(axis=1).tolist(),
        "scope": "visual registration and local morphology; source mechanics unchanged",
    }
    output = SomaAnimation(
        np.array(animation.time, copy=True),
        tuple(meshes),
        {**animation.metadata, "geometry": diagnostics},
    )
    validate_animation(output)
    return GeometryResult(output, moved_joints, targets, diagnostics)
