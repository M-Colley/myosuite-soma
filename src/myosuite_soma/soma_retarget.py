# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0.
# Extracted from the audited MyoSuite SOMA integration; imports adapted.
# Source attribution: THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Bounded recorded-pose IK and quaternion-aware kinematic replay utilities.

This converts a recorded rig into source-model poses. It does not infer muscle
activation, simulate a controller, or change source mechanical parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from myosuite_soma.soma import SomaConfig, _source_poses, retarget_rotations
from myosuite_soma.soma_geometry import GeometryConfig, JointTarget


def _real_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} requires real numeric values")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def decode_sixd(values: np.ndarray, *, packing: str) -> np.ndarray:
    """Decode explicitly interleaved or concatenated first-two-column rotations.

    Original SKIM ``pose6d`` uses interleaved ``R[:, :2].reshape(6)``.
    Training ``pose`` uses concatenated columns and a separate root convention.
    No axis or heading transformation is inferred by this decoder.
    """
    values = _real_array(values, "6D rotations")
    if values.shape[-1:] != (6,) or not np.isfinite(values).all():
        raise ValueError("6D rotations must be finite with final dimension 6")
    if packing == "interleaved":
        columns = values.reshape(*values.shape[:-1], 3, 2)
        x, y = columns[..., 0], columns[..., 1]
    elif packing == "concatenated":
        x, y = values[..., :3], values[..., 3:]
    else:
        raise ValueError("packing must be interleaved or concatenated")
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(length < 1e-8):
        raise ValueError("Degenerate first rotation column")
    x = x / length
    z = np.cross(x, y)
    length = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(length < 1e-8):
        raise ValueError("Collinear rotation columns")
    z /= length
    return np.stack((x, np.cross(z, x), z), axis=-1)


def rig_forward_kinematics(
    local: np.ndarray, roots: np.ndarray, rest: np.ndarray, parents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate recorded local rotations and root positions in their native frame."""
    local, roots, rest = (
        _real_array(value, name)
        for value, name in (
            (local, "local rotations"),
            (roots, "roots"),
            (rest, "rest joints"),
        )
    )
    parents = np.asarray(parents)
    if parents.ndim != 1:
        raise ValueError("Rig parents must be a one-dimensional array")
    joints = len(parents)
    if (
        local.ndim != 4
        or local.shape[1:] != (joints, 3, 3)
        or roots.shape != (len(local), 3)
        or rest.shape != (joints, 3)
        or parents.dtype.kind not in "iu"
        or joints == 0
        or parents[0] != -1
        or any(not 0 <= int(parents[j]) < j for j in range(1, joints))
        or not all(np.isfinite(x).all() for x in (local, roots, rest))
    ):
        raise ValueError("Invalid recorded rig arrays or parent order")
    if not np.allclose(
        local.swapaxes(-1, -2) @ local, np.eye(3), atol=1e-5
    ) or not np.allclose(np.linalg.det(local), 1, atol=1e-5):
        raise ValueError("Recorded local matrices must be proper rotations")
    rotations, positions = [], []
    for joint, parent in enumerate(parents):
        rotations.append(
            local[:, joint] if parent < 0 else rotations[parent] @ local[:, joint]
        )
        positions.append(
            roots
            if parent < 0
            else positions[parent]
            + np.einsum("tij,j->ti", rotations[parent], rest[joint] - rest[parent])
        )
    return np.stack(rotations, 1), np.stack(positions, 1)


def decode_recording(
    values: np.ndarray, *, packing: str, root_convention: str
) -> np.ndarray:
    """Decode native poses or explicitly undo the SKIM training root heading.

    Native root positions must be provided independently for training pose files.
    The training Ry90 heading differs from the profile's output-axis rotation.
    """
    rotations = decode_sixd(values, packing=packing)
    if rotations.ndim != 4:
        raise ValueError("Recording must have shape (frames, joints, 6)")
    if root_convention == "skim-training-ry90":
        rotations[:, 0] = (
            Rotation.from_euler("y", -90, degrees=True).as_matrix() @ rotations[:, 0]
        )
    elif root_convention != "native":
        raise ValueError("Unknown recording root convention")
    return rotations


def differentiate_trajectory(
    model: mujoco.MjModel, qpos: np.ndarray, time: np.ndarray
) -> np.ndarray:
    """Differentiate source poses with MuJoCo quaternion conventions."""
    qpos = _source_poses(model, qpos, "qpos")
    time = _real_array(time, "time")
    if (
        time.shape != (len(qpos),)
        or len(qpos) < 2
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Trajectory requires at least two strictly increasing times")
    velocity = np.empty((len(qpos), model.nv))
    for index, dt in enumerate(np.diff(time)):
        mujoco.mj_differentiatePos(
            model, velocity[index], float(dt), qpos[index], qpos[index + 1]
        )
    velocity[-1] = velocity[-2]
    return velocity


def physical_parameter_digest(model: mujoco.MjModel) -> str:
    """Fingerprint source mechanical constants independently of pose state."""
    digest = hashlib.sha256()
    for name in (
        "body_mass",
        "body_inertia",
        "dof_damping",
        "dof_armature",
        "geom_friction",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_dynprm",
        "actuator_lengthrange",
        "tendon_stiffness",
    ):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(getattr(model, name)).tobytes())
    return digest.hexdigest()


class JointCouplings:
    """Topological equalities and a feasible scalar domain nearest neutral.

    Disconnected polynomial intervals select the one nearest the neutral pose.
    Compiled source bounds remain unchanged.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        active = [
            e
            for e in range(model.neq)
            if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT)
            and model.eq_active0[e]
        ]
        pending = {int(model.eq_obj1id[e]): e for e in active}
        if len(pending) != len(active):
            raise ValueError("Multiple active equalities drive the same joint")
        self.order: list[int] = []
        while pending:
            ready = [
                joint
                for joint, eq in pending.items()
                if int(model.eq_obj2id[eq]) not in pending
            ]
            if not ready:
                raise ValueError("Cyclic joint equality constraints")
            self.order.extend(pending.pop(joint) for joint in ready)
        dependent = {int(model.eq_obj1id[e]) for e in self.order}
        self.joints = [
            j
            for j in range(model.njnt)
            if int(model.jnt_type[j])
            in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
            and j not in dependent
        ]
        if not self.joints or not np.all(model.jnt_limited[self.joints]):
            raise ValueError(
                "Independent scalar joints require explicit compiled limits"
            )
        self.address = model.jnt_qposadr[self.joints].copy()
        self.bounds = model.jnt_range[self.joints].T.copy()
        self._restrict_limits()

    def _restrict_limits(self) -> None:
        expressions = {joint: (joint, Polynomial([0.0, 1.0])) for joint in self.joints}
        intervals = {
            joint: [tuple(self.bounds[:, index])]
            for index, joint in enumerate(self.joints)
        }
        for equality in self.order:
            child, parent = (
                int(self.model.eq_obj1id[equality]),
                int(self.model.eq_obj2id[equality]),
            )
            if int(self.model.jnt_type[child]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError("Joint equality requires scalar joints")
            if parent >= 0 and parent not in expressions:
                raise ValueError("Joint equality parent must be scalar")
            base, polynomial = (
                expressions[parent] if parent >= 0 else (-1, Polynomial([0.0]))
            )
            parent_reference = (
                self.model.qpos0[int(self.model.jnt_qposadr[parent])]
                if parent >= 0
                else 0.0
            )
            child_reference = self.model.qpos0[int(self.model.jnt_qposadr[child])]
            expression = child_reference + Polynomial(self.model.eq_data[equality, :5])(
                polynomial - parent_reference
            )
            expressions[child] = base, expression
            if not self.model.jnt_limited[child]:
                continue
            lower, upper = self.model.jnt_range[child]
            if base < 0:
                if not lower <= expression(0.0) <= upper:
                    raise ValueError("Constant joint equality violates its joint limit")
                continue
            feasible = []
            for low, high in intervals[base]:
                cuts = [low, high]
                for target in (lower, upper):
                    cuts.extend(
                        float(root.real)
                        for root in (expression - target).roots()
                        if abs(root.imag) < 1e-8 and low < root.real < high
                    )
                cuts = sorted(set(cuts))
                feasible.extend(
                    (a, b)
                    for a, b in zip(cuts[:-1], cuts[1:])
                    if lower - 1e-12 <= expression((a + b) / 2) <= upper + 1e-12
                )
            if not feasible:
                raise ValueError("Coupled joint limits have no feasible interval")
            intervals[base] = feasible
        for index, base in enumerate(self.joints):
            neutral = self.model.qpos0[self.address[index]]
            self.bounds[:, index] = min(
                intervals[base], key=lambda pair: abs(np.clip(neutral, *pair) - neutral)
            )

    def apply(self, qpos: np.ndarray) -> np.ndarray:
        """Return a copy satisfying active joint equality polynomials."""
        pose = _real_array(qpos, "qpos").copy()
        if pose.shape != (self.model.nq,) or not np.isfinite(pose).all():
            raise ValueError("Invalid source qpos")
        for equality in self.order:
            child, parent = (
                int(self.model.eq_obj1id[equality]),
                int(self.model.eq_obj2id[equality]),
            )
            parent_address = (
                int(self.model.jnt_qposadr[parent]) if parent >= 0 else None
            )
            value = (
                pose[parent_address] - self.model.qpos0[parent_address]
                if parent_address is not None
                else 0.0
            )
            child_address = int(self.model.jnt_qposadr[child])
            pose[child_address] = self.model.qpos0[child_address] + Polynomial(
                self.model.eq_data[equality, :5]
            )(value)
        return pose


@dataclass(frozen=True)
class IKConfig:
    """Explicit source anchors, orientation weights and bounded solver seeds."""

    landmarks: dict[str, JointTarget]
    orientation_weights: dict[str, float]
    initial_guesses: tuple[dict[str, float], ...] = ()
    temporal_weight: float = 0.002
    neutral_weight: float = 0.0005
    max_nfev: int = 45

    @classmethod
    def from_json(cls, path: str | Path) -> IKConfig:
        """Read data-only configuration; unknown keys fail explicitly."""
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict) or raw.keys() - cls.__dataclass_fields__.keys():
            raise ValueError("Invalid IK configuration fields")
        values = dict(raw)
        values["landmarks"] = GeometryConfig.from_dict(
            {"landmarks": raw.get("landmarks", {})}
        ).landmarks
        values["initial_guesses"] = tuple(raw.get("initial_guesses", ()))
        result = cls(**values)
        if not isinstance(result.orientation_weights, dict):
            raise ValueError("orientation_weights must map exact rig names to weights")
        weights = list(result.orientation_weights.values()) + [
            result.temporal_weight,
            result.neutral_weight,
        ]
        if (
            not result.landmarks
            or not result.orientation_weights
            or any(
                not isinstance(w, (float, int)) or not np.isfinite(w) or w < 0
                for w in weights
            )
            or not isinstance(result.max_nfev, int)
            or isinstance(result.max_nfev, bool)
            or result.max_nfev <= 0
        ):
            raise ValueError(
                "IK requires explicit anchors, finite weights and positive max_nfev"
            )
        return result


class SomaIKRetargeter:
    """Retarget recordings to an unchanged model using a calibrated SOMA profile.

    The profile must use a world-parented free root, matching Hips/root anchors,
    and globally consistent rotation bases. Unsupported profiles fail explicitly.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        profile: SomaConfig,
        predictor: Any,
        config: IKConfig,
    ) -> None:
        self.model, self.profile, self.predictor, self.config = (
            model,
            profile,
            predictor,
            config,
        )
        self.names, self.parents, self.rest = (
            predictor.joint_names,
            predictor.parents,
            predictor.rest_joints,
        )
        if (
            profile.rest_qpos is None
            or profile.root_body != profile.anchor_body
            or profile.anchor_joint != self.names[0]
            or np.any(profile.output_translation)
        ):
            raise ValueError(
                "IK requires explicit rest_qpos, matching root/Hips anchors and zero artistic output_translation"
            )
        self.reference = _source_poses(
            model, np.asarray(profile.rest_qpos)[None], "rest_qpos"
        )[0]
        self.domain = JointCouplings(model)
        self.reference = self.domain.apply(self.reference)
        self.neutral = self.reference[self.domain.address]
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = self.reference
        mujoco.mj_forward(model, self.data)
        self.root_id = model.body(profile.root_body).id
        free = [
            j
            for j in range(model.njnt)
            if model.jnt_bodyid[j] == self.root_id
            and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(free) != 1 or model.body_parentid[self.root_id] != 0:
            raise ValueError("IK root must contain one world-parented free joint")
        self.root_address = int(model.jnt_qposadr[free[0]])
        local = retarget_rotations(
            profile, model, self.reference[None], self.names, self.parents
        )
        self.global_ref = rig_forward_kinematics(
            local, self.rest[None, 0], self.rest, self.parents
        )[0][0]
        self.root_reference = self.data.xmat[self.root_id].reshape(3, 3).copy()
        self.orientation_joints = [
            self.names.index(name) for name in config.orientation_weights
        ]
        self.orientation_ids, self.orientation_reference = [], []
        for joint in self.orientation_joints:
            mapping = profile.body_map.get(self.names[joint])
            if mapping is None:
                raise ValueError(
                    f"IK orientation lacks explicit body mapping: {self.names[joint]}"
                )
            parent = int(self.parents[joint])
            target_parent = np.eye(3) if parent < 0 else self.global_ref[parent]
            source_parent = self.data.xmat[model.body(mapping.parent_body).id].reshape(
                3, 3
            )
            expected = target_parent.T @ profile.output_rotation.T @ source_parent
            if not np.allclose(profile.basis @ mapping.basis, expected, atol=1e-6):
                raise ValueError(
                    "IK inverse requires globally consistent body-map bases"
                )
            body = model.body(mapping.body).id
            self.orientation_ids.append(body)
            self.orientation_reference.append(self.data.xmat[body].reshape(3, 3).copy())
        self.orientation_reference = np.array(self.orientation_reference)
        self.anchor_joints, self.bindings = [], []
        for name, target in config.landmarks.items():
            if (
                (target.body is None) == (target.joint is None)
                or not np.isfinite(target.offset).all()
                or len(target.offset) != 3
            ):
                raise ValueError(
                    "IK landmark requires exactly one source body or joint and a finite offset"
                )
            self.anchor_joints.append(self.names.index(name))
            obj = mujoco.mjtObj.mjOBJ_BODY if target.body else mujoco.mjtObj.mjOBJ_JOINT
            index = mujoco.mj_name2id(model, obj, target.body or target.joint)
            if index < 0:
                raise ValueError(f"Missing source IK anchor: {target}")
            body = index if target.body else int(model.jnt_bodyid[index])
            self.bindings.append((target, index, body))
        self.before = physical_parameter_digest(model)

    def _anchors(self) -> np.ndarray:
        return np.array(
            [
                (self.data.xpos[index] if target.body else self.data.xanchor[index])
                + self.data.xmat[body].reshape(3, 3) @ target.offset
                for target, index, body in self.bindings
            ]
        )

    def solve(
        self,
        rotations: np.ndarray,
        positions: np.ndarray,
        previous: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit one global recorded pose; positions remain in native rig metres."""
        rotations, positions = (
            _real_array(rotations, "global rotations"),
            _real_array(positions, "global positions"),
        )
        if (
            rotations.shape != (len(self.names), 3, 3)
            or positions.shape != (len(self.names), 3)
            or not np.isfinite(rotations).all()
            or not np.isfinite(positions).all()
        ):
            raise ValueError("Recorded pose does not match predictor rig")
        if not np.allclose(
            rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-5
        ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
            raise ValueError("Recorded global matrices must be proper rotations")
        output = self.profile.output_rotation
        desired = self.profile.scale * (positions[self.anchor_joints] @ output.T)
        target_rotation = (
            output
            @ rotations[self.orientation_joints]
            @ self.global_ref[self.orientation_joints].swapaxes(-1, -2)
            @ output.T
            @ self.orientation_reference
        )
        root_rotation = (
            output
            @ rotations[0]
            @ self.global_ref[0].T
            @ output.T
            @ self.root_reference
        )
        root_position = (
            self.profile.scale * (output @ positions[0])
            - root_rotation @ self.profile.root_offset
        )
        prior = (
            self.neutral
            if previous is None
            else _source_poses(self.model, np.asarray(previous)[None], "previous")[
                0, self.domain.address
            ]
        )
        weights = np.array(list(self.config.orientation_weights.values()))[:, None]

        def pose_at(values: np.ndarray) -> np.ndarray:
            pose = self.reference.copy()
            pose[self.root_address : self.root_address + 3] = root_position
            mujoco.mju_mat2Quat(
                pose[self.root_address + 3 : self.root_address + 7],
                root_rotation.reshape(-1),
            )
            pose[self.domain.address] = values
            return self.domain.apply(pose)

        def objective(values: np.ndarray) -> np.ndarray:
            self.data.qpos[:] = pose_at(values)
            mujoco.mj_forward(self.model, self.data)
            orientation = self.data.xmat[self.orientation_ids].reshape(-1, 3, 3)
            angles = Rotation.from_matrix(
                target_rotation.swapaxes(-1, -2) @ orientation
            ).as_rotvec()
            return np.r_[
                (self._anchors() - desired).ravel(),
                (weights * angles).ravel(),
                self.config.temporal_weight * (values - prior),
                self.config.neutral_weight * (values - self.neutral),
            ]

        starts = [prior.copy()]
        if previous is None:
            for guess in self.config.initial_guesses:
                start = prior.copy()
                for name, value in guess.items():
                    if not np.isfinite(value):
                        raise ValueError("IK initial guesses must be finite")
                    start[self.domain.joints.index(self.model.joint(name).id)] = value
                starts.append(start)
        fits = [
            least_squares(
                objective,
                np.clip(
                    start, self.domain.bounds[0] + 1e-9, self.domain.bounds[1] - 1e-9
                ),
                bounds=self.domain.bounds,
                max_nfev=self.config.max_nfev,
                ftol=2e-5,
                xtol=2e-5,
                gtol=2e-5,
            )
            for start in starts
        ]
        fit = min(fits, key=lambda result: result.cost)
        pose = pose_at(fit.x)
        objective(fit.x)
        errors = np.linalg.norm(self._anchors() - desired, axis=1) * 1000
        angles = Rotation.from_matrix(
            target_rotation.swapaxes(-1, -2)
            @ self.data.xmat[self.orientation_ids].reshape(-1, 3, 3)
        ).as_rotvec()
        if physical_parameter_digest(self.model) != self.before:
            raise RuntimeError("Retargeting modified source physical constants")
        return pose, {
            "success": bool(fit.success),
            "nfev": int(fit.nfev),
            "source_anchor_errors_mm": dict(
                zip(self.config.landmarks, errors.tolist())
            ),
            "source_orientation_errors_deg": dict(
                zip(
                    self.config.orientation_weights,
                    np.rad2deg(np.linalg.norm(angles, axis=1)).tolist(),
                )
            ),
        }


def save_replay(
    path: str | Path,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    time: np.ndarray,
    source_frame_index: np.ndarray | None = None,
) -> None:
    """Write a safe full-model replay; omit names if any joint is unnamed.

    Uniform timing is required by the renderer's frequency-based clip format.
    Existing output files are never overwritten.
    """
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Replay output must end in .npz")
    qpos = _source_poses(model, qpos, "qpos")
    times = _real_array(time, "time")
    velocity = differentiate_trajectory(model, qpos, times)
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, rtol=1e-8, atol=1e-10):
        raise ValueError("Replay output requires uniformly sampled times")
    arrays = {
        "qpos": qpos,
        "qvel": velocity,
        "time": times,
        "frequency": np.array(1 / dt),
    }
    if source_frame_index is not None:
        frames = np.asarray(source_frame_index)
        if frames.shape != (len(qpos),) or frames.dtype.kind not in "iu":
            raise ValueError("source_frame_index must be an integer per frame")
        arrays["source_frame_index"] = frames
    names = [model.joint(j).name for j in range(model.njnt)]
    if all(isinstance(name, str) and name for name in names) and len(set(names)) == len(
        names
    ):
        arrays["joint_names"] = np.asarray(names, dtype=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        np.savez_compressed(file, **arrays)


def main() -> None:
    """Run a data-only original-pose recording as explicit kinematic replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    model_arg = parser.add_mutually_exclusive_group(required=True)
    model_arg.add_argument("--env")
    model_arg.add_argument("--model", type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ik-config", required=True, type=Path)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--packing", choices=("interleaved", "concatenated"), required=True
    )
    parser.add_argument(
        "--root-convention", choices=("native", "skim-training-ry90"), required=True
    )
    parser.add_argument(
        "--roots-recording",
        type=Path,
        help="Numeric file containing native roots, e.g. the original SKIM shot for training poses.",
    )
    parser.add_argument("--frequency", required=True, type=float)
    parser.add_argument("--pose-key", default="pose6d")
    parser.add_argument("--root-key", default="root_position")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--frames", default=51, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--world-translation",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
    )
    args = parser.parse_args()
    if (
        args.frequency <= 0
        or not np.isfinite(args.frequency)
        or args.start < 0
        or args.frames < 2
        or args.stride < 1
        or args.threads < 1
    ):
        parser.error(
            "Require positive finite frequency/threads/stride, start>=0 and frames>=2"
        )
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if not np.isfinite(args.world_translation).all():
        parser.error("--world-translation must be finite")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output already exists; choose a new replay path")
    import torch
    from myosuite_soma.soma_model import load_soma_model

    torch.set_num_threads(args.threads)
    env = None
    if args.env:
        try:
            from myosuite import make_env
        except ImportError as exc:
            parser.error(f"--env requires the optional MyoSuite installation: {exc}")
        env = make_env(args.env)
    try:
        model = (
            env.unwrapped.model
            if env
            else mujoco.MjModel.from_xml_path(str(args.model))
        )
        profile = SomaConfig.from_json(args.profile)
        predictor = load_soma_model(
            profile.data_root,
            profile.subject,
            profile.checkpoint,
            device=args.device,
            individual_muscles=profile.individual_muscles,
            batch_size=profile.batch_size,
        )
        config = IKConfig.from_json(args.ik_config)
        solver = SomaIKRetargeter(model, profile, predictor, config)
        with np.load(args.recording, allow_pickle=False) as source:
            values = source[args.pose_key]
            values = values.reshape(len(values), len(predictor.joint_names), 6)
            roots = (
                source[args.root_key].copy() if args.roots_recording is None else None
            )
            source_frames = (
                source["frame_index"].copy()
                if "frame_index" in source
                else np.arange(len(values))
            )
        if args.roots_recording is not None:
            with np.load(args.roots_recording, allow_pickle=False) as source:
                roots = source[args.root_key].copy()
                source_frames = (
                    source["frame_index"].copy()
                    if "frame_index" in source
                    else source_frames
                )
        local = decode_recording(
            values, packing=args.packing, root_convention=args.root_convention
        )
        rotations, positions = rig_forward_kinematics(
            local, roots, predictor.rest_joints, predictor.parents
        )
        indices = np.arange(
            args.start,
            min(len(local), args.start + args.frames * args.stride),
            args.stride,
        )
        if len(indices) < 2:
            parser.error("Requested recording interval contains fewer than two frames")
        poses, metrics, previous = [], [], None
        for index in indices:
            previous, info = solver.solve(rotations[index], positions[index], previous)
            poses.append(previous)
            metrics.append({"source_index": int(index), **info})
            print(
                f"frame {int(index)}: anchor RMSE {np.sqrt(np.mean(np.square(list(info['source_anchor_errors_mm'].values())))):.2f} mm",
                flush=True,
            )
        poses = np.array(poses)
        poses[:, solver.root_address : solver.root_address + 3] += (
            args.world_translation
        )
        times = (indices - indices[0]) / args.frequency
        report = {
            "kind": "Recorded bounded kinematic replay; no controller or activation inference",
            "environment": args.env,
            "source_model": str(args.model) if args.model else None,
            "profile": str(args.profile.resolve()),
            "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            "ik_config": str(args.ik_config.resolve()),
            "ik_config_sha256": hashlib.sha256(args.ik_config.read_bytes()).hexdigest(),
            "recording": str(args.recording.resolve()),
            "recording_sha256": hashlib.sha256(args.recording.read_bytes()).hexdigest(),
            "packing": args.packing,
            "root_convention": args.root_convention,
            "roots_recording": str(args.roots_recording.resolve())
            if args.roots_recording
            else None,
            "roots_recording_sha256": hashlib.sha256(
                (args.roots_recording or args.recording).read_bytes()
            ).hexdigest(),
            "native_to_world_rotation": profile.output_rotation.tolist(),
            "world_translation_m": list(args.world_translation),
            "scale": profile.scale,
            "source_parameter_sha256_before": solver.before,
            "source_parameter_sha256_after": physical_parameter_digest(model),
            "grounding": "Recorded root motion is retained with an optional explicit constant world translation. No contact/floor correction or controller is applied.",
            "frames": metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_replay(args.output, model, poses, times, source_frames[indices])
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if env:
            env.close()


if __name__ == "__main__":
    main()
