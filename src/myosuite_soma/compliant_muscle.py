# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Experimental compliant tendon / constant-thickness pennation dynamics.

This is a separate runtime, not a silent replacement of native MuJoCo muscles.
Selected muscles retain native activation dynamics, but their transmission
forces are replaced by an implicit fibre step and an elastic tendon force.
Forces are applied through the native moment arms as generalized forces.
The implementation is an operator-split research model, not subject validation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.optimize import brentq


def _positive(value: Any, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or result < 0 or (not zero and result == 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )
    return result


@dataclass(frozen=True)
class CompliantParameters:
    """Effective fibre and tendon parameters, in SI units.

    Tendon force is ``Fmax * (positive_strain / strain_at_force_max)**2``.
    Constant muscle thickness is ``optimal_fibre_length * sin(pennation)``.
    ``damping_seconds`` regularizes fibre force balance and dissipates energy.
    These constitutive assumptions need independent measured validation.
    """

    force_N: float
    optimal_fibre_length_m: float
    tendon_slack_length_m: float
    pennation_deg: float = 0.0
    strain_at_force_max: float = 0.049
    damping_seconds: float = 0.05
    min_fibre_ratio: float = 0.4
    max_fibre_ratio: float = 1.8
    max_tendon_strain: float = 0.25

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompliantParameters:
        if (
            not isinstance(value, Mapping)
            or set(value) - cls.__dataclass_fields__.keys()
        ):
            raise ValueError("Unknown compliant muscle parameter fields")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ValueError(f"Missing compliant muscle parameters: {exc}") from exc
        result.validate()
        return result

    def validate(self) -> None:
        for key in self.__dataclass_fields__:
            _positive(getattr(self, key), key, zero=key == "pennation_deg")
        if self.pennation_deg >= 75:
            raise ValueError("pennation_deg must be in [0,75)")
        if self.min_fibre_ratio >= self.max_fibre_ratio:
            raise ValueError("Fibre ratio bounds must be increasing")
        if not self.min_fibre_ratio < 1 < self.max_fibre_ratio:
            raise ValueError("Fibre ratio bounds must contain the optimal length")
        if self.strain_at_force_max >= self.max_tendon_strain:
            raise ValueError("Reference tendon strain must be below maximum strain")


class CompliantMuscle:
    """A single implicit fibre state coupled to an elastic biological tendon."""

    def __init__(
        self,
        parameters: CompliantParameters,
        *,
        native_parameters=None,
        native_passive_parameters=None,
    ):
        parameters.validate()
        self.parameters = parameters
        p = parameters
        self.thickness = p.optimal_fibre_length_m * np.sin(np.deg2rad(p.pennation_deg))
        self.lower = max(
            p.min_fibre_ratio * p.optimal_fibre_length_m,
            self.thickness / np.sin(np.deg2rad(85)),
        )
        self.upper = p.max_fibre_ratio * p.optimal_fibre_length_m
        self.length = p.optimal_fibre_length_m
        self._prm = np.array(
            [0.5, 1.5, p.force_N, 200, 0.5, 1.6, 10, 1.3, 1.2], dtype=float
        )
        if native_parameters is not None:
            native = np.asarray(native_parameters, dtype=float)
            if native.shape != (9,) or not np.isfinite(native).all():
                raise ValueError(
                    "native_parameters must contain nine finite muscle values"
                )
            self._prm[4:] = native[4:]
        self._passive_prm = self._prm.copy()
        if native_passive_parameters is not None:
            native = np.asarray(native_passive_parameters, dtype=float)
            if native.shape != (9,) or not np.isfinite(native).all():
                raise ValueError("native_passive_parameters need nine finite values")
            self._passive_prm[4:] = native[4:]
        # Feed the native FLV curve a range that gives exactly the supplied L0.
        self._range = p.tendon_slack_length_m + p.optimal_fibre_length_m * self._prm[:2]

    def geometry(self, length: float, mtu_length: float) -> tuple[float, float, float]:
        projected = np.sqrt(max(length * length - self.thickness**2, 1e-20))
        cosine = projected / length
        tendon_length = mtu_length - projected
        strain = max(tendon_length / self.parameters.tendon_slack_length_m - 1, 0)
        return cosine, tendon_length, strain

    def tendon_force(self, length: float, mtu_length: float) -> float:
        strain = self.geometry(length, mtu_length)[2]
        return (
            self.parameters.force_N
            * (strain / self.parameters.strain_at_force_max) ** 2
        )

    def _balance(self, length, mtu_length, activation, previous, dt):
        p = self.parameters
        velocity = 0.0 if dt is None else (length - previous) / dt
        total = length + p.tendon_slack_length_m
        active = -mujoco.mju_muscleGain(total, velocity, self._range, 1.0, self._prm)
        passive = -mujoco.mju_muscleBias(total, self._range, 1.0, self._passive_prm)
        damping = p.force_N * p.damping_seconds * velocity / p.optimal_fibre_length_m
        return (activation * active + passive + damping) * self.geometry(
            length, mtu_length
        )[0] - self.tendon_force(length, mtu_length)

    def _solve(self, mtu_length: float, activation: float, dt: float | None):
        mtu_length = _positive(mtu_length, "musculotendon length")
        activation = _positive(activation, "activation", zero=True)
        if activation > 1:
            raise ValueError("activation must be in [0,1]")
        if dt is not None:
            dt = _positive(dt, "dt")
        previous = self.length
        upper = min(self.upper, float(np.sqrt(mtu_length**2 + self.thickness**2)))
        if upper < self.lower:
            raise ValueError(
                "Musculotendon length is incompatible with the fibre geometry bounds"
            )

        def balance(length):
            return self._balance(length, mtu_length, activation, previous, dt)

        candidate = float(np.clip(previous, self.lower, upper))
        if abs(balance(candidate)) <= 1e-10 * self.parameters.force_N:
            length = candidate
        else:
            low, high = balance(self.lower), balance(upper)
            if low * high > 0:
                raise ValueError("No fibre-force equilibrium within configured bounds")
            length = brentq(balance, self.lower, upper, xtol=1e-13)
        cosine, tendon_length, strain = self.geometry(length, mtu_length)
        if tendon_length < -1e-12:
            raise ValueError("Fibre equilibrium would give a negative tendon length")
        tendon_length = max(tendon_length, 0.0)
        if strain > self.parameters.max_tendon_strain:
            raise ValueError("Tendon strain exceeds the declared research-model limit")
        force = self.tendon_force(length, mtu_length)
        residual = float(balance(length))
        self.length = length
        return {
            "fibre_length_m": length,
            "fibre_velocity_m_s": 0.0 if dt is None else (length - previous) / dt,
            "tendon_length_m": tendon_length,
            "tendon_strain": strain,
            "tendon_force_N": force,
            "pennation_deg": float(np.rad2deg(np.arccos(np.clip(cosine, 0, 1)))),
            "force_balance_residual_N": residual,
        }

    def initialize(self, mtu_length: float, activation: float = 0.0):
        """Find static equilibrium at the supplied length and activation."""
        return self._solve(mtu_length, activation, None)

    def step(self, mtu_length: float, activation: float, dt: float):
        """Advance one backward-Euler fibre step, returning explicit diagnostics."""
        return self._solve(mtu_length, activation, dt)


def transmission_force(model, data, actuator: int, force: float) -> np.ndarray:
    """Map a signed transmission force through native dense or sparse moments."""
    moment = np.asarray(data.actuator_moment)
    if moment.ndim == 2:
        return moment[actuator] * force
    start, size = int(data.moment_rowadr[actuator]), int(data.moment_rownnz[actuator])
    result = np.zeros(model.nv)
    np.add.at(
        result,
        data.moment_colind[start : start + size],
        moment[start : start + size] * force,
    )
    return result


class CompliantMuscleSystem:
    """Cloned forward simulation with explicit replacement muscle forces.

    Supply native actuator controls, not normalized environment actions. Native
    activation states still evolve in MuJoCo. Selected actuator outputs are
    zero; their physical forces are reported separately and applied through
    ``qfrc_applied``. Force is held over each native physics step. RK4 and
    early-activation actuators are rejected because this split runtime does not
    integrate their internal stages. No global callbacks or source edits occur.
    """

    def __init__(self, source: mujoco.MjModel, config: dict, *, allow_synthetic=False):
        if not isinstance(allow_synthetic, bool):
            raise ValueError("allow_synthetic must be an explicit boolean")
        if not isinstance(config, dict) or set(config) - {
            "version",
            "provenance",
            "muscles",
        }:
            raise ValueError("Unknown compliant-system configuration fields")
        if (
            config.get("version") != 1
            or isinstance(config.get("version"), bool)
            or not isinstance(config.get("muscles"), dict)
            or not config["muscles"]
        ):
            raise ValueError(
                "Compliant-system version 1 and named muscles are required"
            )
        provenance = config.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("Compliant parameter provenance must be a JSON object")
        if (
            provenance.get("status")
            not in {
                "measured",
                "estimated",
                "synthetic",
            }
            or not isinstance(provenance.get("source"), str)
            or not provenance.get("source")
        ):
            raise ValueError(
                "Explicit parameter provenance status and source are required"
            )
        if provenance["status"] == "synthetic" and not allow_synthetic:
            raise ValueError("Synthetic compliant parameters require explicit opt-in")
        if int(source.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4):
            raise ValueError(
                "The operator-split compliant runtime does not support RK4"
            )
        if np.any(source.jnt_actfrclimited):
            raise ValueError(
                "Aggregate joint actuator-force limits are unsupported by the separate-force runtime"
            )
        if source.opt.disableactuator or int(source.opt.disableflags) & int(
            mujoco.mjtDisableBit.mjDSBL_ACTUATION
        ):
            raise ValueError("Compliant runtime requires enabled native actuation")
        self.model = copy.copy(source)
        self.names = tuple(config["muscles"])
        self.indices, self.muscles = [], []
        for name, values in config["muscles"].items():
            if not isinstance(name, str) or not name:
                raise ValueError("Compliant muscles require exact actuator names")
            index = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if index < 0:
                raise ValueError(f"Unknown compliant muscle actuator: {name}")
            if int(source.actuator_trntype[index]) != int(mujoco.mjtTrn.mjTRN_TENDON):
                raise ValueError(
                    "Compliant anatomical lengths require a spatial tendon transmission in metres"
                )
            tendon = int(source.actuator_trnid[index, 0])
            first, count = (
                int(source.tendon_adr[tendon]),
                int(source.tendon_num[tendon]),
            )
            types = source.wrap_type[first : first + count]
            if (
                np.any(types == int(mujoco.mjtWrap.mjWRAP_JOINT))
                or np.any(types == int(mujoco.mjtWrap.mjWRAP_PULLEY))
                or not np.array_equal(source.actuator_gear[index], [1, 0, 0, 0, 0, 0])
            ):
                raise ValueError(
                    "A spatial tendon with unit gear and no pulley is required for anatomical metre/force parameters"
                )
            if (
                int(source.actuator_gaintype[index])
                != int(mujoco.mjtGain.mjGAIN_MUSCLE)
                or int(source.actuator_biastype[index])
                != int(mujoco.mjtBias.mjBIAS_MUSCLE)
                or int(source.actuator_dyntype[index])
                != int(mujoco.mjtDyn.mjDYN_MUSCLE)
                or source.actuator_actnum[index] != 1
                or source.actuator_actearly[index]
                or source.actuator_forcelimited[index]
            ):
                raise ValueError(
                    f"{name} needs one native, non-early muscle activation state and no force clamp"
                )
            self.indices.append(index)
            self.muscles.append(
                CompliantMuscle(
                    CompliantParameters.from_dict(values),
                    native_parameters=source.actuator_gainprm[index, :9],
                    native_passive_parameters=source.actuator_biasprm[index, :9],
                )
            )
            self.model.actuator_gaintype[index] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[index] = mujoco.mjtBias.mjBIAS_NONE
            self.model.actuator_gainprm[index] = 0
            self.model.actuator_biasprm[index] = 0
        self.data = mujoco.MjData(self.model)
        self.config = copy.deepcopy(config)
        self.initialized = False

    def reset(self, *, qpos=None, qvel=None, activation=None):
        """Reset owned state and solve each selected fibre's static equilibrium."""
        self.initialized = False
        mujoco.mj_resetData(self.model, self.data)
        for value, target, label in (
            (qpos, self.data.qpos, "qpos"),
            (qvel, self.data.qvel, "qvel"),
            (activation, self.data.act, "activation"),
        ):
            if value is not None:
                array = np.asarray(value)
                if (
                    array.dtype.kind not in "fiu"
                    or array.shape != target.shape
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(
                        f"{label} needs finite values shaped {target.shape}"
                    )
                target[:] = array
        mujoco.mj_forward(self.model, self.data)
        result = {}
        for name, index, muscle in zip(self.names, self.indices, self.muscles):
            muscle.length = muscle.parameters.optimal_fibre_length_m
            result[name] = muscle.initialize(
                float(self.data.actuator_length[index]),
                float(self.data.act[self.model.actuator_actadr[index]]),
            )
        self.initialized = True
        return result

    def step(self, controls: np.ndarray, *, external_force=None):
        """Advance native dynamics and return separately named physical forces."""
        controls = np.asarray(controls)
        if (
            controls.dtype.kind not in "fiu"
            or controls.shape != (self.model.nu,)
            or not np.isfinite(controls).all()
        ):
            raise ValueError(f"controls must be finite with shape ({self.model.nu},)")
        if not self.initialized:
            self.reset()
        data, model = self.data, self.model
        start_time = float(data.time)
        original = data.qfrc_applied.copy()
        baseline = (
            original
            if external_force is None
            else np.asarray(external_force, dtype=float)
        )
        if baseline.shape != (model.nv,) or not np.isfinite(baseline).all():
            raise ValueError(
                "external_force must contain one finite force per native DOF"
            )
        old_lengths = [muscle.length for muscle in self.muscles]
        data.ctrl[:] = controls
        mujoco.mj_forward(model, data)
        results, custom = {}, np.zeros(model.nv)
        warnings_before = np.array([w.number for w in data.warning], dtype=np.int64)
        try:
            for name, index, muscle in zip(self.names, self.indices, self.muscles):
                state = muscle.step(
                    float(data.actuator_length[index]),
                    float(data.act[model.actuator_actadr[index]]),
                    model.opt.timestep,
                )
                results[name] = state
                custom += transmission_force(
                    model, data, index, -state["tendon_force_N"]
                )
            data.qfrc_applied[:] = baseline + custom
            mujoco.mj_step(model, data)
            warnings_after = np.array([w.number for w in data.warning], dtype=np.int64)
            if np.any(warnings_after > warnings_before) or not np.isclose(
                data.time, start_time + model.opt.timestep, rtol=1e-10, atol=1e-12
            ):
                raise RuntimeError(
                    "Native numerical warning or state reset during compliant step"
                )
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(
                    "Compliant forward rollout produced nonfinite native state"
                )
        except Exception:
            self.initialized = False
            for muscle, previous in zip(self.muscles, old_lengths):
                muscle.length = previous
            raise
        finally:
            data.qfrc_applied[:] = original
        return {
            "time": float(data.time),
            "force_sample_time": start_time,
            "muscles": results,
            "custom_generalized_force": custom.copy(),
        }
