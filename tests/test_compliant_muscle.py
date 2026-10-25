"""Synthetic constitutive and forward-dynamics checks, not physiology validation."""

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from myosuite_soma.compliant_muscle import (
    CompliantMuscle,
    CompliantMuscleSystem,
    CompliantParameters,
    transmission_force,
)


def _parameters():
    return CompliantParameters(1000.0, 0.1, 0.2, pennation_deg=15)


def _source():
    return mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".0005" gravity="0 0 0"/>
      <worldbody><site name="origin" pos="0 0 0"/>
      <body name="slider" pos=".3 0 0"><joint name="x" type="slide" axis="1 0 0" damping="5"/>
        <geom size=".03" mass="10"/><site name="insertion"/>
      </body></worldbody>
      <tendon><spatial name="path"><site site="origin"/><site site="insertion"/></spatial></tendon>
      <actuator><muscle name="pull" tendon="path" force="1000" lengthrange=".25 .35"/></actuator>
    </mujoco>""")


def _config():
    return {
        "version": 1,
        "provenance": {"status": "synthetic", "source": "invented test constants"},
        "muscles": {
            "pull": {
                "force_N": 1000,
                "optimal_fibre_length_m": 0.1,
                "tendon_slack_length_m": 0.2,
                "pennation_deg": 15,
            }
        },
    }


def test_constitutive_force_and_pennation_match_independent_geometry():
    p = _parameters()
    muscle = CompliantMuscle(p)
    lf = 0.105
    thickness = 0.1 * np.sin(np.deg2rad(15))
    projected = np.sqrt(lf**2 - thickness**2)
    mtu = projected + 0.2 * (1 + p.strain_at_force_max)
    assert muscle.tendon_force(lf, mtu) == pytest.approx(1000, rel=1e-12)
    cosine, tendon, strain = muscle.geometry(lf, mtu)
    assert cosine == pytest.approx(projected / lf)
    assert tendon == pytest.approx(0.2 * (1 + p.strain_at_force_max))
    assert strain == pytest.approx(p.strain_at_force_max)
    assert muscle.tendon_force(lf, projected + 0.19) == 0


def test_implicit_fibre_response_has_force_equilibrium_and_constant_thickness():
    muscle = CompliantMuscle(_parameters())
    muscle.initialize(0.3)
    initial = muscle.length
    states = [muscle.step(0.3, 0.7, 0.001) for _ in range(100)]
    assert states[-1]["fibre_length_m"] < initial
    assert states[-1]["tendon_force_N"] > 100
    assert max(abs(s["force_balance_residual_N"]) for s in states) < 1e-5
    for state in states:
        assert state["fibre_length_m"] * np.sin(
            np.deg2rad(state["pennation_deg"])
        ) == pytest.approx(muscle.thickness)
        projected = state["fibre_length_m"] * np.cos(np.deg2rad(state["pennation_deg"]))
        assert projected + state["tendon_length_m"] == pytest.approx(0.3)


def test_steady_state_tendon_force_changes_with_activation_and_length():
    values = []
    for length, activation in [(0.3, 0), (0.3, 0.4), (0.31, 0.4)]:
        muscle = CompliantMuscle(_parameters())
        state = muscle.initialize(length, activation)
        assert abs(state["force_balance_residual_N"]) < 1e-5
        values.append(state["tendon_force_N"])
    assert values[1] > values[0]
    assert values[2] != pytest.approx(values[1])


@pytest.mark.parametrize(
    "field,value",
    [
        ("force_N", -1),
        ("pennation_deg", 80),
        ("damping_seconds", 0),
        ("strain_at_force_max", 0.3),
        ("min_fibre_ratio", 2),
    ],
)
def test_invalid_parameter_values_rejected(field, value):
    with pytest.raises(ValueError):
        replace(_parameters(), **{field: value}).validate()


def test_invalid_input_and_outside_range_leave_fibre_state_unchanged():
    muscle = CompliantMuscle(_parameters())
    before = muscle.length
    for length, activation, dt in [
        (0.3, 2, 0.001),
        (np.nan, 0.5, 0.001),
        (0.3, 0.5, 0),
        (2.0, 0.5, 0.001),
    ]:
        with pytest.raises(ValueError):
            muscle.step(length, activation, dt)
        assert muscle.length == before


def test_forward_runtime_changes_physics_without_mutating_source_model():
    source = _source()
    original_gain = source.actuator_gainprm.copy()
    original_bias = source.actuator_biasprm.copy()
    config = _config()
    with pytest.raises(ValueError, match="Synthetic"):
        CompliantMuscleSystem(source, config)
    system = CompliantMuscleSystem(source, config, allow_synthetic=True)
    system.reset()
    baseline = system.data.qfrc_applied.copy()
    states = [system.step(np.array([0.5])) for _ in range(120)]
    assert system.data.qpos[0] < 0
    assert states[-1]["muscles"]["pull"]["tendon_force_N"] > 0
    assert (
        np.max([abs(s["muscles"]["pull"]["force_balance_residual_N"]) for s in states])
        < 1e-5
    )
    assert states[-1]["custom_generalized_force"][0] < 0
    assert system.data.act[0] > 0.1
    np.testing.assert_array_equal(system.data.qfrc_applied, baseline)
    np.testing.assert_array_equal(source.actuator_gainprm, original_gain)
    np.testing.assert_array_equal(source.actuator_biasprm, original_bias)
    assert config == _config()


def test_native_moment_projection_and_external_force_accounting():
    source = _source()
    system = CompliantMuscleSystem(source, _config(), allow_synthetic=True)
    system.reset()
    mujoco.mj_forward(system.model, system.data)
    np.testing.assert_allclose(
        transmission_force(system.model, system.data, 0, -10), [-10]
    )
    system.data.qfrc_applied[:] = 3
    system.step(np.array([0.0]), external_force=np.array([4.0]))
    np.testing.assert_array_equal(system.data.qfrc_applied, [3])
    with pytest.raises(ValueError, match="controls"):
        system.step(np.array([np.nan]))


def test_rk4_and_bad_actuator_configuration_are_rejected():
    source = _source()
    source.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    with pytest.raises(ValueError, match="RK4"):
        CompliantMuscleSystem(source, _config(), allow_synthetic=True)
    source.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    source.actuator_actearly[:] = 1
    with pytest.raises(ValueError, match="non-early"):
        CompliantMuscleSystem(source, _config(), allow_synthetic=True)
    config = copy.deepcopy(_config())
    config["muscles"]["missing"] = config["muscles"].pop("pull")
    with pytest.raises(ValueError, match="Unknown compliant muscle"):
        CompliantMuscleSystem(_source(), config, allow_synthetic=True)
