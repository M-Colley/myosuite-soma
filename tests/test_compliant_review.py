"""Independent numerical and input-contract review of the research runtime."""

import copy

import mujoco
import numpy as np
import pytest

from myosuite_soma import compliant_muscle as compliant


def _parameters():
    return compliant.CompliantParameters(1000.0, 0.1, 0.2, pennation_deg=15)


def _model(dt=0.0005):
    return mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option timestep="{dt}" gravity="0 0 0"/>
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
        "provenance": {"status": "synthetic", "source": "Invented review fixture"},
        "muscles": {
            "pull": {
                "force_N": 1000.0,
                "optimal_fibre_length_m": 0.1,
                "tendon_slack_length_m": 0.2,
                "pennation_deg": 15,
            }
        },
    }


def test_very_short_mtu_cannot_return_a_negative_anatomical_tendon_length():
    muscle = compliant.CompliantMuscle(_parameters())
    before = muscle.length
    with pytest.raises(ValueError):
        muscle.initialize(0.02, 0.0)
    assert muscle.length == before
    # A slack tendon may carry zero tension, but its geometric length must
    # remain nonnegative even when zero force admits many equilibria.
    state = muscle.initialize(0.08, 0.0)
    assert state["tendon_length_m"] >= 0
    lf = state["fibre_length_m"]
    assert np.sqrt(lf**2 - muscle.thickness**2) <= 0.08 + 1e-12


def test_backward_euler_fibre_response_converges_under_step_halving():
    def endpoint(dt):
        muscle = compliant.CompliantMuscle(_parameters())
        muscle.initialize(0.3, 0)
        for _ in range(round(0.08 / dt)):
            state = muscle.step(0.3, 0.6, dt)
        return state["fibre_length_m"]

    reference = endpoint(0.000125)
    coarse, medium, fine = (
        abs(endpoint(dt) - reference) for dt in (0.004, 0.002, 0.001)
    )
    assert fine < medium < coarse
    assert fine < coarse / 2


def test_operator_split_native_motion_converges_under_step_halving():
    def endpoint(dt):
        system = compliant.CompliantMuscleSystem(
            _model(dt), _config(), allow_synthetic=True
        )
        system.reset()
        for _ in range(round(0.06 / dt)):
            system.step(np.array([0.4]))
        return float(system.data.qpos[0])

    reference = endpoint(0.0000625)
    coarse, medium, fine = (
        abs(endpoint(dt) - reference) for dt in (0.001, 0.0005, 0.00025)
    )
    assert fine < medium < coarse
    assert fine < coarse / 2


@pytest.mark.parametrize(
    "kind", ["version_bool", "truthy_gate", "provenance_type", "source_bool"]
)
def test_configuration_provenance_and_synthetic_gates_are_strict(kind):
    config = _config()
    allow = True
    if kind == "version_bool":
        config["version"] = True
    elif kind == "truthy_gate":
        allow = "yes"
    elif kind == "provenance_type":
        config["provenance"] = ["synthetic"]
    else:
        config["provenance"]["source"] = True
    with pytest.raises(ValueError):
        compliant.CompliantMuscleSystem(_model(), config, allow_synthetic=allow)


def test_nonspatial_transmission_cannot_be_treated_as_anatomical_metres():
    source = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body><joint name="j"/><geom size=".1" mass="1"/></body></worldbody>
      <actuator><muscle name="pull" joint="j" force="1000" lengthrange=".25 .35"/></actuator>
    </mujoco>""")
    with pytest.raises(ValueError, match="spatial|transmission|metre"):
        compliant.CompliantMuscleSystem(source, _config(), allow_synthetic=True)


def test_native_numerical_warning_or_reset_is_not_reported_as_a_finite_success(
    monkeypatch,
):
    system = compliant.CompliantMuscleSystem(_model(), _config(), allow_synthetic=True)
    system.reset()

    def warned_reset(model, data):
        mujoco.mj_resetData(model, data)
        data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number += 1

    monkeypatch.setattr(compliant.mujoco, "mj_step", warned_reset)
    with pytest.raises(RuntimeError, match="warning|reset|advance|numerical"):
        system.step(np.array([0.5]))


def test_failed_reset_does_not_leave_old_initialized_flag_on_partial_state():
    system = compliant.CompliantMuscleSystem(_model(), _config(), allow_synthetic=True)
    system.reset()
    for _ in range(10):
        system.step(np.array([0.4]))
    original_qpos = system.data.qpos.copy()
    original_lengths = [m.length for m in system.muscles]
    with pytest.raises(ValueError):
        system.reset(qpos=np.array([np.nan]))
    assert not system.initialized or (
        np.array_equal(system.data.qpos, original_qpos)
        and [m.length for m in system.muscles] == original_lengths
    )


def test_force_clamp_is_rejected_before_custom_force_can_bypass_it():
    source = _model()
    source.actuator_forcelimited[0] = True
    source.actuator_forcerange[0] = [-100, 0]
    original = copy.copy(source)
    with pytest.raises(ValueError, match="force clamp"):
        compliant.CompliantMuscleSystem(source, _config(), allow_synthetic=True)
    np.testing.assert_array_equal(source.actuator_gainprm, original.actuator_gainprm)


@pytest.mark.parametrize("gear", [0.5, -1.0, 2.0])
def test_anatomical_force_and_length_units_require_unit_tendon_gear(gear):
    source = _model()
    source.actuator_gear[0, 0] = gear
    with pytest.raises(ValueError, match="unit gear|metre|spatial"):
        compliant.CompliantMuscleSystem(source, _config(), allow_synthetic=True)


def test_sparse_moment_force_matches_independent_transmission_length_derivatives():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><site name="left" pos="0 .4 .2"/><site name="right" pos="0 -.4 .2"/>
      <body name="upper" pos=".15 .1 .2"><joint name="rotate" axis="0 0 1"/>
      <geom size=".03" mass="1"/><body name="lower" pos=".2 0 0">
      <joint name="translate" type="slide" axis="1 0 0"/><geom size=".025" mass="1"/>
      <site name="tip" pos=".1 .1 0"/></body></body>
      <body name="unrelated" pos="1 0 0"><joint name="other" axis="0 0 1"/><geom size=".02" mass="1"/></body>
    </worldbody>
    <tendon><spatial name="leftpath"><site site="left"/><site site="tip"/></spatial>
      <spatial name="rightpath"><site site="right"/><site site="tip"/></spatial></tendon>
    <actuator><muscle name="leftm" tendon="leftpath" force="100" lengthrange=".2 .9"/>
      <muscle name="rightm" tendon="rightpath" force="100" lengthrange=".2 .9"/></actuator></mujoco>""")
    data = mujoco.MjData(model)
    pose = np.array([0.3, 0.05, 0.7])
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    actual = compliant.transmission_force(model, data, 1, -12.3)
    derivative = []
    for coordinate in range(3):
        plus, minus = pose.copy(), pose.copy()
        plus[coordinate] += 1e-6
        minus[coordinate] -= 1e-6
        data.qpos[:] = plus
        mujoco.mj_forward(model, data)
        length_plus = data.actuator_length[1]
        data.qpos[:] = minus
        mujoco.mj_forward(model, data)
        length_minus = data.actuator_length[1]
        derivative.append((length_plus - length_minus) / 2e-6)
    np.testing.assert_allclose(
        actual, -12.3 * np.array(derivative), rtol=1e-8, atol=1e-9
    )
    assert actual[2] == 0
