"""Analytic rigid-body truths without fabricated muscle activations."""

import copy
import json

import mujoco
import numpy as np
import pytest

from myosuite_soma.physical_dynamics import validate_inverse_dynamics
from myosuite_soma.physical_model import build_physical_model, physical_digest


def _base(model, qpos, qvel=None, qacc=None):
    qpos = np.asarray(qpos)
    count = len(qpos)
    return {
        "time": np.arange(count) * 0.01,
        "qpos": qpos,
        "qvel": np.zeros((count, model.nv)) if qvel is None else np.asarray(qvel),
        "qacc": np.zeros((count, model.nv)) if qacc is None else np.asarray(qacc),
        "external_forces": [],
        "provenance": {
            "status": "synthetic",
            "source": "Independent analytic rigid-body fixture",
        },
    }


def _free_model():
    return mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 -9.81"/>
    <worldbody><geom type="plane" size="1 1 .01"/>
    <body name="body" pos="0 0 .05"><freejoint name="root"/>
      <inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/>
      <geom type="sphere" size=".1"/>
    </body></worldbody></mujoco>""")


def test_measured_ground_reaction_balances_free_base_without_simulated_contact_double_counting():
    model = _free_model()
    observations = _base(model, np.tile(model.qpos0, (3, 1)))
    observations["external_forces"] = [
        {
            "body": "body",
            "point_world_m": np.zeros((3, 3)),
            "force_world_N": np.tile([0, 0, 19.62], (3, 1)),
            "torque_world_Nm": np.zeros((3, 3)),
        }
    ]
    before = physical_digest(model)
    result = validate_inverse_dynamics(model, observations, allow_synthetic=True)
    assert physical_digest(model) == before
    np.testing.assert_allclose(result["required_generalized_force"], 0, atol=1e-12)
    assert result["free_base_residuals"]["root"]["force_rms_norm_N"] < 1e-12
    assert result["maximum_native_contacts_ignored"] > 0
    assert result["native_contact_forces_used"] is False
    assert result["muscle_activation_required"] is False
    assert len(result["unactuated_residuals"]) == 6
    json.dumps(result, allow_nan=False)
    observations["external_forces"] = []
    unbalanced = validate_inverse_dynamics(model, observations, allow_synthetic=True)
    assert unbalanced["free_base_residuals"]["root"][
        "force_rms_norm_N"
    ] == pytest.approx(19.62)


def test_world_force_point_and_moment_map_to_native_free_joint_rotational_basis():
    model = _free_model()
    pose = model.qpos0.copy()
    pose[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    observations = _base(model, pose[None])
    force, point, torque = (
        np.array([0, 0, 19.62]),
        np.array([0.1, 0, 0]),
        np.array([1.0, 2.0, 3.0]),
    )
    observations["external_forces"] = [
        {
            "body": "body",
            "point_world_m": point[None],
            "force_world_N": force[None],
            "torque_world_Nm": torque[None],
        }
    ]
    report = validate_inverse_dynamics(model, observations, allow_synthetic=True)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = -rotation.T @ (np.cross(point - pose[:3], force) + torque)
    np.testing.assert_allclose(
        report["free_base_residuals"]["root"]["torque_Nm"][0], expected, atol=1e-12
    )
    np.testing.assert_allclose(
        report["free_base_residuals"]["root"]["force_N"], 0, atol=1e-12
    )


def test_pendulum_mass_gravity_acceleration_and_passive_damping_match_analytic_torque():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <option gravity="0 -9.81 0"/><worldbody><body name="pendulum">
      <joint name="hinge" axis="0 0 1" damping=".2"/>
      <inertial pos=".5 0 0" mass="2" diaginertia=".01 .02 .03"/>
    </body></worldbody></mujoco>""")
    qpos = np.array([[0.0], [0.3], [-0.7], [1.1]])
    qvel = np.array([[0.2], [-0.1], [0.5], [0.0]])
    qacc = np.array([[0.4], [-0.3], [0.1], [0.7]])
    observations = _base(model, qpos, qvel, qacc)
    expected = 0.53 * qacc + 9.81 * np.cos(qpos) + 0.2 * qvel
    observations["reference_generalized_force"] = expected
    observations["dof_names"] = ["hinge"]
    report = validate_inverse_dynamics(model, observations, allow_synthetic=True)
    np.testing.assert_allclose(
        report["required_generalized_force"], expected, atol=1e-12
    )
    assert report["reference_comparison"]["by_dof"]["hinge"]["rmse"] < 1e-12
    assert report["coordinate_units"] == ["Nm"]


def test_polynomial_couplings_project_virtual_work_before_joint_torque_comparison():
    model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0"/>
    <worldbody><body><joint name="driver" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
      <body><joint name="dependent" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
      </body></body></worldbody>
    <equality><joint joint1="dependent" joint2="driver" polycoef="0 .5 0 0 0"/></equality></mujoco>""")
    observations = _base(model, [[0.3, 0.15]], [[0.2, 0.1]], [[0.3, 0.15]])
    observations["reference_generalized_force"] = [[0.325 * 0.3]]
    observations["dof_names"] = ["driver"]
    result = validate_inverse_dynamics(model, observations, allow_synthetic=True)
    assert result["reference_comparison"]["by_dof"]["driver"]["rmse"] < 1e-12
    assert result["projected_required_generalized_force"][0][1] == 0
    observations["dof_names"] = ["dependent"]
    with pytest.raises(ValueError, match="independent joints"):
        validate_inverse_dynamics(model, observations, allow_synthetic=True)


def test_slide_reference_force_uses_newtons_and_explicit_external_loads():
    model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 -9.81"/>
    <worldbody><body name="body"><joint name="height" type="slide" axis="0 0 1" damping=".4"/>
      <inertial pos="0 0 0" mass="3" diaginertia=".1 .1 .1"/>
    </body></worldbody></mujoco>""")
    observations = _base(model, [[0.2]], [[0.2]], [[0.5]])
    observations["external_forces"] = [
        {
            "body": "body",
            "point_world_m": [[0.1, 0, 0]],
            "force_world_N": [[0, 0, 5]],
            "torque_world_Nm": [[0, 0, 3]],
        }
    ]
    observations["reference_generalized_force"] = [[3 * 0.5 + 3 * 9.81 + 0.4 * 0.2 - 5]]
    observations["dof_names"] = ["height"]
    report = validate_inverse_dynamics(model, observations, allow_synthetic=True)
    assert report["reference_comparison"]["by_dof"]["height"]["unit"] == "N"
    assert report["reference_comparison"]["by_dof"]["height"]["rmse"] < 1e-12


def test_changed_segment_mass_is_detected_against_unchanged_measured_support():
    source = _free_model()
    observations = _base(source, source.qpos0[None])
    observations["external_forces"] = [
        {
            "body": "body",
            "point_world_m": [[0, 0, 0]],
            "force_world_N": [[0, 0, 19.62]],
            "torque_world_Nm": [[0, 0, 0]],
        }
    ]
    observations["conventions"] = {"source_model_sha256": physical_digest(source)}
    profile = {
        "version": 1,
        "provenance": {"status": "synthetic", "source": "Invented mass variant"},
        "bodies": {"body": {"mass_kg": 3}},
    }
    candidate = build_physical_model(source, profile, allow_synthetic=True).model
    report = validate_inverse_dynamics(candidate, observations, allow_synthetic=True)
    assert report["same_input_physics"] is False
    assert report["free_base_residuals"]["root"]["force_rms_norm_N"] == pytest.approx(
        9.81
    )


def _nonlinear_coupled_model(*, fixed_parent=False):
    constraint = (
        '<joint joint1="dependent" polycoef=".2 0 0 0 0"/>'
        if fixed_parent
        else '<joint joint1="dependent" joint2="driver" polycoef="0 .5 .2 0 0"/>'
    )
    return mujoco.MjModel.from_xml_string(f"""<mujoco><option gravity="0 0 0"/>
    <worldbody><body><joint name="driver" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
      <body><joint name="dependent" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
      </body></body></worldbody><equality>{constraint}</equality></mujoco>""")


def test_nonlinear_equality_acceleration_includes_second_derivative_velocity_term():
    model = _nonlinear_coupled_model()
    position, velocity, acceleration = 0.3, 0.4, 0.3
    derivative = 0.5 + 0.4 * position
    child_acceleration = derivative * acceleration + 0.4 * velocity**2
    observations = _base(
        model,
        [[position, 0.5 * position + 0.2 * position**2]],
        [[velocity, derivative * velocity]],
        [[acceleration, child_acceleration]],
    )
    # Two coaxial inertias: child absolute angular acceleration is the sum
    # of driver and relative-child acceleration, including the p''*v² term.
    expected = 0.1 * acceleration + 0.1 * (1 + derivative) * (
        acceleration + child_acceleration
    )
    observations["reference_generalized_force"] = [[expected]]
    observations["dof_names"] = ["driver"]
    result = validate_inverse_dynamics(model, observations, allow_synthetic=True)
    assert result["reference_comparison"]["by_dof"]["driver"]["rmse"] < 1e-12
    residuals = result["joint_equality_kinematics"]["by_joint"]["dependent"]
    assert residuals["maximum_abs_velocity_residual"] < 1e-12
    assert residuals["maximum_abs_acceleration_residual"] < 1e-12
    observations["qacc"][0, 1] = derivative * acceleration
    with pytest.raises(ValueError, match="acceleration violates scalar joint equality"):
        validate_inverse_dynamics(model, observations, allow_synthetic=True)
    observations["qacc"][0, 1] = child_acceleration
    observations["qvel"][0, 1] += 0.1
    with pytest.raises(ValueError, match="velocity violates scalar joint equality"):
        validate_inverse_dynamics(model, observations, allow_synthetic=True)


@pytest.mark.parametrize("field", ["qvel", "qacc"])
def test_fixed_parent_equality_requires_zero_child_derivatives(field):
    model = _nonlinear_coupled_model(fixed_parent=True)
    observations = _base(model, [[0.3, 0.2]], [[0.4, 0]], [[0.3, 0]])
    assert validate_inverse_dynamics(model, observations, allow_synthetic=True)[
        "joint_equality_kinematics"
    ]["within_declared_tolerance"]
    observations[field][0, 1] = 0.1
    with pytest.raises(ValueError, match="violates scalar joint equality"):
        validate_inverse_dynamics(model, observations, allow_synthetic=True)


@pytest.mark.parametrize(
    "kind",
    [
        "synthetic_gate",
        "missing_external",
        "world_load",
        "bad_force_shape",
        "bad_quaternion",
        "missing_reference_names",
        "nonfinite_acceleration",
        "bad_provenance",
        "unknown_fields",
    ],
)
def test_invalid_inputs_do_not_become_muscle_force_claims(kind):
    model = _free_model()
    observations = _base(model, np.tile(model.qpos0, (2, 1)))
    allow = True
    if kind == "synthetic_gate":
        allow = False
    elif kind == "missing_external":
        observations.pop("external_forces")
    elif kind in ("world_load", "bad_force_shape"):
        observations["external_forces"] = [
            {
                "body": "world" if kind == "world_load" else "body",
                "point_world_m": np.zeros((2, 3)),
                "force_world_N": np.zeros((2, 2 if kind == "bad_force_shape" else 3)),
                "torque_world_Nm": np.zeros((2, 3)),
            }
        ]
    elif kind == "bad_quaternion":
        observations["qpos"][:, 3:7] = 0
    elif kind == "missing_reference_names":
        observations["reference_generalized_force"] = [[0], [0]]
    elif kind == "nonfinite_acceleration":
        observations["qacc"][0, 0] = np.nan
    elif kind == "bad_provenance":
        observations["provenance"] = {"status": "measured", "source": True}
    else:
        observations["activation"] = [[0], [0]]
    original = copy.copy(model)
    with pytest.raises(ValueError):
        validate_inverse_dynamics(model, observations, allow_synthetic=allow)
    assert physical_digest(model) == physical_digest(original)
