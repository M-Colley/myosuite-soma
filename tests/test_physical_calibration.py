"""Native-engine ground truth tests for local physical calibration."""

from __future__ import annotations

import copy
import json

import mujoco
import numpy as np
import pytest

from myosuite_soma.physical_calibration import (
    fit_physical_model,
    validate_physical_model,
)
from myosuite_soma.physical_model import build_physical_model, physical_digest
from myosuite_soma.soma import model_digest


_XML = """<mujoco><compiler angle="radian"/>
<option gravity="0 0 0" timestep=".001"/>
<worldbody>
  <site name="anchor_f" pos="0 .4 0"/>
  <site name="anchor_e" pos="0 -.4 0"/>
  <body name="limb"><joint name="ankle" axis="0 0 1" range="-.8 .8" damping=".15"/>
    <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/>
    <site name="tip" pos=".2 0 0"/>
  </body>
</worldbody>
<tendon>
  <spatial name="flexor_path"><site site="anchor_f"/><site site="tip"/></spatial>
  <spatial name="extensor_path"><site site="anchor_e"/><site site="tip"/></spatial>
</tendon>
<actuator>
  <muscle name="flexor" tendon="flexor_path" force="80" lengthrange=".3 .6" range=".5 1.5"/>
  <muscle name="extensor" tendon="extensor_path" force="60" lengthrange=".3 .6" range=".5 1.5"/>
</actuator></mujoco>"""


def _fixture(*, target_type="actuator", identical_activations=False):
    source = mujoco.MjModel.from_xml_string(_XML)
    truth = copy.copy(source)
    truth.actuator_gainprm[:, 2] = [170.0, 95.0]
    truth.actuator_biasprm[:, 2] = [170.0, 95.0]
    count = 64
    rng = np.random.default_rng(321)
    qpos = np.linspace(-0.65, 0.65, count)[:, None]
    qvel = np.sin(np.arange(count) * 0.23)[:, None] * 0.12
    activation = rng.uniform(0.05, 0.95, (count, 2))
    if identical_activations:
        # Isometric repeated geometry with equal activations makes the muscle
        # force contributions linearly dependent, even with regularization.
        qpos[:] = 0
        qvel[:] = 0
        activation[:, 1] = activation[:, 0]
    data = mujoco.MjData(truth)
    torque = []
    for position, velocity, active in zip(qpos, qvel, activation):
        mujoco.mj_resetData(truth, data)
        data.qpos[:] = position
        data.qvel[:] = velocity
        data.act[:] = active
        mujoco.mj_forward(truth, data)
        value = data.qfrc_actuator.copy()
        if target_type == "net":
            value += data.qfrc_passive
        torque.append(value)
    observations = {
        "time": np.tile(np.arange(count // 2) * 0.01, 2),
        "qpos": qpos,
        "qvel": qvel,
        "activation": activation,
        "joint_torque_Nm": np.array(torque),
        "actuator_names": ["flexor", "extensor"],
        "dof_names": ["ankle"],
        "recording_ids": np.array(
            ["training"] * (count // 2) + ["untouched"] * (count // 2)
        ),
        "provenance": {
            "status": "synthetic",
            "source": "Invented native-engine fixture",
        },
        "activation_source": "synthetic",
    }
    config = {
        "version": 1,
        "muscles": {
            "flexor": {"force_N": [20.0, 300.0]},
            "extensor": {"force_N": [20.0, 300.0]},
        },
        "target_type": target_type,
        "holdout_recordings": ["untouched"],
        "rom_samples": 5,
        "rollout_steps": 30,
    }
    return source, truth, observations, config


@pytest.mark.parametrize("target_type", ["actuator", "net"])
def test_fit_recovers_native_forces_on_independent_recordings(target_type):
    source, truth, observations, config = _fixture(target_type=target_type)
    before = physical_digest(source)
    original = {
        k: v.copy() for k, v in observations.items() if isinstance(v, np.ndarray)
    }
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    assert result.model is not source
    assert physical_digest(source) == before
    np.testing.assert_allclose(
        result.model.actuator_gainprm[:, 2], [170, 95], atol=2e-6
    )
    np.testing.assert_allclose(
        result.model.actuator_biasprm[:, 2], [170, 95], atol=2e-6
    )
    assert result.report["optimizer"]["success"]
    assert result.report["fitted"]["train"]["rmse_Nm"] < 1e-7
    assert result.report["fitted"]["validation"]["rmse_Nm"] < 1e-7
    assert result.report["baseline"]["validation"]["rmse_Nm"] > 1
    assert result.report["fitted"]["train"]["recordings"] == ["training"]
    assert result.report["fitted"]["validation"]["recordings"] == ["untouched"]
    assert result.report["identifiability"]["full_column_rank"]
    assert result.report["synthetic_only"]
    assert result.profile["provenance"]["status"] == "synthetic"
    with pytest.raises(ValueError, match="synthetic|Synthetic"):
        build_physical_model(source, result.profile)
    built = build_physical_model(source, result.profile, allow_synthetic=True)
    np.testing.assert_allclose(
        built.model.actuator_gainprm[:, :9], result.model.actuator_gainprm[:, :9]
    )
    validated = validate_physical_model(
        truth, observations, config, allow_synthetic=True
    )
    assert validated["metrics"]["all"]["max_abs_error_Nm"] < 1e-12
    assert validated["checks"]["rollout"]["finite"]
    assert len(validated["checks"]["rom"]["ankle"]["moment_arms_m"]) == 5
    for key, value in original.items():
        np.testing.assert_array_equal(observations[key], value)


def test_fit_requires_explicit_synthetic_and_emg_proxy_opt_in():
    source, _, observations, config = _fixture()
    with pytest.raises(ValueError, match="Synthetic observations"):
        fit_physical_model(source, observations, config)
    observations["provenance"] = {
        "status": "measured",
        "source": "Example observation schema",
    }
    observations["activation_source"] = "emg_proxy"
    with pytest.raises(ValueError, match="EMG is not activation"):
        fit_physical_model(source, observations, config)
    config["allow_emg_proxy"] = True
    result = fit_physical_model(source, observations, config)
    assert result.report["activation_source"] == "emg_proxy"
    assert result.profile["provenance"]["status"] == "estimated"


def test_group_split_rejects_frame_leakage_and_missing_holdouts():
    source, _, observations, config = _fixture()
    del config["holdout_recordings"]
    with pytest.raises(ValueError, match="whole-recording validation"):
        fit_physical_model(source, observations, config, allow_synthetic=True)
    observations["split"] = np.array(["train", "validation"] * 32)
    with pytest.raises(ValueError, match="cannot cross"):
        fit_physical_model(source, observations, config, allow_synthetic=True)
    observations["split"] = np.array(["train"] * 32 + ["validation"] * 32)
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    assert result.report["fitted"]["validation"]["samples"] == 32
    config["holdout_recordings"] = ["untouched"]
    with pytest.raises(ValueError, match="not both"):
        fit_physical_model(source, observations, config, allow_synthetic=True)


def test_unidentifiable_force_fit_is_reported_even_with_regularization():
    source, _, observations, config = _fixture(identical_activations=True)
    config["regularization"] = 1e-8
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    assert result.report["identifiability"]["training_data_rank"] == 1
    assert not result.report["identifiability"]["full_column_rank"]
    assert result.report["identifiability"]["condition_number"] is None


@pytest.mark.parametrize(
    "change,match",
    [
        ({"dof_names": ["missing"]}, "hinge joints"),
        ({"activation": np.zeros((64, 1))}, "activation"),
        ({"actuator_names": ["flexor"]}, "all native muscle"),
        ({"qpos": np.full((64, 1), np.nan)}, "finite"),
        ({"qvel": np.full((64, 1), np.inf)}, "finite"),
        ({"joint_torque_Nm": np.zeros((63, 1))}, "joint_torque"),
        ({"activation_source": "EMG"}, "activation_source"),
    ],
)
def test_invalid_observations_are_rejected(change, match):
    source, _, observations, config = _fixture()
    observations.update(change)
    with pytest.raises(ValueError, match=match):
        fit_physical_model(source, observations, config, allow_synthetic=True)


@pytest.mark.parametrize(
    "bounds,match",
    [
        ({"force_N": [-1, 50]}, "nonnegative bounds"),
        ({"force_N": [100, 100]}, "increasing"),
        ({"range0": [0.7, 1.2], "range1": [1.1, 1.6]}, "effective fibre"),
        ({"range0": [0.6, 0.8]}, "nonnegative tendon"),
        ({"bad_parameter": [1, 2]}, "Unknown muscle or parameter"),
    ],
)
def test_unphysical_or_unknown_parameter_bounds_fail(bounds, match):
    source, _, observations, config = _fixture()
    config["muscles"] = {"flexor": bounds}
    with pytest.raises(ValueError, match=match):
        fit_physical_model(source, observations, config, allow_synthetic=True)


def test_operation_range_and_passive_parameter_fit_matches_engine_holdout():
    source, _, observations, config = _fixture()
    truth = copy.copy(source)
    truth.actuator_gainprm[0, :2] = [0.4, 1.6]
    truth.actuator_biasprm[0, :2] = [0.4, 1.6]
    truth.actuator_gainprm[0, 7] = 2.1
    truth.actuator_biasprm[0, 7] = 2.1
    observations["qpos"][:, 0] = np.tile(np.linspace(-0.75, 0.75, 32), 2)
    data = mujoco.MjData(truth)
    for f in range(64):
        data.qpos[:] = observations["qpos"][f]
        data.qvel[:] = observations["qvel"][f]
        data.act[:] = observations["activation"][f]
        mujoco.mj_forward(truth, data)
        observations["joint_torque_Nm"][f] = data.qfrc_actuator
    config["muscles"] = {
        "flexor": {
            "range0": [0.25, 0.55],
            "range1": [1.3, 1.8],
            "passive_force_scale": [0.1, 4.0],
        }
    }
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    assert result.report["fitted"]["validation"]["rmse_Nm"] < 1e-6
    np.testing.assert_allclose(
        result.model.actuator_gainprm[0, :2], [0.4, 1.6], atol=2e-5
    )
    np.testing.assert_allclose(
        result.model.actuator_biasprm[0, :2], [0.4, 1.6], atol=2e-5
    )
    np.testing.assert_allclose(result.model.actuator_biasprm[0, 7], 2.1, atol=2e-4)


def test_actearly_or_nonmuscle_models_are_not_silently_misinterpreted():
    source, _, observations, config = _fixture()
    source.actuator_actearly[0] = True
    with pytest.raises(ValueError, match="actearly=false"):
        fit_physical_model(source, observations, config, allow_synthetic=True)


def test_validation_labels_never_change_the_fitted_training_parameters():
    source, _, observations, config = _fixture()
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    observations["joint_torque_Nm"][32:] += 1000
    changed = fit_physical_model(source, observations, config, allow_synthetic=True)
    np.testing.assert_array_equal(
        changed.model.actuator_gainprm, result.model.actuator_gainprm
    )
    assert changed.report["fitted"]["validation"]["rmse_Nm"] > 999


def test_native_activation_and_individual_force_clipping_are_honoured():
    _, truth, observations, config = _fixture()
    truth.actuator_actlimited[:] = True
    truth.actuator_actrange[:] = [0.2, 0.7]
    truth.actuator_forcelimited[:] = True
    truth.actuator_forcerange[:] = [-110, -5]
    data = mujoco.MjData(truth)
    for frame in range(64):
        data.qpos[:] = observations["qpos"][frame]
        data.qvel[:] = observations["qvel"][frame]
        data.act[:] = observations["activation"][frame]
        mujoco.mj_forward(truth, data)
        observations["joint_torque_Nm"][frame] = data.qfrc_actuator
    report = validate_physical_model(truth, observations, config, allow_synthetic=True)
    assert report["metrics"]["all"]["max_abs_error_Nm"] < 1e-12


def test_polynomial_coupled_joints_use_effective_virtual_work_and_rom():
    _, _, observations, config = _fixture()
    xml = _XML.replace(
        '<site name="tip" pos=".2 0 0"/>',
        '<body name="child" pos=".08 0 0">'
        '<joint name="dependent" axis="0 0 1" range="-.4 .4"/>'
        '<geom size=".02" mass=".2"/>'
        '<site name="tip" pos=".12 0 0"/></body>',
    ).replace(
        "<actuator>",
        '<equality><joint joint1="dependent" joint2="ankle" polycoef="0 .5 0 0 0"/></equality><actuator>',
    )
    source = mujoco.MjModel.from_xml_string(xml)
    truth = copy.copy(source)
    truth.actuator_gainprm[:, 2] = truth.actuator_biasprm[:, 2] = [170, 95]
    observations["qpos"] = np.c_[observations["qpos"], observations["qpos"] * 0.5]
    observations["qvel"] = np.c_[observations["qvel"], observations["qvel"] * 0.5]
    data = mujoco.MjData(truth)
    for frame in range(64):
        data.qpos[:] = observations["qpos"][frame]
        data.qvel[:] = observations["qvel"][frame]
        data.act[:] = observations["activation"][frame]
        mujoco.mj_forward(truth, data)
        observations["joint_torque_Nm"][frame] = data.qfrc_actuator @ [1.0, 0.5]
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    np.testing.assert_allclose(
        result.model.actuator_gainprm[:, 2], [170, 95], atol=1e-6
    )
    assert result.report["fitted"]["validation"]["rmse_Nm"] < 1e-8
    observations["qpos"][0, 1] += 0.01
    with pytest.raises(ValueError, match="polynomial equalities"):
        validate_physical_model(truth, observations, config, allow_synthetic=True)


@pytest.mark.parametrize("kind", ["actuation", "group", "joint_clip"])
def test_disabled_or_aggregate_actuation_is_rejected_explicitly(kind):
    source, _, observations, config = _fixture()
    if kind == "actuation":
        source.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    elif kind == "group":
        source.opt.disableactuator = 1
    else:
        source.jnt_actfrclimited[0] = True
    with pytest.raises(ValueError, match="enabled|clipping"):
        fit_physical_model(source, observations, config, allow_synthetic=True)


def test_local_numeric_loader_metadata_connects_to_fitter(tmp_path):
    from myosuite_soma.physical_data import load_observations

    source, _, observations, config = _fixture()
    metadata = {
        "version": 1,
        "provenance": observations["provenance"],
        "activation_source": observations["activation_source"],
        "target_type": "actuator",
        "units": {
            "time": "s",
            "qpos": "SI",
            "qvel": "SI/s",
            "joint_torque_Nm": "Nm",
            "activation": "unitless",
        },
        "conventions": {
            "torque_sign": "same positive direction as native hinge",
            "gravity": "excluded",
        },
    }
    arrays = {
        key: value
        for key, value in observations.items()
        if key not in ("provenance", "activation_source")
    }
    path = tmp_path / "observations.npz"
    np.savez(path, **arrays, metadata_json=np.array(json.dumps(metadata)))
    loaded = load_observations(path)
    result = fit_physical_model(source, loaded, config, allow_synthetic=True)
    assert result.report["fitted"]["validation"]["rmse_Nm"] < 1e-7
    assert (
        result.report["provenance"]["observation_file_sha256"] == loaded["input_sha256"]
    )
    assert result.report["provenance"]["conventions"] == metadata["conventions"]
    json.dumps(result.report, allow_nan=False)
    loaded["target_type"] = "net"
    with pytest.raises(ValueError, match="target_type disagrees"):
        fit_physical_model(source, loaded, config, allow_synthetic=True)


def test_nonphysical_inertia_and_nonboolean_gates_fail():
    _, truth, observations, config = _fixture()
    truth.body_inertia[1] = [1, 0.1, 0.1]
    with pytest.raises(ValueError, match="triangle inequalities"):
        validate_physical_model(truth, observations, config, allow_synthetic=True)
    with pytest.raises(ValueError, match="must be boolean"):
        fit_physical_model(truth, observations, config, allow_synthetic="yes")


def test_legacy_negative_effective_slack_is_reported_not_relabelled_as_anatomy():
    source, _, observations, config = _fixture()
    source.actuator_gainprm[0, :2] = source.actuator_biasprm[0, :2] = [0.9, 1.1]
    truth = copy.copy(source)
    truth.actuator_gainprm[:, 2] = truth.actuator_biasprm[:, 2] = [170, 95]
    data = mujoco.MjData(truth)
    for frame in range(64):
        data.qpos[:] = observations["qpos"][frame]
        data.qvel[:] = observations["qvel"][frame]
        data.act[:] = observations["activation"][frame]
        mujoco.mj_forward(truth, data)
        observations["joint_torque_Nm"][frame] = data.qfrc_actuator
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    np.testing.assert_allclose(
        result.model.actuator_gainprm[:, 2], [170, 95], atol=1e-6
    )
    lengths = result.report["checks"]["native_muscle_lengths"]["flexor"]
    assert lengths["effective_inelastic_tendon_length_m"] < 0
    assert lengths["anatomical_length_interpretation_valid"] is False
    assert "range" not in result.profile["muscles"]["flexor"]
    np.testing.assert_array_equal(
        result.model.actuator_gainprm[0, :2], source.actuator_gainprm[0, :2]
    )
    config["muscles"] = {"flexor": {"range0": [0.85, 0.95]}}
    with pytest.raises(ValueError, match="nonnegative tendon"):
        fit_physical_model(source, observations, config, allow_synthetic=True)


def test_observation_model_bindings_distinguish_source_from_fitted_candidate():
    source, _, observations, config = _fixture()
    observations["conventions"] = {
        "source_model_sha256": physical_digest(source),
        "source_kinematic_sha256": model_digest(source),
    }
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    assert result.report["validated_initial_physical_source"] is True
    report = validate_physical_model(
        result.model, observations, config, allow_synthetic=True
    )
    assert report["same_input_physics"] is False
    assert report["observation_source_model_sha256"] == physical_digest(source)
    assert report["candidate_model_sha256"] == physical_digest(result.model)
    assert report["metrics"]["all"]["rmse_Nm"] < 1e-7
    observations["conventions"]["source_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="initial calibration model"):
        fit_physical_model(source, observations, config, allow_synthetic=True)
    observations["conventions"]["source_kinematic_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="kinematic digest"):
        validate_physical_model(source, observations, config, allow_synthetic=True)


def test_torque_fit_checks_coupled_velocities_with_explicit_residual_tolerance():
    _, _, observations, config = _fixture()
    xml = _XML.replace(
        '<site name="tip" pos=".2 0 0"/>',
        '<site name="tip" pos=".2 0 0"/><body name="child"><joint name="dependent" axis="0 0 1" range="-.4 .4"/><geom size=".02" mass=".2"/></body>',
    ).replace(
        "<actuator>",
        '<equality><joint joint1="dependent" joint2="ankle" polycoef="0 .5 0 0 0"/></equality><actuator>',
    )
    source = mujoco.MjModel.from_xml_string(xml)
    observations["qpos"] = np.c_[observations["qpos"], observations["qpos"] * 0.5]
    observations["qvel"] = np.c_[observations["qvel"], observations["qvel"] * 0.5 + 0.1]
    with pytest.raises(ValueError, match="velocity violates scalar joint equality"):
        fit_physical_model(source, observations, config, allow_synthetic=True)
    observations["conventions"] = {
        "equality_kinematics_tolerances": {"velocity_abs": 0.2}
    }
    result = fit_physical_model(source, observations, config, allow_synthetic=True)
    checks = result.report["joint_equality_kinematics"]
    assert checks["velocity_abs_tolerance"] == 0.2
    assert checks["by_joint"]["dependent"][
        "maximum_abs_velocity_residual"
    ] == pytest.approx(0.1)
