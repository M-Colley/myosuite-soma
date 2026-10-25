"""Independent native geometric truths for reviewed MTU path fitting."""

import copy
import json

import mujoco
import numpy as np
import pytest

from myosuite_soma.physical_model import physical_digest
from myosuite_soma.physical_paths import fit_path_lengths
from myosuite_soma.soma import model_digest


_XML = """<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
<worldbody><site name="left" pos="0 .4 0"/><site name="right" pos="0 -.4 0"/>
<body name="leg"><joint name="ankle" axis="0 0 1" range="-.8 .8"/>
<geom size=".04" mass="1"/><site name="tip" pos=".2 0 0"/></body></worldbody>
<tendon><spatial name="leftpath"><site site="left"/><site site="tip"/></spatial>
<spatial name="rightpath"><site site="right"/><site site="tip"/></spatial></tendon>
<actuator><muscle name="flexor" tendon="leftpath" force="-1" scale="200" lengthrange=".3 .6"/>
<muscle name="extensor" tendon="rightpath" force="100" lengthrange=".3 .6"/></actuator></mujoco>"""


def _fixture():
    source = mujoco.MjModel.from_xml_string(_XML)
    truth = copy.copy(source)
    truth.site_pos[truth.site("tip").id, 0] = 0.245
    poses = np.linspace(-0.7, 0.7, 40)[:, None]
    data = mujoco.MjData(truth)
    lengths = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(truth, data)
        lengths.append(data.actuator_length.copy())
    observations = {
        "time": np.tile(np.arange(20) * 0.01, 2),
        "qpos": poses,
        "actuator_names": ["flexor", "extensor"],
        "mtu_length_m": np.array(lengths),
        "recording_ids": np.array(["training"] * 20 + ["untouched"] * 20),
        "provenance": {
            "status": "synthetic",
            "source": "Independently generated native geometric observations",
        },
        "units": {"time": "s", "qpos": "SI", "mtu_length_m": "m"},
        "conventions": {
            "source_model_sha256": physical_digest(source),
            "source_kinematic_sha256": model_digest(source),
        },
    }
    config = {
        "version": 1,
        "holdout_recordings": ["untouched"],
        "sites": {"tip": {"lower_m": [0.1, 0, 0], "upper_m": [0.3, 0, 0]}},
    }
    return source, observations, config


def test_native_path_truth_recovers_site_and_preserves_source_and_strength():
    source, observations, config = _fixture()
    before = physical_digest(source)
    force = source.actuator_gainprm[0, 3] / source.actuator_acc0[0]
    arrays = {
        key: value.copy()
        for key, value in observations.items()
        if isinstance(value, np.ndarray)
    }
    result = fit_path_lengths(source, observations, config, allow_synthetic=True)
    assert physical_digest(source) == before
    assert result.model is not source
    np.testing.assert_allclose(
        result.model.site_pos[source.site("tip").id], [0.245, 0, 0], atol=1e-7
    )
    assert result.report["optimizer"]["success"]
    assert result.report["baseline"]["validation"]["rmse_m"] > 0.001
    assert result.report["fitted"]["validation"]["rmse_m"] < 1e-8
    assert result.report["identifiability"]["full_column_rank"]
    assert result.model.actuator_gainprm[0, 2] == pytest.approx(force)
    assert result.model.actuator_biasprm[0, 2] == pytest.approx(force)
    assert result.report["physical_rebuild"]["lengthrange_recomputed"]
    assert not np.array_equal(
        result.model.actuator_lengthrange, source.actuator_lengthrange
    )
    assert result.profile["recompute_lengthrange"] is True
    assert result.profile["provenance"]["status"] == "synthetic"
    json.dumps(result.report, allow_nan=False)
    for name, array in arrays.items():
        np.testing.assert_array_equal(observations[name], array)


def test_holdout_targets_are_excluded_from_optimization():
    source, observations, config = _fixture()
    first = fit_path_lengths(source, observations, config, allow_synthetic=True)
    observations["mtu_length_m"][20:] += 0.05
    changed = fit_path_lengths(source, observations, config, allow_synthetic=True)
    np.testing.assert_array_equal(first.model.site_pos, changed.model.site_pos)
    assert changed.report["fitted"]["validation"]["rmse_m"] > 0.049


def test_unrelated_site_coordinate_is_reported_unidentifiable():
    source, observations, config = _fixture()
    # A site's z coordinate is locally symmetric around this planar fixture.
    # A model-surface landmark absent from all paths has exactly zero sensitivity.
    source = mujoco.MjModel.from_xml_string(
        _XML.replace("<worldbody>", '<worldbody><site name="unrelated" pos="0 0 0"/>')
    )
    observations["conventions"] = {}
    config["sites"] = {"unrelated": {"lower_m": [-0.1, 0, 0], "upper_m": [0.1, 0, 0]}}
    config["regularization"] = 0.1
    result = fit_path_lengths(source, observations, config, allow_synthetic=True)
    assert result.report["identifiability"]["training_data_rank"] == 0
    assert not result.report["identifiability"]["full_column_rank"]


@pytest.mark.parametrize(
    "kind",
    [
        "synthetic",
        "truthy_gate",
        "wrong_model",
        "wrong_kinematics",
        "wrong_units",
        "nonpositive_length",
        "missing_holdout",
        "all_holdout",
        "missing_site",
        "inverted_bounds",
        "all_fixed",
        "wrong_shape",
        "nonunit_gear",
    ],
)
def test_invalid_sources_observations_and_fitting_contracts_fail(kind):
    source, observations, config = _fixture()
    allow = True
    if kind == "synthetic":
        allow = False
    elif kind == "truthy_gate":
        allow = "yes"
    elif kind == "wrong_model":
        observations["conventions"]["source_model_sha256"] = "0" * 64
    elif kind == "wrong_kinematics":
        observations["conventions"]["source_kinematic_sha256"] = "0" * 64
    elif kind == "wrong_units":
        observations["units"]["mtu_length_m"] = "mm"
    elif kind == "nonpositive_length":
        observations["mtu_length_m"][0, 0] = 0
    elif kind == "missing_holdout":
        config["holdout_recordings"] = ["absent"]
    elif kind == "all_holdout":
        config["holdout_recordings"] = ["training", "untouched"]
    elif kind == "missing_site":
        config["sites"] = {"absent": config["sites"]["tip"]}
    elif kind == "inverted_bounds":
        config["sites"]["tip"]["lower_m"][0] = 1
    elif kind == "all_fixed":
        config["sites"]["tip"]["upper_m"] = config["sites"]["tip"]["lower_m"][:]
    elif kind == "wrong_shape":
        observations["mtu_length_m"] = observations["mtu_length_m"][:-1]
    else:
        source.actuator_gear[0, 0] = 2
        observations["conventions"] = {}
    with pytest.raises(ValueError):
        fit_path_lengths(source, observations, config, allow_synthetic=allow)
