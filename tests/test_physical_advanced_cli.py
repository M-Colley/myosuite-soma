"""Public command wiring for path fitting and measured-load dynamics checks."""

import json

import mujoco
import numpy as np
import pytest

from myosuite_soma.cli import main
from .test_physical_paths import _XML, _fixture
from .test_physical_dynamics import _base, _free_model


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def test_path_fitting_cli_creates_reloadable_independent_model(tmp_path):
    _, observations, config = _fixture()
    source = tmp_path / "source.xml"
    source.write_text(_XML)
    data, options = tmp_path / "paths.json", tmp_path / "bounds.json"
    data.write_text(json.dumps(_plain(observations)))
    options.write_text(json.dumps(config))
    output = tmp_path / "fit"
    main(
        [
            "physical",
            "fit-paths",
            "--model",
            str(source),
            "--observations",
            str(data),
            "--config",
            str(options),
            "--allow-synthetic",
            "--output",
            str(output),
        ]
    )
    model = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
    np.testing.assert_allclose(
        model.site_pos[model.site("tip").id], [0.245, 0, 0], atol=1e-7
    )
    report = json.loads((output / "report.json").read_text())
    assert report["path_calibration"]["synthetic_only"]
    assert report["path_calibration"]["fitted"]["validation"]["rmse_m"] < 1e-8
    assert (output / "fitted-profile.json").is_file()


def test_inverse_dynamics_cli_uses_measured_loads_without_activation(tmp_path):
    model = _free_model()
    source = tmp_path / "source.mjb"
    mujoco.mj_saveModel(model, str(source))
    observations = _base(model, np.tile(model.qpos0, (3, 1)))
    observations["external_forces"] = [
        {
            "body": "body",
            "point_world_m": np.zeros((3, 3)),
            "force_world_N": np.tile([0, 0, 19.62], (3, 1)),
            "torque_world_Nm": np.zeros((3, 3)),
        }
    ]
    data = tmp_path / "loads.json"
    data.write_text(json.dumps(_plain(observations)))
    output = tmp_path / "validation.json"
    arguments = [
        "physical",
        "validate-dynamics",
        "--model",
        str(source),
        "--observations",
        str(data),
        "--output",
        str(output),
    ]
    with pytest.raises(SystemExit) as stopped:
        main(arguments)
    assert stopped.value.code == 2 and not output.exists()
    main([*arguments, "--allow-synthetic"])
    result = json.loads(output.read_text())
    assert result["muscle_activation_required"] is False
    assert result["free_base_residuals"]["root"]["force_rms_norm_N"] < 1e-12
    assert result["native_contact_forces_used"] is False
