"""Local physical commands exercise actual native exports and calibration."""

import json

import mujoco
import numpy as np
import pytest

from myosuite_soma import cli, physical
from myosuite_soma.physical_data import load_observations, save_observations
from myosuite_soma.soma import save_animation

from .test_compliant_muscle import (
    _config as compliant_config,
    _source as compliant_source,
)
from .test_package_boundary import _run
from .test_physical_calibration import _XML, _fixture as calibration_fixture
from .test_physical_data import _architecture, _metadata
from .test_physical_geometry import PROVENANCE, _cache_fixture
from .test_physical_model import XML, _profile


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def test_help_and_dataset_catalog_do_not_need_assets(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["physical", "--help"])
    assert stopped.value.code == 0
    assert "simulate-compliant" in capsys.readouterr().out
    cli.main(["physical", "datasets"])
    report = json.loads(capsys.readouterr().out)
    assert {"charles", "hand", "nordez", "bizet", "addbiomechanics"} <= set(report)


@pytest.mark.parametrize("binary", [False, True])
def test_build_cli_exports_native_variant_without_torch_or_myosuite(tmp_path, binary):
    source = tmp_path / ("source.mjb" if binary else "source.xml")
    if binary:
        mujoco.mj_saveModel(mujoco.MjModel.from_xml_string(XML), str(source))
    else:
        source.write_text(XML)
    profile = _json(
        tmp_path / "profile.json", _profile(muscles={"muscle": {"force_N": 321}})
    )
    output = tmp_path / "converted"
    arguments = [
        "physical",
        "build",
        "--model",
        str(source),
        "--profile",
        profile,
        "--output",
        str(output),
        "--allow-synthetic",
    ]
    _run(
        f"""
from myosuite_soma.cli import main
main({arguments!r})
assert 'torch' not in sys.modules
assert not any(n == 'myosuite' or n.startswith('myosuite.') for n in sys.modules)
""",
        block_torch=True,
    )
    model = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
    assert model.actuator_gainprm[0, 2] == 321
    assert model.actuator_biasprm[0, 2] == 321
    report = json.loads((output / "report.json").read_text())
    assert report["provenance"]["status"] == "synthetic"
    if binary:
        assert report["editable_spec_available"] is False
    original = (output / "model.mjb").read_bytes()
    with pytest.raises(SystemExit) as exists:
        cli.main(arguments)
    assert exists.value.code == 2
    assert (output / "model.mjb").read_bytes() == original


def test_build_requires_explicit_synthetic_permission_before_creating_output(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text(XML)
    profile = _json(tmp_path / "profile.json", _profile())
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "physical",
                "build",
                "--model",
                str(source),
                "--profile",
                profile,
                "--output",
                str(output),
            ]
        )
    assert stopped.value.code == 2
    assert not output.exists()


def test_fit_and_validate_cli_use_recording_holdouts_and_export_reusable_model(
    tmp_path,
):
    _, _, observations, config = calibration_fixture()
    observations["units"] = _metadata()["units"]
    source = tmp_path / "source.xml"
    source.write_text(_XML)
    cache = save_observations(observations, tmp_path / "observations.npz")
    config_path = _json(tmp_path / "fit.json", config)
    output = tmp_path / "fitted"
    common = [
        "--observations",
        str(cache),
        "--config",
        config_path,
        "--allow-synthetic",
    ]
    cli.main(
        ["physical", "fit", "--model", str(source), "--output", str(output), *common]
    )
    model = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
    np.testing.assert_allclose(model.actuator_gainprm[:, 2], [170, 95], atol=2e-6)
    report = json.loads((output / "report.json").read_text())
    assert report["calibration"]["fitted"]["validation"]["recordings"] == ["untouched"]
    assert report["calibration"]["fitted"]["validation"]["rmse_Nm"] < 1e-7
    assert (output / "fitted-profile.json").is_file()
    validation = tmp_path / "validation.json"
    cli.main(
        [
            "physical",
            "validate",
            "--model",
            str(output / "model.mjb"),
            "--output",
            str(validation),
            *common,
        ]
    )
    result = json.loads(validation.read_text())
    assert result["metrics"]["all"]["max_abs_error_Nm"] < 1e-6
    assert result["checks"]["rollout"]["finite"] is True


def test_architecture_and_observation_imports_round_trip_local_files(tmp_path):
    _architecture(tmp_path)
    source = tmp_path / "synthetic.docx"
    data_path = tmp_path / "architecture.json"
    profile_path = tmp_path / "profile.json"
    mapping = _json(
        tmp_path / "mapping.json",
        {"Imaginary calf": {"actuator": "muscle", "force_source": "reported"}},
    )
    cli.main(
        [
            "physical",
            "import-architecture",
            "--source",
            "charles",
            "--input",
            str(source),
            "--mapping",
            mapping,
            "--profile-output",
            str(profile_path),
            "--output",
            str(data_path),
            "--subject",
            "invented-test",
        ]
    )
    profile = json.loads(profile_path.read_text())
    assert profile["muscles"]["muscle"]["force_N"] == 450
    assert json.loads(data_path.read_text())["provenance"]["status"] == "estimated"
    _, _, observations, _ = calibration_fixture()
    observations["units"] = _metadata()["units"]
    original = save_observations(observations, tmp_path / "source-observations.npz")
    target = tmp_path / "portable-observations.npz"
    cli.main(
        [
            "physical",
            "import-observations",
            "--input",
            str(original),
            "--output",
            str(target),
        ]
    )
    np.testing.assert_array_equal(
        load_observations(target)["joint_torque_Nm"], observations["joint_torque_Nm"]
    )


def test_allometry_and_cache_capsule_commands_create_explicit_profiles(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text(XML)
    scales = _json(tmp_path / "scales.json", {"lower": 1.1})
    provenance = _json(tmp_path / "provenance.json", PROVENANCE)
    output = tmp_path / "allometry.json"
    cli.main(
        [
            "physical",
            "allometry",
            "--model",
            str(source),
            "--scales",
            scales,
            "--provenance",
            provenance,
            "--mass-policy",
            "target_total",
            "--target-mass-kg",
            "5",
            "--output",
            str(output),
        ]
    )
    profile = json.loads(output.read_text())
    assert sum(e["mass_kg"] for e in profile["bodies"].values()) == pytest.approx(5)
    model, poses, animation, assignments, _ = _cache_fixture()
    cache = tmp_path / "anatomy.npz"
    save_animation(cache, animation)
    trajectory = tmp_path / "reference.npz"
    np.savez(
        trajectory,
        qpos=poses,
        time=animation.time,
        qvel=np.zeros((2, model.nv)),
        joint_names=np.array(["hinge"]),
    )
    assignment_path = _json(tmp_path / "assignments.json", assignments)
    capsule = tmp_path / "capsule-profile.json"
    cli.main(
        [
            "physical",
            "fit-capsules",
            "--model",
            str(source),
            "--cache",
            str(cache),
            "--trajectory",
            str(trajectory),
            "--assignments",
            assignment_path,
            "--provenance",
            provenance,
            "--output",
            str(capsule),
        ]
    )
    generated = json.loads(capsule.read_text())
    assert "lower-geom" in generated["geoms"]
    assert generated["recompute_lengthrange"] is True
    bad = tmp_path / "wrong-time.npz"
    np.savez(bad, qpos=poses, time=animation.time + 1)
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "physical",
                "fit-capsules",
                "--model",
                str(source),
                "--cache",
                str(cache),
                "--trajectory",
                str(bad),
                "--assignments",
                assignment_path,
                "--provenance",
                provenance,
                "--output",
                str(tmp_path / "bad.json"),
            ]
        )
    assert stopped.value.code == 2
    assert not (tmp_path / "bad.json").exists()


def test_compliant_cli_generates_actual_forward_motion_and_research_report(tmp_path):
    model = compliant_source()
    source = tmp_path / "source.mjb"
    mujoco.mj_saveModel(model, str(source))
    parameters = _json(tmp_path / "parameters.json", compliant_config())
    controls = tmp_path / "controls.npz"
    np.savez(controls, ctrl=np.full((30, 1), 0.5))
    output = tmp_path / "compliant"
    arguments = [
        "physical",
        "simulate-compliant",
        "--model",
        str(source),
        "--parameters",
        parameters,
        "--controls",
        str(controls),
        "--output",
        str(output),
    ]
    with pytest.raises(SystemExit) as blocked:
        cli.main(arguments)
    assert blocked.value.code == 2
    assert not output.exists()
    cli.main([*arguments, "--allow-synthetic"])
    with np.load(output / "rollout.npz", allow_pickle=False) as archive:
        assert len(archive["time"]) == 30
        assert archive["qpos"][-1, 0] < 0
        assert archive["tendon_force_N"].max() > 0
        assert np.abs(archive["force_balance_residual_N"]).max() < 1e-5
    report = json.loads((output / "report.json").read_text())
    assert report["research_only"] is True
    assert report["physiological_accuracy_validated"] is False


def test_optional_environment_is_closed_when_build_rejects_input(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    closed = []
    fake = SimpleNamespace(
        unwrapped=SimpleNamespace(model=mujoco.MjModel.from_xml_string(XML)),
        close=lambda: closed.append(True),
    )
    monkeypatch.setitem(
        sys.modules, "myosuite", SimpleNamespace(make_env=lambda name: fake)
    )
    profile = _json(tmp_path / "bad-profile.json", _profile())
    with pytest.raises(SystemExit):
        physical.main(
            [
                "build",
                "--env",
                "synthetic",
                "--profile",
                profile,
                "--output",
                str(tmp_path / "none"),
            ]
        )
    assert closed == [True]
