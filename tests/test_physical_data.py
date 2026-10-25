"""Invented architecture/recording fixtures exercise local data adapters."""

import csv
import json
from zipfile import ZipFile
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import pytest

from myosuite_soma import physical_data as data


def _docx(path, tables):
    body = []
    for table in tables:
        rows = []
        for row in table:
            cells = [
                f"<w:tc><w:p><w:r><w:t>{escape(str(c))}</w:t></w:r></w:p></w:tc>"
                for c in row
            ]
            rows.append("<w:tr>" + "".join(cells) + "</w:tr>")
        body.append("<w:tbl>" + "".join(rows) + "</w:tbl>")
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            + "".join(body)
            + "</w:body></w:document>",
        )
    return path


def _architecture(tmp_path, source="charles"):
    # Header layout matches the published table; all numerical values are invented.
    table = [
        [
            "Muscle",
            "Muscle Volume (cm3)",
            "Belly Length (mm)",
            "Optimal fiber length (mm)",
            "Lf:Lm",
            "Pennation angle (°)",
            "PCSA (mm2)",
            "Fmax (N)",
            "Fmax (%BW)",
        ],
        ["Imaginary calf", "123", "250", "75 ± 5", ".3", "20 ± 2", "1500", "450", "60"],
        ["Imaginary flexor", "12", "200", "60 ± 4", ".3", "N/A", "200", "60", "8"],
        ["Ankle plantarflexors", "60", "225", "70", ".3", "20", "800", "240", "30"],
    ]
    path = _docx(tmp_path / "synthetic.docx", [table])
    return data.import_architecture(source, path, subject="synthetic-table")


def _metadata():
    return {
        "provenance": {
            "status": "synthetic",
            "source": "invented calibration observations",
        },
        "activation_source": "synthetic",
        "units": {
            "time": "s",
            "qpos": "SI",
            "qvel": "SI/s",
            "joint_torque_Nm": "Nm",
            "activation": "unitless",
        },
    }


def _arrays():
    return {
        "time": np.array([0, 0.1, 0, 0.1]),
        "qpos": np.array([[0], [0.1], [0.2], [0.3]]),
        "qvel": np.array([[1], [1], [1], [1]]),
        "activation": np.array([[0.1], [0.2], [0.3], [0.4]]),
        "joint_torque_Nm": np.array([[1], [2], [3], [4]]),
        "recording_ids": np.array(["a", "a", "b", "b"]),
        "actuator_names": np.array(["calf"]),
        "dof_names": np.array(["ankle"]),
        "split": np.array(["train", "train", "validation", "validation"]),
    }


def test_catalog_is_independent_and_never_downloads():
    catalog = data.dataset_catalog()
    assert set(catalog) == {"charles", "hand", "nordez", "bizet", "addbiomechanics"}
    catalog["charles"]["limitations"].clear()
    assert data.dataset_catalog()["charles"]["limitations"]


def test_native_charles_headers_convert_units_means_and_preserve_uncertainty(tmp_path):
    result = _architecture(tmp_path)
    assert len(result["rows"]) == 2
    row = result["rows"][0]
    assert row["volume_m3"] == pytest.approx(123e-6)
    assert row["fibre_length_m"] == pytest.approx(0.075)
    assert row["pcsa_m2"] == pytest.approx(0.0015)
    assert row["pennation_deg"] == 20
    assert row["pcsa_kind"] == "projected"
    assert row["force_kind"] == "estimated"
    assert "± 5" in row["raw_cells"]["fibre_length_m"]
    assert result["provenance"]["status"] == "estimated"
    assert len(result["source_sha256"]) == 64
    assert "pennation_deg" not in result["rows"][1]


def test_word_formatting_runs_do_not_insert_spaces_into_numbers(tmp_path):
    path = tmp_path / "fragmented.docx"
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Muscle</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Muscle Volume (cm3)</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>Invented calf</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>4</w:t></w:r><w:r><w:t>56</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    result = data.import_architecture("charles", path)
    assert result["rows"][0]["volume_m3"] == pytest.approx(456e-6)
    assert result["rows"][0]["raw_cells"]["volume_m3"] == "456"
    with pytest.raises(ValueError, match="Ambiguous"):
        data._number("4 56")


def test_hand_main_table_csv_and_perpendicular_pcsa(tmp_path):
    path = tmp_path / "hand-table.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows(
            [
                [
                    "Muscle",
                    "Number of via points",
                    "Origin",
                    "Insertion",
                    "Optimum Fiber Length (mm)",
                    "Tendon length (mm)",
                    "Pennation angle (deg)",
                    "Mass (g)",
                    "PCSA (mm2)",
                ],
                ["Wrist", "", "", "", "", "", "", "", ""],
                ["Invented wrist", "1", "arm", "hand", "40", "120", "12", "4", "100"],
                ["Absent muscle", "X", "Not present", "", "x", "x", "x", "x", "x"],
            ]
        )
    result = data.import_architecture("hand", path, subject="invented")
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["pcsa_kind"] == "perpendicular"
    assert row["fibre_length_m"] == 0.04
    profile = data.architecture_profile(
        result, {"Invented wrist": {"actuator": "wrist", "specific_tension_pa": 200000}}
    )
    assert profile["muscles"]["wrist"]["architecture"] == {
        "pcsa_m2": pytest.approx(0.0001),
        "pcsa_kind": "perpendicular",
        "pennation_deg": 12.0,
        "specific_tension_pa": 200000.0,
    }


def test_source_allocations_do_not_duplicate_force_or_volume(tmp_path):
    architecture = _architecture(tmp_path)
    profile = data.architecture_profile(
        architecture,
        {
            "Imaginary calf": [
                {"actuator": "head1", "specific_tension_pa": 300000, "fraction": 0.3},
                {"actuator": "head2", "specific_tension_pa": 300000, "fraction": 0.7},
            ]
        },
    )
    assert sum(
        x["architecture"]["volume_m3"] for x in profile["muscles"].values()
    ) == pytest.approx(123e-6)
    direct = data.architecture_profile(
        architecture,
        {"Imaginary calf": {"actuator": "head", "force_source": "reported"}},
    )
    assert direct["muscles"]["head"]["force_N"] == 450
    with pytest.raises(ValueError, match="fractions"):
        data.architecture_profile(
            architecture,
            {
                "Imaginary calf": [
                    {"actuator": "a", "force_source": "reported"},
                    {"actuator": "b", "force_source": "reported"},
                ]
            },
        )
    with pytest.raises(ValueError, match="specific_tension"):
        data.architecture_profile(architecture, {"Imaginary calf": {"actuator": "a"}})
    # Missing pennation uses the explicitly projected Charles PCSA, with no second cosine.
    projected = data.architecture_profile(
        architecture,
        {"Imaginary flexor": {"actuator": "a", "specific_tension_pa": 300000}},
    )
    assert projected["muscles"]["a"]["architecture"]["pcsa_kind"] == "projected"
    assert "pennation_deg" not in projected["muscles"]["a"]["architecture"]


def test_hand_attachments_keep_source_frame_and_transform_to_native_site(tmp_path):
    path = _docx(
        tmp_path / "attachment.docx",
        [
            [
                ["Muscle", "Abb.", "Po.", "Seg.", "Position (mm)", "Info"],
                ["", "", "", "", "X", "Y", "Z", ""],
                ["Invented muscle", "IM", "I", "hand", "10", "-20", "30", ""],
                ["", "", "Via", "arm", "20", "0", "40", "note"],
                ["", "", "Op", "arm", "x", "x", "x", "absent"],
            ]
        ],
    )
    result = data.import_hand_attachments(path)
    assert len(result["points"]) == 2
    np.testing.assert_allclose(result["points"][0]["position_m"], [0.01, -0.02, 0.03])
    assert result["points"][1]["muscle_name"] == "Invented muscle"
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="arm" pos=".1 .2 .3" quat=".7071067812 0 0 .7071067812"><geom size=".03"/><site name="attachment" pos="0 0 0"/></body></worldbody></mujoco>'
    )
    before = model.site_pos.copy()
    profile = data.attachment_profile(
        result,
        model,
        {result["points"][0]["id"]: "attachment"},
        rotation=np.eye(3),
        translation_m=[0.5, 0.5, 0.5],
        scale=2,
    )
    # Independently use the native body's rest rotation to check the final world coordinate.
    point = np.array(profile["sites"]["attachment"]["pos_m"])
    scratch = mujoco.MjData(model)
    mujoco.mj_forward(model, scratch)
    actual = scratch.xmat[1].reshape(3, 3) @ point + scratch.xpos[1]
    np.testing.assert_allclose(actual, [0.52, 0.46, 0.56], atol=1e-8)
    np.testing.assert_array_equal(model.site_pos, before)
    from myosuite_soma.physical_model import build_physical_model

    compiled = build_physical_model(model, profile)
    changed = mujoco.MjData(compiled.model)
    mujoco.mj_forward(compiled.model, changed)
    np.testing.assert_allclose(changed.site_xpos[0], [0.52, 0.46, 0.56], atol=1e-8)


def test_docx_entities_and_duplicate_members_are_rejected(tmp_path):
    path = tmp_path / "unsafe.docx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml", '<!DOCTYPE doc [<!ENTITY x "payload">]><doc/>'
        )
    with pytest.raises(ValueError, match="entities"):
        data.import_architecture("charles", path)
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", "<doc/>")
        with pytest.warns(UserWarning):
            archive.writestr("word/document.xml", "<doc/>")
    with pytest.raises(ValueError, match="duplicated"):
        data.import_architecture("charles", path)


def test_numeric_npz_observations_are_safe_and_keep_recording_splits(tmp_path):
    path = tmp_path / "observations.npz"
    np.savez(path, **_arrays(), metadata_json=np.array(json.dumps(_metadata())))
    result = data.load_observations(path)
    np.testing.assert_array_equal(result["time"], [0, 0.1, 0, 0.1])
    assert result["provenance"]["status"] == "synthetic"
    assert result["activation_source"] == "synthetic"
    assert len(result["input_sha256"]) == 64
    np.savez(path, **_arrays(), metadata_json=np.array(_metadata(), dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        data.load_observations(path)


def test_wide_csv_observations_use_explicit_names_units_and_signs(tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text(
        "time,recording_id,split,qpos:0,qvel:0,activation:calf,torque:ankle\n0,a,train,0,1,.2,-4\n.1,a,train,.1,1,.4,-8\n0,b,validation,0,1,.3,-6\n"
    )
    metadata = {**_metadata(), "actuator_names": ["calf"], "dof_names": ["ankle"]}
    result = data.load_observations(path, metadata=metadata)
    np.testing.assert_array_equal(result["joint_torque_Nm"].ravel(), [-4, -8, -6])
    assert list(result["dof_names"]) == ["ankle"]
    bad = dict(metadata)
    bad["units"] = {**bad["units"], "qpos": "degrees"}
    with pytest.raises(ValueError, match="units"):
        data.load_observations(path, metadata=bad)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("activation", np.full((4, 1), 2.0), "Activation"),
        ("time", np.array([0.0, 0.0, 0.0, 0.1]), "increase"),
        ("time", np.array(0.0), "vector"),
        ("qpos", np.full((4, 1), np.nan), "numeric"),
        (
            "split",
            np.array(["train", "validation", "validation", "validation"]),
            "leak",
        ),
        ("actuator_names", np.array(["a", "b"]), "columns"),
    ],
)
def test_bad_observations_cannot_silently_change_interpretation(
    tmp_path, key, value, match
):
    path = tmp_path / "bad.npz"
    arrays = _arrays()
    arrays[key] = value
    np.savez(path, **arrays, metadata_json=np.array(json.dumps(_metadata())))
    with pytest.raises(ValueError, match=match):
        data.load_observations(path)


def test_json_output_never_overwrites_existing_file(tmp_path):
    path = data.save_json({"source": "invented"}, tmp_path / "result.json")
    with pytest.raises(ValueError, match="new"):
        data.save_json({"source": "changed"}, path)
    assert json.loads(path.read_text())["source"] == "invented"


def test_saved_observation_archive_round_trip_preserves_metadata_without_pickle(
    tmp_path,
):
    original = {
        **_metadata(),
        **_arrays(),
        "conventions": {"torque": "invented signed native actuator torque"},
    }
    path = data.save_observations(original, tmp_path / "portable.npz")
    loaded = data.load_observations(path)
    for name, value in _arrays().items():
        np.testing.assert_array_equal(loaded[name], value)
    assert loaded["conventions"] == original["conventions"]
    assert original["provenance"]["status"] == "synthetic"
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[k].dtype.kind != "O" for k in archive.files)
    with pytest.raises(ValueError, match="new"):
        data.save_observations(original, path)


def test_c3d_subframes_preserve_chronological_analog_order(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    path = tmp_path / "invented.c3d"
    path.write_bytes(b"invented backend fixture, not a real C3D")
    raw = np.array([[[1, 3, 5], [10, 30, 50]], [[2, 4, 6], [20, 40, 60]]])
    backend = {
        "parameters": {
            "ANALOG": {
                "LABELS": {"value": ["angle", "torque"]},
                "RATE": {"value": [1000]},
                "UNITS": {"value": ["deg", "Nm"]},
            }
        },
        "data": {"analogs": raw},
    }
    monkeypatch.setitem(sys.modules, "ezc3d", SimpleNamespace(c3d=lambda _: backend))
    result = data.read_c3d_signals(path)
    np.testing.assert_array_equal(result["channels"]["angle"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(
        result["channels"]["torque"], [10, 20, 30, 40, 50, 60]
    )
    np.testing.assert_allclose(result["time"], np.arange(6) / 1000)
    assert result["units"] == {"angle": "deg", "torque": "Nm"}


def test_dynamometer_adapter_requires_explicit_signs_and_emg_normalization(tmp_path):
    time = np.arange(100) / 100
    signals = {
        "time": time,
        "channels": {
            "angle": 10 * time,
            "torque": -20 * time,
            "emg": np.full(100, -0.4),
        },
        "source_sha256": "a" * 64,
    }
    config = {
        "recording_id": "invented-recording",
        "provenance": _metadata()["provenance"],
        "activation_source": "emg_proxy",
        "target_type": "actuator",
        "torque_convention": "invented signal already gravity and inertia corrected",
        "qpos_reference": [0],
        "qvel_reference": [0],
        "qpos_index": 0,
        "dof_index": 0,
        "dof_name": "ankle",
        "split": "train",
        "angle": {"channel": "angle", "gain": np.pi / 180, "offset": 0.1},
        "torque": {"channel": "torque", "gain": -1, "offset": 2},
        "activations": {
            "calf": {
                "channel": "emg",
                "gain": 2,
                "offset": 0,
                "rectify": True,
                "lowpass_hz": 5,
            }
        },
    }
    result = data.dynamometer_observations(signals, config)
    np.testing.assert_allclose(result["qpos"][:, 0], 10 * time * np.pi / 180 + 0.1)
    np.testing.assert_allclose(result["qvel"][:, 0], 10 * np.pi / 180)
    np.testing.assert_allclose(result["joint_torque_Nm"][:, 0], 20 * time + 2)
    np.testing.assert_allclose(result["activation"], 0.8)
    assert result["activation_source"] == "emg_proxy"
    output = data.save_observations(result, tmp_path / "converted.npz")
    assert (
        data.load_observations(output)["conventions"]["source_input_sha256"] == "a" * 64
    )
    bad = {
        **config,
        "activations": {
            "calf": {"channel": "emg", "gain": 10, "offset": 0, "rectify": True}
        },
    }
    with pytest.raises(ValueError, match="normalize explicitly"):
        data.dynamometer_observations(signals, bad)


def test_optional_real_ezc3d_backend_round_trip(tmp_path):
    ezc3d = pytest.importorskip("ezc3d")
    path = tmp_path / "invented.c3d"
    source = ezc3d.c3d()
    source.add_parameter("POINT", "RATE", [100])
    source.add_parameter("POINT", "LABELS", ["invented-marker"])
    source.add_parameter("ANALOG", "RATE", [1000])
    source.add_parameter("ANALOG", "LABELS", ["angle", "torque"])
    source.add_parameter("ANALOG", "UNITS", ["deg", "Nm"])
    source["data"]["points"] = np.ones((4, 1, 10))
    source["data"]["analogs"] = np.stack([np.arange(100), 2 * np.arange(100)])[None]
    source.write(str(path))
    result = data.read_c3d_signals(path)
    assert result["sample_rate_hz"] == 1000
    assert len(result["time"]) == 100
    np.testing.assert_array_equal(result["channels"]["torque"], 2 * np.arange(100))
