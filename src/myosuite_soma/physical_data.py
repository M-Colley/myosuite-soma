# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Local anatomy-table and calibration-observation adapters.

No dataset is downloaded automatically and no anatomical measurement is shipped
in this module. Published table values are priors from other people, rather than
measurements of a SOMA subject. Imports retain source hashes and unit decisions.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile

import numpy as np

from myosuite_soma.soma_numeric import inspect_numeric_archive

_MAX_BYTES = 64 * 1024**2
_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_CATALOG = {
    "charles": {
        "title": "Charles et al. 2019 lower-limb muscle architecture",
        "url": "https://doi.org/10.1371/journal.pone.0223531",
        "license": "CC BY (article and supporting tables)",
        "access": "Public supporting DOCX tables S1-S10; obtain separately",
        "scope": "20 leg muscles in 10 adults; MRI/DTI architecture estimates",
        "adapter": "import_architecture(source='charles', input_path=DOCX)",
        "limitations": [
            "Optimal fibre length uses a published sarcomere-length prior",
            "Maximum force is estimated, not directly measured",
            "PCSA already includes the pennation projection cosine",
        ],
    },
    "hand": {
        "title": "Mirakhorlo et al. 2016 hand/wrist anatomy",
        "url": "https://doi.org/10.6084/m9.figshare.4434791",
        "license": "CC BY 4.0",
        "access": "Public supplementary tables; obtain separately",
        "scope": "48 muscles and 24 joints from one fresh-frozen female arm",
        "adapter": "Main-paper Table 2 CSV architecture; Suppl2 DOCX attachment points",
        "limitations": [
            "Single specimen, not a population distribution or SOMA subject",
            "PCSA is perpendicular; apply pennation projection once",
        ],
    },
    "nordez": {
        "title": "3D-ultrasound muscle-tendon unit lengths",
        "url": "https://doi.org/10.57745/TBCTLO",
        "license": "Etalab Open License 2.0; retain the downloaded notice",
        "access": "Local imaging/landmark files; obtain separately",
        "scope": "Lower-limb MTU paths/lengths over several joint configurations",
        "adapter": "import_mtu_markups: local Slicer JSON control polylines; reviewed native poses",
        "limitations": ["MTU length is not isolated fibre length or tendon slack"],
    },
    "bizet": {
        "title": "Calf ultrasound, EMG and dynamometer mechanics",
        "url": "https://doi.org/10.57745/SFBCL7",
        "license": "Etalab Open License 2.0; retain the downloaded notice",
        "access": "Local recordings; obtain separately",
        "scope": "Calf fascicles, EMG, torque/angle/velocity and passive/active trials",
        "adapter": "Review channel conventions, then load_observations(CSV or NPZ)",
        "limitations": [
            "EMG is a proxy, not measured muscle activation or force",
            "Dynamometer torque requires explicit sign and gravity conventions",
        ],
    },
    "addbiomechanics": {
        "title": "AddBiomechanics 1.0 core dataset",
        "url": "https://www.addbiomechanics.org/download_data.html",
        "license": "CC BY 4.0; retain original-source attribution",
        "access": "Public B3D files; obtain separately",
        "scope": "273 participants; measured ground forces and processed dynamics",
        "adapter": "Local processed OpenSim segment properties with explicit body-frame registration; reviewed native dynamics CSV/NPZ",
        "limitations": [
            "Joint torques, masses and inertias are processed estimates",
            "Individual muscle activation and force are not measured",
            "Net inverse-dynamics torque is not automatically actuator torque",
        ],
    },
}


def dataset_catalog() -> dict[str, Any]:
    """Copy the curated source inventory without contacting external services."""
    return copy.deepcopy(_CATALOG)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]", "", value)


def _number(value: str) -> float | None:
    """Read a mean, retaining uncertainty separately instead of treating SD as mean."""
    value = value.strip().replace("\u2212", "-")
    if _text(value) in ("", "na", "nan", "notavailable", "none", "x") or value in (
        "-",
        "—",
    ):
        return None
    if "," in value:
        raise ValueError(
            "Use explicit decimal points and no thousands separators in numeric cells"
        )
    match = re.match(r"^[<>]?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", value)
    if match is None:
        raise ValueError(f"Unrecognized numeric table value: {value!r}")
    tail = value[match.end() :].strip()
    if tail and not tail.startswith(("±", "+/-", "*")):
        raise ValueError(f"Ambiguous extra text in numeric table value: {value!r}")
    result = float(match.group(1))
    if not np.isfinite(result):
        raise ValueError("Architecture values must be finite")
    return result


def _docx_tables(path: Path) -> list[list[list[str]]]:
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("DOCX exceeds the 64 MiB input limit")
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if (
                len(entries) > 10000
                or len({e.filename for e in entries}) != len(entries)
                or sum(e.file_size for e in entries) > _MAX_BYTES
            ):
                raise ValueError("Oversized or duplicated DOCX members")
            info = archive.getinfo("word/document.xml")
            if info.file_size > 8 * 1024**2:
                raise ValueError("DOCX document XML exceeds 8 MiB")
            xml = archive.read(info)
    except (BadZipFile, KeyError) as exc:
        raise ValueError("Expected a DOCX containing word/document.xml") from exc
    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise ValueError("DOCX document declarations/entities are not permitted")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError("Invalid DOCX document XML") from exc
    return [
        [
            [
                # Word runs are formatting fragments, not whitespace-separated
                # tokens: inserting spaces turns e.g. 5 + 67 into '5 67'.
                "\n".join(
                    "".join(t.text or "" for t in paragraph.findall(".//w:t", _NS))
                    for paragraph in cell.findall("w:p", _NS)
                ).strip()
                for cell in row.findall("w:tc", _NS)
            ]
            for row in table.findall("w:tr", _NS)
        ]
        for table in root.findall(".//w:tbl", _NS)
    ]


def _column(header: str) -> tuple[str, float] | None:
    key = _text(header)
    if key in ("muscle", "musclename", "muscleabbreviation", "name"):
        return "muscle_name", 1.0
    if "pcsa" in key or "physiologicalcrosssectionalarea" in key:
        factor = (
            1e-4
            if "cm2" in key
            else 1e-6
            if "mm2" in key
            else 1.0
            if "m2" in key
            else None
        )
        field = "pcsa_m2"
    elif "volume" in key:
        factor = (
            1e-6
            if "cm3" in key or "ml" in key
            else 1e-9
            if "mm3" in key
            else 1.0
            if "m3" in key
            else None
        )
        field = "volume_m3"
    elif "pennation" in key or key.startswith("angle"):
        factor = (
            180 / np.pi
            if "rad" in key
            else 1.0
            if "deg" in key or "°" in header
            else None
        )
        field = "pennation_deg"
    elif (
        "fiberlength" in key or "fibrelength" in key or key.startswith("lopt")
    ) and "ratio" not in key:
        factor = (
            1e-3
            if "mm" in key
            else 1e-2
            if "cm" in key
            else 1.0
            if "m" in key
            else None
        )
        field = "fibre_length_m"
    elif ("fmax" in key or "maximumisometricforce" in key) and "bw" not in key:
        factor = 1.0 if "n" in key else None
        field = "force_N"
    else:
        return None
    if factor is None:
        return None
    return field, factor


def import_architecture(
    source: str, input_path: str | Path, *, subject: str | None = None
) -> dict[str, Any]:
    """Normalize published local DOCX or explicitly unit-labelled CSV tables.

    Headers must identify units. Charles' native units are cm³/mm/mm²/N; their
    projected PCSA already incorporates cos(pennation). Means are imported and
    complete source cell strings retained so SD/uncertainty is not discarded.
    No anatomical source name is silently bound to a simulator actuator. The
    hand supplement's Suppl2.docx contains attachment geometry, not the main
    paper's architecture table: use import_hand_attachments for that file.
    """
    if source not in ("charles", "hand"):
        raise ValueError("Architecture source must be 'charles' or 'hand'")
    path = Path(input_path).resolve()
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("Architecture input exceeds 64 MiB")
    if path.suffix.lower() == ".docx":
        tables = _docx_tables(path)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            tables = [list(csv.reader(stream))]
    else:
        raise ValueError("Architecture input must be .docx or a unit-labelled .csv")
    records, ignored = [], []
    groups = {
        "hipadductors",
        "kneeflexors",
        "kneeextensors",
        "ankledorsiflexors",
        "ankleplantarflexors",
        "mean",
        "total",
        "wrist",
        "thumb",
        "indexfinger",
        "middlefinger",
        "ringfinger",
        "littlefinger",
    }
    for table_index, table in enumerate(tables):
        columns = None
        for row_index, cells in enumerate(table):
            candidates = {
                i: parsed
                for i, c in enumerate(cells)
                if (parsed := _column(c)) is not None
            }
            if any(field == "muscle_name" for field, _ in candidates.values()) and any(
                field in ("fibre_length_m", "pcsa_m2", "volume_m3")
                for field, _ in candidates.values()
            ):
                columns = candidates
                continue
            if columns is None:
                continue
            if max(columns) >= len(cells):
                ignored.append(
                    {
                        "table": table_index,
                        "row": row_index,
                        "reason": "short or merged row",
                    }
                )
                continue
            name_column = next(
                i for i, (field, _) in columns.items() if field == "muscle_name"
            )
            name = cells[name_column].strip()
            if not name or _text(name) in groups:
                continue
            record: dict[str, Any] = {
                "muscle_name": name,
                "subject": subject or "unspecified",
                "source": source,
                "pcsa_kind": "projected" if source == "charles" else "perpendicular",
                "force_kind": "estimated" if source == "charles" else "not_provided",
                "raw_cells": {field: cells[i] for i, (field, _) in columns.items()},
                "table_index": table_index,
                "row_index": row_index,
            }
            for i, (field, factor) in columns.items():
                if field == "muscle_name":
                    continue
                value = _number(cells[i])
                if value is not None:
                    converted = value * factor
                    if converted < 0 or (field != "pennation_deg" and converted == 0):
                        raise ValueError(
                            f"Nonpositive architecture value for {name}.{field}"
                        )
                    if field == "pennation_deg" and converted >= 90:
                        raise ValueError("Pennation must lie in [0, 90) degrees")
                    record[field] = converted
            if not any(
                field in record for field in ("volume_m3", "fibre_length_m", "pcsa_m2")
            ):
                ignored.append(
                    {
                        "table": table_index,
                        "row": row_index,
                        "reason": "No usable measurements (e.g. absent muscle)",
                    }
                )
                continue
            if "force_N" in record and source == "hand":
                record["force_kind"] = "source_reported_unverified"
            records.append(record)
    if not records:
        raise ValueError(
            "No architecture table found; use explicit muscle/units headers"
        )
    names = [r["muscle_name"] for r in records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate muscle rows: import one subject/table at a time")
    return {
        "version": 1,
        "source": source,
        "source_url": _CATALOG[source]["url"],
        "source_sha256": _hash(path),
        "source_filename": path.name,
        "license": _CATALOG[source]["license"],
        "provenance": {"status": "estimated", "source": _CATALOG[source]["url"]},
        "interpretation": "Population/specimen architecture prior, not a target-person measurement",
        "rows": records,
        "ignored_rows": ignored,
    }


def import_hand_attachments(input_path: str | Path) -> dict[str, Any]:
    """Read Mirakhorlo Suppl2.docx Table S7 without guessing native bindings.

    Coordinates are converted from published specimen-rest millimetres to
    metres. Insertion/origin/via labels and source segment names are retained;
    they are not silently treated as MuJoCo local coordinates or named sites.
    """
    path = Path(input_path).resolve()
    points = []
    for table_index, table in enumerate(_docx_tables(path)):
        if not table or [_text(c) for c in table[0]][:4] != [
            "muscle",
            "abb",
            "po",
            "seg",
        ]:
            continue
        if "mm" not in _text(" ".join(table[0])):
            raise ValueError("Hand attachment table must declare millimetres")
        muscle = abbreviation = None
        for row_index, row in enumerate(table[1:], 1):
            if len(row) != 8 or not row[2].strip() or not row[3].strip():
                continue
            if row[0].strip():
                muscle, abbreviation = row[0].strip(), row[1].strip()
            if muscle is None or abbreviation is None:
                raise ValueError("Attachment point has no source muscle identity")
            values = [_number(c) for c in row[4:7]]
            if any(v is None for v in values):
                if all(v is None for v in values):
                    continue
                raise ValueError(
                    "Hand attachment coordinates must be all numeric or explicitly absent"
                )
            points.append(
                {
                    "id": f"table{table_index}:row{row_index}",
                    "muscle_name": muscle,
                    "abbreviation": abbreviation,
                    "point_kind": row[2].strip(),
                    "source_segment": row[3].strip(),
                    "position_m": [float(v) * 0.001 for v in values],
                    "note": row[7].strip(),
                }
            )
    if not points:
        raise ValueError("No hand Table S7 attachment point table found")
    return {
        "version": 1,
        "provenance": {"status": "estimated", "source": _CATALOG["hand"]["url"]},
        "source_sha256": _hash(path),
        "source_filename": path.name,
        "coordinate_frame": "published_specimen_rest_frame",
        "units": "m",
        "points": points,
        "binding": "Explicit specimen-to-native registration and point-to-site map required",
    }


def attachment_profile(
    data: Mapping[str, Any],
    model: Any,
    mapping: Mapping[str, str],
    *,
    rotation: Any,
    translation_m: Any,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Transform reviewed specimen points into existing native site's body frames.

    mapping maps exact imported point IDs to exact existing site names.
    Rotation/translation explicitly register the published specimen rest frame
    to the target model's world frame at qpos0. No attachment is auto-approved.
    """
    import mujoco

    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation_m, dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)
        or translation.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            "Attachment registration needs a positive scale/proper rotation/translation"
        )
    if (
        data.get("units") != "m"
        or data.get("coordinate_frame") != "published_specimen_rest_frame"
    ):
        raise ValueError(
            "Attachment data coordinate convention does not match the importer"
        )
    source = {point["id"]: point for point in data["points"]}
    if (
        not mapping
        or set(mapping) - source.keys()
        or len(set(mapping.values())) != len(mapping)
    ):
        raise ValueError(
            "Provide a unique explicit imported-point to existing-site map"
        )
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = model.qpos0
    mujoco.mj_forward(model, scratch)
    sites = {}
    for point, name in mapping.items():
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            raise ValueError(f"Unknown native attachment site: {name}")
        body = int(model.site_bodyid[site])
        world = scale * rotation @ np.asarray(source[point]["position_m"]) + translation
        local = scratch.xmat[body].reshape(3, 3).T @ (world - scratch.xpos[body])
        sites[name] = {"pos_m": local.tolist()}
    return {
        "version": 1,
        "provenance": dict(data["provenance"]),
        "sites": sites,
        "recompute_lengthrange": True,
    }


def architecture_profile(
    data: Mapping[str, Any], mapping: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind source rows to explicit native actuators and form a physical profile.

    Each mapping value is an object containing ``actuator`` and either
    ``force_source='reported'`` or ``specific_tension_pa``. Missing pennation or
    unknown PCSA convention must be supplied explicitly. One-to-many anatomical
    allocation requires explicit ``fraction`` values summing to at most one.
    """
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("An explicit nonempty source-muscle mapping is required")
    rows = {row["muscle_name"]: row for row in data["rows"]}
    if set(mapping) - rows.keys():
        raise ValueError(
            f"Unknown source muscles: {sorted(set(mapping) - rows.keys())}"
        )
    muscles = {}
    for name, specs in mapping.items():
        record = rows[name]
        specs = specs if isinstance(specs, list) else [specs]
        if not specs or any(not isinstance(spec, Mapping) for spec in specs):
            raise ValueError("Each mapping needs an actuator/specific-tension object")
        if any(isinstance(spec.get("fraction", 1.0), (bool, str)) for spec in specs):
            raise ValueError("Allocation fractions must be real numbers")
        fractions = [float(spec.get("fraction", 1.0)) for spec in specs]
        if (
            not all(np.isfinite(f) and 0 < f <= 1 for f in fractions)
            or sum(fractions) > 1 + 1e-10
        ):
            raise ValueError(
                "Source-muscle allocation fractions must sum to at most one"
            )
        for spec, fraction in zip(specs, fractions):
            unknown = set(spec) - {
                "actuator",
                "fraction",
                "force_source",
                "specific_tension_pa",
                "pennation_deg",
                "pcsa_kind",
            }
            if unknown:
                raise ValueError(
                    f"Unknown architecture mapping keys: {sorted(unknown)}"
                )
            actuator = spec.get("actuator")
            if not isinstance(actuator, str) or not actuator or actuator in muscles:
                raise ValueError(
                    "Each native actuator must have one explicit unique source"
                )
            provenance = {
                "status": "estimated",
                "source": f"{data['source_url']} ({record['subject']}, {name}); sha256={data['source_sha256']}",
            }
            entry: dict[str, Any] = {"provenance": provenance}
            if spec.get("force_source") == "reported":
                if "force_N" not in record:
                    raise ValueError(f"No reported force for source muscle {name}")
                entry["force_N"] = float(record["force_N"]) * fraction
            else:
                if spec.get("force_source") not in (None, "architecture"):
                    raise ValueError(
                        "force_source must be 'reported' or 'architecture'"
                    )
                sigma = spec.get("specific_tension_pa")
                if (
                    sigma is None
                    or isinstance(sigma, (bool, str))
                    or not np.isfinite(float(sigma))
                    or float(sigma) <= 0
                ):
                    raise ValueError(
                        "Force estimation requires explicit positive specific_tension_pa"
                    )
                angle = spec.get("pennation_deg", record.get("pennation_deg"))
                if (
                    "volume_m3" in record
                    and "fibre_length_m" in record
                    and angle is not None
                ):
                    architecture = {
                        "volume_m3": record["volume_m3"] * fraction,
                        "fibre_length_m": record["fibre_length_m"],
                        "pennation_deg": float(angle),
                    }
                elif "pcsa_m2" in record:
                    kind = spec.get("pcsa_kind", record.get("pcsa_kind"))
                    if kind not in ("projected", "perpendicular"):
                        raise ValueError(
                            "PCSA needs an explicit projected/perpendicular convention"
                        )
                    architecture = {
                        "pcsa_m2": record["pcsa_m2"] * fraction,
                        "pcsa_kind": kind,
                    }
                    if kind == "perpendicular":
                        if angle is None:
                            raise ValueError(
                                "Perpendicular PCSA requires explicit pennation_deg"
                            )
                        architecture["pennation_deg"] = float(angle)
                else:
                    raise ValueError(f"Insufficient architecture values for {name}")
                architecture["specific_tension_pa"] = float(sigma)
                entry["architecture"] = architecture
            muscles[actuator] = entry
    return {"version": 1, "provenance": dict(data["provenance"]), "muscles": muscles}


def save_json(data: Mapping[str, Any], output: str | Path) -> Path:
    """Write a new local JSON artifact without overwriting an existing file."""
    target = Path(output).resolve()
    if target.suffix.lower() != ".json" or target.exists():
        raise ValueError("Choose a new .json output file")
    encoded = json.dumps(data, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    return target


def _observations_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("Observation metadata must be a JSON object")
    result = dict(metadata)
    provenance = result.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("status") not in ("measured", "estimated", "synthetic")
        or not isinstance(provenance.get("source"), str)
        or not provenance["source"].strip()
    ):
        raise ValueError("Observation provenance needs status and a nonempty source")
    if result.get("activation_source") not in (
        "measured_activation",
        "emg_proxy",
        "synthetic",
    ):
        raise ValueError(
            "Declare measured_activation, emg_proxy or synthetic activation_source"
        )
    units = result.get("units")
    expected = {
        "time": "s",
        "qpos": "SI",
        "qvel": "SI/s",
        "joint_torque_Nm": "Nm",
        "activation": "unitless",
    }
    if units != expected:
        raise ValueError(f"Observation units must explicitly equal {expected}")
    return result


def load_observations(
    input_path: str | Path, *, metadata: Mapping[str, Any] | str | Path | None = None
) -> dict[str, Any]:
    """Load the portable numeric calibration schema from local NPZ or wide CSV.

    CSV columns: time, recording_id, optional split, qpos:0..N, qvel:0..M,
    activation:EXACT_ACTUATOR and torque:EXACT_HINGE_JOINT. A JSON sidecar is
    required, declaring provenance, activation_source, units and name order.
    NPZ stores arrays plus a scalar Unicode ``metadata_json``; object/pickle
    arrays are prohibited and headers are checked before allocation.
    No sign, EMG normalization or dynamometer-gravity convention is inferred.
    """
    path = Path(input_path).resolve()
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("Observations exceed the 64 MiB input limit; split recordings")
    if isinstance(metadata, (str, Path)):
        metadata = json.loads(Path(metadata).read_text(encoding="utf-8"))
    if path.suffix.lower() == ".npz":
        inspect_numeric_archive(path, max_bytes=_MAX_BYTES, allow_text=True)
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
        embedded = arrays.pop("metadata_json", None)
        if embedded is None and metadata is None:
            raise ValueError("NPZ needs metadata_json or an explicit metadata sidecar")
        if embedded is not None:
            if embedded.shape != () or embedded.dtype.kind not in "US":
                raise ValueError("metadata_json must be a scalar text array")
            item = embedded.item()
            embedded_metadata = json.loads(
                item.decode("utf-8") if isinstance(item, bytes) else item
            )
            if metadata is not None and dict(metadata) != embedded_metadata:
                raise ValueError(
                    "Conflicting embedded and sidecar observation metadata"
                )
            metadata = embedded_metadata
        result = _observations_metadata(metadata)
        forbidden = set(result) & set(arrays)
        if forbidden:
            raise ValueError(
                f"Metadata cannot override numeric arrays: {sorted(forbidden)}"
            )
        result.update(arrays)
    elif path.suffix.lower() == ".csv":
        result = _observations_metadata(metadata)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            if len(set(headers)) != len(headers):
                raise ValueError("Duplicate CSV headers")
            rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError("CSV rows contain more cells than headers")
        if not rows or "time" not in headers or "recording_id" not in headers:
            raise ValueError("CSV needs time, recording_id and at least one sample")
        names = result.get("actuator_names"), result.get("dof_names")
        if any(
            not isinstance(ns, list)
            or not ns
            or len(set(ns)) != len(ns)
            or not all(isinstance(n, str) and n for n in ns)
            for ns in names
        ):
            raise ValueError(
                "CSV metadata needs unique ordered actuator_names and dof_names"
            )

        def numbered(prefix: str) -> list[str]:
            selected = [h for h in headers if h.startswith(prefix)]
            wanted = [f"{prefix}{i}" for i in range(len(selected))]
            if not wanted or set(selected) != set(wanted):
                raise ValueError(
                    f"CSV {prefix} columns must be contiguous indices from zero"
                )
            return wanted

        columns = {
            "qpos": numbered("qpos:"),
            "qvel": numbered("qvel:"),
            "activation": [f"activation:{n}" for n in names[0]],
            "joint_torque_Nm": [f"torque:{n}" for n in names[1]],
        }
        required = {
            "time",
            "recording_id",
            *(h for group in columns.values() for h in group),
        }
        if set(headers) - required - {"split"} or required - set(headers):
            raise ValueError("CSV columns do not match the declared calibration schema")
        try:
            result["time"] = np.array([float(row["time"]) for row in rows])
            for key, group in columns.items():
                result[key] = np.array([[float(row[h]) for h in group] for row in rows])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CSV numeric cells must contain finite real numbers"
            ) from exc
        result["recording_ids"] = np.array([row["recording_id"] for row in rows])
        if "split" in headers:
            result["split"] = np.array([row["split"] for row in rows])
    else:
        raise ValueError("Observation input must be .csv or numeric .npz")
    _validate_observations(result)
    result["input_sha256"] = _hash(path)
    return result


def _validate_observations(result: dict[str, Any]) -> None:
    required_arrays = {
        "time",
        "qpos",
        "qvel",
        "activation",
        "joint_torque_Nm",
        "recording_ids",
        "actuator_names",
        "dof_names",
    }
    if required_arrays - result.keys():
        raise ValueError(
            f"Missing observation fields: {sorted(required_arrays - result.keys())}"
        )
    times = np.asarray(result["time"])
    if times.ndim != 1 or not len(times):
        raise ValueError("Observation time must be a nonempty vector")
    count = len(times)
    for key in ("time", "qpos", "qvel", "activation", "joint_torque_Nm"):
        value = np.asarray(result[key])
        if (
            value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
            or value.ndim != (1 if key == "time" else 2)
            or value.shape[0] != count
        ):
            raise ValueError(f"Invalid numeric observation array {key}")
        result[key] = value.astype(float)
    for key in ("recording_ids", "actuator_names", "dof_names"):
        value = np.asarray(result[key])
        if (
            value.ndim != 1
            or value.dtype.kind not in "US"
            or any(not str(n) for n in value)
        ):
            raise ValueError(f"{key} must be a vector of nonempty names")
        result[key] = value.astype(str)
    if len(result["recording_ids"]) != count:
        raise ValueError("recording_ids must have one entry per sample")
    for key, matrix in (
        ("actuator_names", "activation"),
        ("dof_names", "joint_torque_Nm"),
    ):
        if (
            len(set(result[key])) != len(result[key])
            or len(result[key]) != result[matrix].shape[1]
        ):
            raise ValueError(f"{key} do not match matrix columns")
    if np.any(result["activation"] < 0) or np.any(result["activation"] > 1):
        raise ValueError(
            "Activation/proxy values must lie in [0,1]; normalize explicitly"
        )
    for group in set(result["recording_ids"]):
        times = result["time"][result["recording_ids"] == group]
        if np.any(np.diff(times) <= 0):
            raise ValueError("Times must increase within each recording")
    if "split" in result:
        split = np.asarray(result["split"]).astype(str)
        if split.shape != (count,) or not set(split).issubset({"train", "validation"}):
            raise ValueError("split must contain train/validation for every sample")
        if any(
            len(set(split[result["recording_ids"] == group])) != 1
            for group in set(result["recording_ids"])
        ):
            raise ValueError("A recording cannot leak across train and validation")
        result["split"] = split


def save_observations(observations: Mapping[str, Any], output: str | Path) -> Path:
    """Save a new portable numeric NPZ with JSON metadata and no pickle arrays."""
    target = Path(output).resolve()
    if target.suffix.lower() != ".npz" or target.exists():
        raise ValueError("Choose a new .npz output file")
    values = dict(observations)
    _observations_metadata(values)
    _validate_observations(values)
    keys = {
        "time",
        "qpos",
        "qvel",
        "activation",
        "joint_torque_Nm",
        "recording_ids",
        "actuator_names",
        "dof_names",
        "split",
    }
    arrays = {key: np.asarray(value) for key, value in values.items() if key in keys}
    metadata = {key: value for key, value in values.items() if key not in keys}
    if "input_sha256" in values:
        metadata["conventions"] = dict(metadata.get("conventions", {}))
        metadata["conventions"].setdefault(
            "source_input_sha256", values["input_sha256"]
        )
    arrays["metadata_json"] = np.array(json.dumps(metadata, allow_nan=False))
    if sum(value.nbytes for value in arrays.values()) > _MAX_BYTES:
        raise ValueError("Observations exceed the 64 MiB output limit")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return target


def read_c3d_signals(input_path: str | Path) -> dict[str, Any]:
    """Read local raw C3D analog channels using optional ezc3d, without guessing units.

    The Bizet dataset distributes dynamometer and EMG recordings as C3D. This
    helper only extracts analog channels; it does not infer torque calibration,
    anatomical angle zero/sign or EMG normalization from a channel label.
    """
    path = Path(input_path).resolve()
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("C3D exceeds 64 MiB; choose a smaller recording")
    try:
        import ezc3d
    except ImportError as exc:
        raise ImportError(
            "C3D import needs the optional ezc3d package; pip install ezc3d"
        ) from exc
    source = ezc3d.c3d(str(path))
    group = source["parameters"]["ANALOG"]
    labels = [str(v).strip() for v in group["LABELS"]["value"]]
    rate = float(group["RATE"]["value"][0])
    raw = np.asarray(source["data"]["analogs"])
    if (
        raw.ndim != 3
        or raw.shape[1] != len(labels)
        or len(set(labels)) != len(labels)
        or any(not n for n in labels)
        or not np.isfinite(raw).all()
        or not np.isfinite(rate)
        or rate <= 0
    ):
        raise ValueError("Invalid C3D analog labels, rate or sample layout")
    # ezc3d uses (subframes, channels, frames). Preserve chronological samples.
    values = raw.transpose(2, 0, 1).reshape(-1, len(labels)).astype(float)
    if not len(values) or values.nbytes > _MAX_BYTES:
        raise ValueError("C3D analog payload is empty or exceeds 64 MiB")
    units = [str(v).strip() for v in group.get("UNITS", {}).get("value", [])]
    return {
        "time": np.arange(len(values), dtype=float) / rate,
        "channels": {name: values[:, i] for i, name in enumerate(labels)},
        "units": dict(zip(labels, units)),
        "sample_rate_hz": rate,
        "source_sha256": _hash(path),
        "source_filename": path.name,
    }


def dynamometer_observations(
    signals: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert reviewed single-axis dynamometer channels to native observations.

    Explicit angle/torque gains and offsets define radians and signed Nm.
    qpos/qvel reference vectors and their ankle indices define the native model
    convention. EMG channels require user-specified normalization; filtering,
    rectification and clipping are opt-in and recorded as proxy preprocessing.
    """
    from scipy.signal import butter, sosfiltfilt

    time = np.asarray(signals["time"], dtype=float)
    if (
        time.ndim != 1
        or len(time) < 3
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Dynamometer time requires at least three increasing samples")
    intervals = np.diff(time)
    if not np.allclose(intervals, intervals[0], rtol=1e-5, atol=1e-9):
        raise ValueError("Dynamometer preprocessing needs uniform sample times")
    rate = 1 / float(intervals[0])
    if not isinstance(config.get("recording_id"), str) or not config["recording_id"]:
        raise ValueError("Declare a nonempty dynamometer recording_id")
    if (
        config.get("target_type") not in ("actuator", "net")
        or not isinstance(config.get("torque_convention"), str)
        or not config["torque_convention"]
    ):
        raise ValueError(
            "Declare target_type and corrected torque_convention explicitly"
        )
    channels = signals["channels"]

    def convert(spec: Mapping[str, Any], label: str) -> np.ndarray:
        if (
            not isinstance(spec, Mapping)
            or spec.get("channel") not in channels
            or "gain" not in spec
            or "offset" not in spec
        ):
            raise ValueError(f"{label} requires exact channel, gain and offset")
        value = np.asarray(channels[spec["channel"]], dtype=float).copy()
        if value.shape != time.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid dynamometer channel {label}")
        gain, offset = float(spec["gain"]), float(spec["offset"])
        if not np.isfinite(gain) or not np.isfinite(offset) or gain == 0:
            raise ValueError("Dynamometer gains must be finite and nonzero")
        for key, mode in (("highpass_hz", "highpass"), ("lowpass_hz", "lowpass")):
            if key == "lowpass_hz" and spec.get("rectify", False):
                value = np.abs(value)
            if key in spec:
                cutoff = float(spec[key])
                if not np.isfinite(cutoff) or not 0 < cutoff < rate / 2:
                    raise ValueError("Filter cutoff must lie below Nyquist")
                try:
                    value = sosfiltfilt(
                        butter(4, cutoff, btype=mode, fs=rate, output="sos"), value
                    )
                except ValueError as exc:
                    raise ValueError(
                        "Dynamometer recording is too short for the chosen filter"
                    ) from exc
        return value * gain + offset

    qpos0 = np.asarray(config.get("qpos_reference"), dtype=float)
    qvel0 = np.asarray(config.get("qvel_reference"), dtype=float)
    index = config.get("qpos_index")
    dof = config.get("dof_index")
    if (
        qpos0.ndim != 1
        or qvel0.ndim != 1
        or not np.isfinite(qpos0).all()
        or not np.isfinite(qvel0).all()
        or not isinstance(index, int)
        or not isinstance(dof, int)
        or not 0 <= index < len(qpos0)
        or not 0 <= dof < len(qvel0)
    ):
        raise ValueError("Declare native reference vectors and valid angle/dof indices")
    angle = convert(config["angle"], "angle")
    velocity = (
        convert(config["velocity"], "velocity")
        if "velocity" in config
        else np.gradient(angle, time)
    )
    torque = convert(config["torque"], "torque")
    qpos = np.tile(qpos0, (len(time), 1))
    qvel = np.tile(qvel0, (len(time), 1))
    qpos[:, index], qvel[:, dof] = angle, velocity
    activations = config.get("activations")
    if not isinstance(activations, Mapping) or not activations:
        raise ValueError(
            "Provide an explicit normalized channel for every native muscle"
        )
    values = []
    for name, spec in activations.items():
        if "constant" in spec:
            value = np.full(len(time), float(spec["constant"]))
        else:
            value = convert(spec, f"activation:{name}")
        if spec.get("clip", False):
            value = np.clip(value, 0, 1)
        values.append(value)
    result = {
        "time": time.copy(),
        "qpos": qpos,
        "qvel": qvel,
        "activation": np.stack(values, 1),
        "joint_torque_Nm": torque[:, None],
        "actuator_names": np.array(list(activations)),
        "dof_names": np.array([config["dof_name"]]),
        "recording_ids": np.full(len(time), config["recording_id"]),
        "provenance": dict(config["provenance"]),
        "activation_source": config.get("activation_source", "emg_proxy"),
        "target_type": config["target_type"],
        "units": {
            "time": "s",
            "qpos": "SI",
            "qvel": "SI/s",
            "joint_torque_Nm": "Nm",
            "activation": "unitless",
        },
        "conventions": {
            "torque_convention": config["torque_convention"],
            "dynamometer_conversion": dict(config),
            "source_input_sha256": signals.get("source_sha256"),
            "source_units": signals.get("units", {}),
            "velocity": "explicit channel"
            if "velocity" in config
            else "numerical angle derivative",
        },
    }
    if "split" in config:
        result["split"] = np.full(len(time), config["split"])
    _observations_metadata(result)
    _validate_observations(result)
    return result
