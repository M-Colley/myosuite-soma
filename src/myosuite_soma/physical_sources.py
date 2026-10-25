# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Local OpenSim segment and 3D Slicer path adapters.

Only numerical properties are read. No model, scan, geometry or measurement is
downloaded or included. Registration and native-model pose review remain explicit
user decisions; an external participant is not thereby matched to a SOMA subject.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

_MAX_BYTES = 64 * 1024**2
_OPENSIM_SPEC = (
    "https://github.com/opensim-org/opensim-core/blob/main/"
    "OpenSim/Simulation/SimbodyEngine/Body.h"
)


def _read(path: Path) -> bytes:
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("Input must be an existing local file smaller than 64 MiB")
    return path.read_bytes()


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"{label} must contain finite numbers with shape {shape}")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _inertia(value: Any, label: str) -> np.ndarray:
    tensor = _array(value, (3, 3), label)
    magnitude = float(np.max(np.abs(tensor)))
    if magnitude == 0 or np.max(np.abs(tensor - tensor.T)) > 1e-10 * magnitude:
        raise ValueError(f"{label} must be a nonzero symmetric inertia tensor")
    tensor = (tensor + tensor.T) * 0.5
    principal = np.linalg.eigvalsh(tensor)
    if np.any(principal <= 0) or 2 > (principal / principal[-1]).sum() + 1e-10:
        raise ValueError(f"{label} needs positive physical principal moments")
    return tensor


def _children(element: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in element if child.tag.rsplit("}", 1)[-1] == tag]


def _one(element: ET.Element, tag: str) -> ET.Element:
    items = _children(element, tag)
    if len(items) != 1:
        raise ValueError(f"Expected exactly one {tag} in {element.tag}")
    return items[0]


def _xml_numbers(element: ET.Element, count: int) -> np.ndarray:
    parts = (element.text or "").split()
    if len(parts) != count:
        raise ValueError(f"{element.tag} needs {count} explicit SI numbers")
    try:
        result = np.array([float(v) for v in parts])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric {element.tag}") from exc
    return _array(result, (count,), element.tag)


def import_segment_properties(path: str | Path) -> dict[str, Any]:
    """Read explicit SI body properties from a local processed ``.osim`` file.

    OpenSim 4's ``inertia`` is [xx, yy, zz, xy, xz, yz]. OpenSim 3's six
    ``inertia_xx`` ... ``inertia_yz`` properties have the same meaning. Products
    are already the tensor's off-diagonals: no additional sign flip is applied.
    All inertias are about COM and expressed in the source body frame. Invalid
    inertias are rejected rather than replaced by OpenSim's spherical fallback.
    """
    path = Path(path)
    raw = _read(path)
    # Also inspect UTF-16 XML declarations before parsing. No entities/includes
    # are needed for numerical body properties.
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)", raw.replace(b"\x00", b""), re.I):
        raise ValueError("XML DOCTYPE and ENTITY declarations are not supported")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("Invalid OpenSim XML") from exc
    if root.tag.rsplit("}", 1)[-1] != "OpenSimDocument":
        raise ValueError("Expected an OpenSimDocument XML file")
    model = _one(root, "Model")
    declared_length_units = _children(model, "length_units")
    if declared_length_units and (
        len(declared_length_units) != 1
        or (declared_length_units[0].text or "").strip() not in ("m", "meters")
    ):
        raise ValueError("Source OpenSim length_units must use SI meters")
    objects = _one(_one(model, "BodySet"), "objects")
    segments: dict[str, Any] = {}
    massless: list[str] = []
    names: set[str] = set()
    for body in _children(objects, "Body"):
        name = body.get("name", "")
        if not name or name in names:
            raise ValueError("Source Body names must be nonempty and unique")
        names.add(name)
        mass = float(_xml_numbers(_one(body, "mass"), 1)[0])
        if mass < 0:
            raise ValueError(f"{name}: mass must be nonnegative")
        if mass == 0:
            massless.append(name)
            continue
        com = _xml_numbers(_one(body, "mass_center"), 3)
        modern = _children(body, "inertia")
        legacy = [f"inertia_{axis}" for axis in ("xx", "yy", "zz", "xy", "xz", "yz")]
        if modern and any(_children(body, field) for field in legacy):
            raise ValueError(f"{name}: ambiguous modern and legacy inertia fields")
        if modern:
            vector = _xml_numbers(_one(body, "inertia"), 6)
        else:
            vector = np.array([_xml_numbers(_one(body, k), 1)[0] for k in legacy])
        xx, yy, zz, xy, xz, yz = vector
        tensor = _inertia([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], name)
        segments[name] = {
            "mass_kg": mass,
            "com_m": com.tolist(),
            "inertia_kg_m2": tensor.tolist(),
        }
    if not segments:
        raise ValueError("No positive-mass source bodies with explicit properties")
    return {
        "version": 1,
        "provenance": {
            "status": "estimated",
            "source": f"Processed local OpenSim model {path.name}",
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "specification": _OPENSIM_SPEC,
            "note": "Processed estimates; source participant is not a matched SOMA donor",
        },
        "units": {"mass": "kg", "com": "m", "inertia": "kg*m^2"},
        "frame": "Source body axes, inertia about source COM",
        "massless_bodies_ignored": massless,
        "segments": segments,
    }


def segment_profile(
    data: Mapping[str, Any], model: mujoco.MjModel, mapping: Mapping[str, Any]
) -> dict[str, Any]:
    """Map reviewed one-to-one body correspondences into a builder profile.

    Every mapping entry must contain ``body``, ``rotation`` and ``translation_m``
    with ``x_native_body = rotation @ x_source_body + translation_m``. This is a
    rigid frame registration, with no anatomical scaling or joint retargeting.
    COM inertia becomes R I R.T; translation does not add a parallel-axis term.
    """
    from myosuite_soma.physical_model import physical_digest

    if not isinstance(model, mujoco.MjModel):
        raise ValueError("model must be the explicitly selected native MjModel")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("A nonempty source-body registration mapping is required")
    if not isinstance(data, Mapping) or data.get("units") != {
        "mass": "kg",
        "com": "m",
        "inertia": "kg*m^2",
    }:
        raise ValueError("Imported SI segment properties and units are required")
    segments = data.get("segments", {})
    provenance = data.get("provenance", {})
    if (
        not isinstance(segments, Mapping)
        or not isinstance(provenance, Mapping)
        or not isinstance(provenance.get("input_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", provenance["input_sha256"])
        or provenance.get("status") != "estimated"
        or not isinstance(provenance.get("source"), str)
        or not provenance["source"].strip()
    ):
        raise ValueError("Imported processed segment provenance is required")
    edits: dict[str, Any] = {}
    for source, registration in mapping.items():
        if source not in segments:
            raise ValueError(f"Unknown positive-mass source body: {source}")
        if not isinstance(registration, Mapping) or set(registration) != {
            "body",
            "rotation",
            "translation_m",
        }:
            raise ValueError("Each body mapping requires body, rotation, translation_m")
        name = registration["body"]
        if not isinstance(name, str) or not name:
            raise ValueError("Native body names must be nonempty strings")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id <= 0 or name in edits:
            raise ValueError(f"Native body must exist, be unique and not world: {name}")
        rotation = _array(registration["rotation"], (3, 3), f"{source}.rotation")
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0
        ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0):
            raise ValueError(
                "Registration rotation must be proper orthonormal (det=+1)"
            )
        translation = _array(registration["translation_m"], (3,), "translation_m")
        item = segments[source]
        if not isinstance(item, Mapping) or not {
            "mass_kg",
            "com_m",
            "inertia_kg_m2",
        }.issubset(item):
            raise ValueError(
                "Every selected segment requires explicit mass, COM and inertia"
            )
        mass = item["mass_kg"]
        if (
            isinstance(mass, bool)
            or not isinstance(mass, (int, float, np.number))
            or not np.isfinite(mass)
            or mass <= 0
        ):
            raise ValueError("Source segment mass must be positive and finite")
        com = _array(item["com_m"], (3,), "com_m")
        inertia = _inertia(item["inertia_kg_m2"], "inertia_kg_m2")
        edits[name] = {
            "mass_kg": float(mass),
            "com_m": (rotation @ com + translation).tolist(),
            "inertia_kg_m2": (rotation @ inertia @ rotation.T).tolist(),
            "provenance": {
                **provenance,
                "source_body": source,
                "registration": dict(registration),
            },
        }
    return {
        "version": 1,
        "provenance": dict(provenance),
        "source_model_sha256": physical_digest(model),
        "muscles": {},
        "bodies": edits,
    }


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def import_mtu_markups(
    files: Mapping[str, str | Path], *, units: str, coordinate_system: str
) -> dict[str, Any]:
    """Read one pose's local Slicer MTU control polylines, in supplied order.

    ``units`` and ``coordinate_system`` acknowledge source conventions and must
    agree with declarations when present. A Curve's control polygon is measured,
    not its interpolated spline. Coordinates are normalized to RAS metres only;
    they are not registered to native bodies. Harden any scene transforms before
    exporting the markups. Neither total length nor a path
    identifies optimal fibre length or tendon slack length independently.
    """
    if units not in ("mm", "m") or coordinate_system not in ("LPS", "RAS"):
        raise ValueError(
            "Explicit source units mm/m and coordinate_system LPS/RAS are required"
        )
    if not isinstance(files, Mapping) or not files:
        raise ValueError(
            "Map at least one reviewed native actuator name to a local markup file"
        )
    paths: dict[str, Any] = {}
    for name, filename in files.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Native actuator names must be nonempty strings")
        path = Path(filename)
        raw = _read(path)
        try:
            document = json.loads(raw, object_pairs_hook=_json_object)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid local markup JSON: {path.name}") from exc
        markups = document.get("markups") if isinstance(document, dict) else None
        if not isinstance(markups, list) or len(markups) != 1:
            raise ValueError(
                "Each local file must contain exactly one unambiguous markup"
            )
        markup = markups[0]
        if not isinstance(markup, dict) or markup.get("type") not in (
            "Fiducial",
            "Line",
            "Curve",
        ):
            raise ValueError(
                "MTU imports support open Fiducial, Line or Curve polylines"
            )
        if markup.get("coordinateSystem", coordinate_system) != coordinate_system:
            raise ValueError("coordinate_system disagrees with the markup declaration")
        declared_units = markup.get("coordinateUnits", units)
        # Slicer also serializes units as a DICOM terminology triple.
        if isinstance(declared_units, list):
            if (
                len(declared_units) != 3
                or not all(isinstance(x, str) for x in declared_units)
                or declared_units[1] != "UCUM"
            ):
                raise ValueError("Unsupported coordinateUnits coding scheme")
            declared_units = declared_units[0]
        if declared_units != units:
            raise ValueError(
                "units disagree with the markup declaration; convert explicitly first"
            )
        controls = markup.get("controlPoints")
        if not isinstance(controls, list) or len(controls) < 2:
            raise ValueError(
                "Each MTU path requires at least two ordered control points"
            )
        positions = []
        for point in controls:
            if (
                not isinstance(point, dict)
                or point.get("positionStatus", "defined") != "defined"
            ):
                raise ValueError("Every control point must have a defined position")
            positions.append(
                _array(point.get("position"), (3,), "control point position")
            )
        points = np.array(positions) * (0.001 if units == "mm" else 1.0)
        if coordinate_system == "LPS":
            points *= np.array([-1, -1, 1])
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if not np.isfinite(length) or length <= 0:
            raise ValueError("MTU control polyline must have positive finite length")
        paths[name] = {
            "points_ras_m": points.tolist(),
            "polyline_length_m": length,
            "markup_type": markup["type"],
            "source_filename": path.name,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "source_units": units,
            "source_coordinate_system": coordinate_system,
        }
    return {
        "version": 1,
        "provenance": {
            "status": "estimated",
            "source": "Reviewed local 3D Slicer MTU control polylines",
            "note": "Control polygon length, not spline arc length, fibre length or tendon slack",
        },
        "actuator_names": list(paths),
        "mtu_length_m": [[path["polyline_length_m"] for path in paths.values()]],
        "paths": paths,
    }


def mtu_path_observations(
    markups: Sequence[Mapping[str, Any]],
    qpos: Any,
    *,
    time: Any,
    recording_ids: Sequence[str],
    source_model_sha256: str | None = None,
    source_kinematic_sha256: str | None = None,
) -> dict[str, Any]:
    """Combine one-pose imports with reviewed whole-native-model coordinates.

    No OpenSim-angle or ultrasound-pose retargeting is attempted. ``qpos`` uses
    the chosen MuJoCo model's full coordinate ordering; native units are SI,
    including radians and unit quaternions where applicable. The path fitter
    checks model dimensions, native spatial tendons and quaternion validity.
    """
    if not markups:
        raise ValueError("At least one reviewed pose import is required")
    count = len(markups)
    raw_qpos = np.asarray(qpos)
    if raw_qpos.ndim != 2 or raw_qpos.shape[1] < 1:
        raise ValueError("qpos must be T by full native nq")
    poses = _array(raw_qpos, (count, raw_qpos.shape[1]), "qpos")
    times = _array(time, (count,), "time")
    ids = list(recording_ids)
    if len(ids) != count or any(not isinstance(x, str) or not x.strip() for x in ids):
        raise ValueError("recording_ids must contain one nonempty string per pose")
    for recording in set(ids):
        if np.any(np.diff(times[np.array(ids) == recording]) <= 0):
            raise ValueError("Time must increase within each recording")
    names = list(markups[0].get("actuator_names", []))
    if not names or len(set(names)) != len(names):
        raise ValueError("Imported actuator_names must be nonempty and unique")
    lengths = []
    sources = []
    for pose in markups:
        if pose.get("actuator_names") != names:
            raise ValueError("All poses must have identical ordered actuator_names")
        lengths.append(
            _array(pose.get("mtu_length_m"), (1, len(names)), "mtu_length_m")[0]
        )
        sources.append(pose.get("paths", {}))
    if np.any(np.array(lengths) <= 0):
        raise ValueError("MTU lengths must be positive")
    conventions: dict[str, Any] = {
        "path_representation": "Ordered control polygon; Curve spline is not evaluated",
        "pose_registration": "Explicit user-reviewed full native qpos; no automatic retargeting",
        "pose_sources": sources,
        "pose_provenance": [dict(pose["provenance"]) for pose in markups],
    }
    for key, value in (
        ("source_model_sha256", source_model_sha256),
        ("source_kinematic_sha256", source_kinematic_sha256),
    ):
        if value is not None:
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{key} must be a lowercase SHA256 digest")
            conventions[key] = value
    return {
        "version": 1,
        "provenance": dict(markups[0]["provenance"]),
        "units": {"time": "s", "qpos": "SI", "mtu_length_m": "m"},
        "time": times.tolist(),
        "qpos": poses.tolist(),
        "actuator_names": names,
        "mtu_length_m": np.array(lengths).tolist(),
        "recording_ids": ids,
        "conventions": conventions,
    }
