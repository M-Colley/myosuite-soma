# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Local-only conversion of SOMA caches and reference anatomy to GLB."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from myosuite_soma.soma import (
    SomaConfig,
    _body_id,
    _generate_animation,
    _load_predictor,
    _mappings,
    _require_qpos_body,
    model_digest,
)


def _joint_bodies(
    config: SomaConfig,
    model: mujoco.MjModel,
    names: Sequence[str],
    parents: np.ndarray,
    joint_map: Mapping[str, str] | None,
) -> tuple[dict[str, str | None], np.ndarray]:
    """Resolve named rig bindings, inheriting only along the SOMA hierarchy."""
    names = tuple(names)
    parents = np.asarray(parents)
    if (
        not names
        or len(set(names)) != len(names)
        or parents.shape != (len(names),)
        or parents.dtype.kind not in "iu"
        or int(parents[0]) != -1
        or any(p < 0 or p >= i for i, p in enumerate(parents[1:], 1))
    ):
        raise ValueError("SOMA joint hierarchy must be unique and root-first")
    explicit = {name: spec.body for name, spec in _mappings(config).items()}
    root = config.root_body
    if config.preset == "myoElbow1D6":
        root = root or "r_humerus"
        explicit.setdefault("RightArm", "r_humerus")
    if root is not None:
        explicit.setdefault(names[0], root)
    if config.anchor_joint is not None:
        explicit.setdefault(config.anchor_joint, config.anchor_body)
    if joint_map is not None:
        if not isinstance(joint_map, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) and k and v
            for k, v in joint_map.items()
        ):
            raise ValueError("joint_map must map exact SOMA joint names to body names")
        explicit.update(joint_map)
    unknown = explicit.keys() - set(names)
    if unknown:
        raise ValueError(f"Unknown SOMA joint bindings: {sorted(unknown)}")
    # Validate every explicit name, including joints with zero vertex weight.
    for body in explicit.values():
        body_id = _body_id(model, body)
        _require_qpos_body(model, body_id)
    resolved: dict[str, str | None] = {}
    ids = np.full(len(names), -1, dtype=np.int64)
    for index, name in enumerate(names):
        parent = int(parents[index])
        body = explicit.get(name)
        if body is None and parent >= 0:
            body = resolved[names[parent]]
        resolved[name] = body
        if body is not None:
            ids[index] = _body_id(model, body)
    return resolved, ids


def _rig(
    model: mujoco.MjModel,
    native_weights: Mapping[str, np.ndarray],
    joint_ids: np.ndarray,
    joint_names: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fold all native influences onto named source bodies without pruning."""
    used: set[int] = set()
    for name, value in native_weights.items():
        weight = np.asarray(value)
        if (
            weight.dtype.kind not in "fiu"
            or weight.ndim != 2
            or weight.shape[1] != len(joint_ids)
            or not len(weight)
            or not np.isfinite(weight).all()
            or np.any(weight < 0)
            or np.any(weight.sum(1) <= 0)
        ):
            raise ValueError(f"Invalid native skinning weights for {name}")
        active = np.any(weight > 0, axis=0)
        missing = active & (joint_ids < 0)
        if np.any(missing):
            unknown = [n for n, absent in zip(joint_names, missing) if absent]
            raise ValueError(
                f"Weighted SOMA joints have no source binding: {unknown}. "
                "Set root_body or provide --joint-map; no spatial guess is made."
            )
        used.update(int(body) for body in joint_ids[active])
    # Include named native ancestors to retain an editable chained skeleton.
    for body in tuple(used):
        parent = int(model.body_parentid[body])
        while parent > 0:
            if model.body(parent).name:
                used.add(parent)
            parent = int(model.body_parentid[parent])
    body_ids = sorted(used)
    body_names = tuple(model.body(i).name for i in body_ids)
    if not body_names or any(not n for n in body_names):
        raise ValueError("GLB bindings require named source bodies")
    columns = {body: i for i, body in enumerate(body_ids)}
    parents = np.full(len(body_ids), -1, dtype=np.int64)
    for column, body in enumerate(body_ids):
        if body == 0:
            continue
        parent = int(model.body_parentid[body])
        while parent > 0 and parent not in columns:
            parent = int(model.body_parentid[parent])
        if parent in columns:
            parents[column] = columns[parent]
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    matrices = np.broadcast_to(np.eye(4), (len(body_ids), 4, 4)).copy()
    matrices[:, :3, :3] = data.xmat[body_ids].reshape(-1, 3, 3)
    matrices[:, :3, 3] = data.xpos[body_ids]
    weights = {}
    for name, value in native_weights.items():
        folded = np.zeros((len(value), len(body_ids)), dtype=np.float64)
        for native, body in enumerate(joint_ids):
            if body in columns:
                folded[:, columns[body]] += value[:, native]
        weights[name] = folded / folded.sum(1, keepdims=True)
    return body_names, parents, matrices, weights


def export_reference_glb(
    config_path: Path | str,
    model: mujoco.MjModel,
    output: Path | str,
    *,
    include_muscles: bool = False,
    joint_map: Mapping[str, str] | None = None,
) -> Path:
    """Export a body-named reference rig compatible with MyoSuite's skin loader.

    Anatomy is predicted at ``model.qpos0``, even when the calibration stores a
    different rest pose. Native SOMA weights follow explicit joint bindings or
    their nearest mapped SOMA ancestor. The resulting ordinary skinning is an
    approximation to SOMA; it does not evaluate the network during playback.
    ``--skin`` reads the skin surface only; additional muscles are for GLB tools.
    No input model or caller-owned state is modified, and no asset is downloaded.
    """
    from myosuite_soma.glb import export_rigged_glb

    output = Path(output).resolve()
    if output.exists() or output.suffix.lower() != ".glb":
        raise ValueError("Choose a new .glb file")
    config = SomaConfig.from_json(config_path)
    if config.conditioning_model is not None:
        raise ValueError(
            "Reference rig export needs a pose-only config. Export a measured "
            "conditioned animation cache with 'export-glb animation' instead."
        )
    before = model_digest(model)
    predictor = _load_predictor(config, config_path)
    resolved, joint_ids = _joint_bodies(
        config, model, predictor.joint_names, predictor.parents, joint_map
    )
    names = tuple(
        model.actuator(i).name for i in range(model.nu) if model.actuator(i).name
    )
    animation = _generate_animation(
        config_path,
        model,
        model.qpos0[None].copy(),
        np.array([0.0]),
        names,
        np.zeros((1, len(names))),
        predictor=predictor,
    )
    if not include_muscles:
        animation = replace(
            animation, meshes=tuple(m for m in animation.meshes if m.kind == "skin")
        )
    native = predictor.skinning_weights()
    selected = {mesh.name: native[mesh.name] for mesh in animation.meshes}
    body_names, parents, matrices, weights = _rig(
        model, selected, joint_ids, predictor.joint_names
    )
    animation.metadata["glb_rigging"] = {
        "mode": "reference_native_body_lbs",
        "reference": "model.qpos0",
        "reference_qpos": model.qpos0.tolist(),
        "resolved_joint_bodies": resolved,
        "unmapped_joint_behavior": "nearest_mapped_soma_ancestor_or_error",
        "weight_policy": "sum_all_native_influences_without_pruning",
        "deformation_during_playback": "native_body_lbs_approximation",
        "forward_physics_changed": False,
        "includes_muscles": include_muscles,
    }
    if model_digest(model) != before:
        raise RuntimeError("Reference GLB export changed the input model")
    return export_rigged_glb(
        animation,
        output,
        body_names=body_names,
        parents=parents,
        bind_matrices=matrices,
        weights=weights,
    )


def main(argv: list[str] | None = None) -> None:
    """Convert independently obtained assets locally; never download or upload."""
    parser = argparse.ArgumentParser(prog="myosuite-soma export-glb")
    commands = parser.add_subparsers(dest="mode", required=True)
    animation_parser = commands.add_parser("animation", help="Bake mesh keyframes")
    animation_parser.add_argument("cache", type=Path)
    animation_parser.add_argument("output", type=Path)
    animation_parser.add_argument("--max-targets", type=int, default=256)
    rigged = commands.add_parser("rigged", help="Export a MyoSuite-compatible skin")
    rigged.add_argument("--config", type=Path, required=True)
    source = rigged.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Native MuJoCo XML model")
    source.add_argument("--env", help="Optional installed MyoSuite environment")
    rigged.add_argument("output", type=Path)
    rigged.add_argument("--include-muscles", action="store_true")
    rigged.add_argument("--joint-map", type=Path, help="JSON SOMA-joint to body names")
    args = parser.parse_args(argv)
    env: Any = None
    try:
        if args.mode == "animation":
            from myosuite_soma.glb import export_glb
            from myosuite_soma.soma import load_animation
            from myosuite_soma.soma_cache import AnimationArchive

            animation = (
                AnimationArchive(args.cache).load()
                if args.cache.is_dir()
                else load_animation(args.cache)
            )
            print(export_glb(animation, args.output, max_targets=args.max_targets))
            return
        joint_map = (
            json.loads(args.joint_map.read_text(encoding="utf-8"))
            if args.joint_map is not None
            else None
        )
        if args.env is not None:
            try:
                from myosuite import make_env
            except ImportError as exc:
                raise ImportError(
                    "--env requires optional MyoSuite; --model accepts XML directly"
                ) from exc
            env = make_env(args.env)
            model = env.unwrapped.model
        else:
            model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        print(
            export_reference_glb(
                args.config,
                model,
                args.output,
                include_muscles=args.include_muscles,
                joint_map=joint_map,
            )
        )
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
