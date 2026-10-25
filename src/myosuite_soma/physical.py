# Copyright (c) Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Local-only CLI for explicit physical variants, calibration and research models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import mujoco
import numpy as np


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration files must contain JSON objects")
    return value


@contextmanager
def _model(args):
    env = None
    try:
        if args.env:
            try:
                from myosuite import make_env
            except ImportError as exc:
                raise ImportError(
                    "--env requires optional MyoSuite; --model accepts XML/MJB directly"
                ) from exc
            env = make_env(args.env)
            model = env.unwrapped.model
            source = model
            # Pinned MyoSuite stores its native spec separately from Gym's
            # EnvSpec. Copy live parameter edits into a private spec, then use
            # its canonical compiled baseline consistently across commands.
            for name in ("spec", "_mj_spec"):
                candidate = getattr(env.unwrapped, name, None)
                if isinstance(candidate, mujoco.MjSpec):
                    candidate = candidate.copy()
                    if hasattr(candidate, "copy_back"):
                        try:
                            candidate.copy_back(model)
                            snapshot = candidate.compile()
                        except (ValueError, RuntimeError):
                            continue
                        source, model = candidate, snapshot
                        break
        elif args.model.suffix.lower() == ".mjb":
            model = mujoco.MjModel.from_binary_path(str(args.model.resolve()))
            source = model
        else:
            source = mujoco.MjSpec.from_file(str(args.model.resolve()))
            model = source.compile()
        yield source, model
    finally:
        if env is not None:
            env.close()


def _native_source(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Local MuJoCo XML or MJB")
    source.add_argument("--env", help="Optional installed MyoSuite environment")


def _numeric(path, allowed):
    from myosuite_soma.soma_numeric import inspect_numeric_archive

    inspect_numeric_archive(path, max_bytes=256 * 1024**2)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) - set(allowed):
            raise ValueError(f"Unexpected arrays; allowed keys are {sorted(allowed)}")
        result = {key: archive[key].copy() for key in archive.files}
    if any(
        a.dtype.kind not in "fiu" or not np.isfinite(a).all() for a in result.values()
    ):
        raise ValueError("Trajectory/control arrays must be finite real numbers")
    return result


def _simulate(source, args):
    from myosuite_soma.compliant_muscle import CompliantMuscleSystem
    from myosuite_soma.physical_data import save_json

    output = args.output.resolve()
    if output.exists():
        raise ValueError("Choose a new compliant rollout output directory")
    arrays = _numeric(
        args.controls, {"ctrl", "initial_qpos", "initial_qvel", "initial_activation"}
    )
    controls = arrays.get("ctrl")
    if (
        controls is None
        or controls.ndim != 2
        or controls.shape[1] != source.nu
        or not len(controls)
    ):
        raise ValueError("ctrl must have shape (steps, model.nu)")
    if len(controls) > 100000:
        raise ValueError("Limit compliant research rollouts to 100000 native steps")
    system = CompliantMuscleSystem(
        source, _json(args.parameters), allow_synthetic=args.allow_synthetic
    )
    system.reset(
        qpos=arrays.get("initial_qpos"),
        qvel=arrays.get("initial_qvel"),
        activation=arrays.get("initial_activation"),
    )
    time, poses, velocity, force, length, angle, residual = [], [], [], [], [], [], []
    for ctrl in controls:
        sample = system.step(ctrl)
        time.append(sample["time"])
        poses.append(system.data.qpos.copy())
        velocity.append(system.data.qvel.copy())
        states = [sample["muscles"][name] for name in system.names]
        force.append([s["tendon_force_N"] for s in states])
        length.append([s["fibre_length_m"] for s in states])
        angle.append([s["pennation_deg"] for s in states])
        residual.append([s["force_balance_residual_N"] for s in states])
    output.mkdir(parents=True, exist_ok=False)
    # These are private simulation results, never package/release resources.
    with (output / "rollout.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            time=time,
            qpos=poses,
            qvel=velocity,
            tendon_force_N=force,
            fibre_length_m=length,
            pennation_deg=angle,
            force_balance_residual_N=residual,
        )
    save_json(
        {
            "research_only": True,
            "constitutive_model": "quadratic elastic tendon; constant-thickness pennation; native FLV; implicit damped fibre step",
            "integration": "operator-split forces held over each native physics step",
            "native_actuator_force_contains_replacement_forces": False,
            "muscles": list(system.names),
            "parameters": system.config,
            "samples": len(time),
            "dt_s": float(source.opt.timestep),
            "max_force_balance_residual_N": float(np.max(np.abs(residual))),
            "physiological_accuracy_validated": False,
        },
        output / "report.json",
    )
    return output


def main(argv: list[str] | None = None) -> None:
    """Run explicit offline physical operations; never download or upload assets."""
    from myosuite_soma.physical_data import (
        architecture_profile,
        attachment_profile,
        dataset_catalog,
        import_architecture,
        import_hand_attachments,
        load_observations,
        save_json,
        save_observations,
    )

    parser = argparse.ArgumentParser(prog="myosuite-soma physical")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("datasets", help="List source data, licenses and supported adapters")
    architecture = sub.add_parser(
        "import-architecture", help="Normalize local DOCX/CSV anatomy tables"
    )
    architecture.add_argument("--source", choices=["charles", "hand"], required=True)
    architecture.add_argument("--input", type=Path, required=True)
    architecture.add_argument("--subject")
    architecture.add_argument("--mapping", type=Path)
    architecture.add_argument("--profile-output", type=Path)
    architecture.add_argument("--output", type=Path, required=True)
    attachment = sub.add_parser(
        "import-attachments", help="Read hand landmark coordinates"
    )
    attachment.add_argument("--input", type=Path, required=True)
    attachment.add_argument("--output", type=Path, required=True)
    observation = sub.add_parser(
        "import-observations", help="Normalize local torque CSV/NPZ"
    )
    observation.add_argument("--input", type=Path, required=True)
    observation.add_argument("--metadata", type=Path)
    observation.add_argument("--output", type=Path, required=True)
    dynamometer = sub.add_parser(
        "import-dynamometer", help="Convert explicitly mapped local C3D channels"
    )
    dynamometer.add_argument("--input", type=Path, required=True)
    dynamometer.add_argument("--config", type=Path, required=True)
    dynamometer.add_argument("--output", type=Path, required=True)
    segments = sub.add_parser(
        "import-segments", help="Read processed OpenSim segment estimates"
    )
    segments.add_argument("--input", type=Path, required=True)
    segments.add_argument("--output", type=Path, required=True)
    mtu = sub.add_parser(
        "import-mtu", help="Read explicitly named local Slicer path markups"
    )
    mtu.add_argument(
        "--files",
        type=Path,
        required=True,
        help="JSON actuator-name to markup-path map",
    )
    mtu.add_argument("--units", choices=["mm", "m"], required=True)
    mtu.add_argument("--coordinate-system", choices=["LPS", "RAS"], required=True)
    mtu.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser(
        "prepare-paths", help="Combine path imports with reviewed native qpos"
    )
    prepare.add_argument(
        "--markups",
        type=Path,
        required=True,
        help="JSON object with ordered poses file list",
    )
    prepare.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="Numeric NPZ with qpos/time/recording_ids",
    )
    prepare.add_argument("--output", type=Path, required=True)
    for command in (
        "build",
        "fit",
        "validate",
        "allometry",
        "fit-capsules",
        "map-attachments",
        "simulate-compliant",
        "map-segments",
        "fit-paths",
        "validate-dynamics",
    ):
        p = sub.add_parser(command)
        _native_source(p)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--allow-synthetic", action="store_true")
        if command == "build":
            p.add_argument("--profile", type=Path, required=True)
        if command in {"fit", "validate"}:
            p.add_argument("--observations", type=Path, required=True)
            p.add_argument("--metadata", type=Path)
            p.add_argument("--config", type=Path, required=True)
        if command == "fit-paths":
            p.add_argument("--observations", type=Path, required=True)
            p.add_argument("--config", type=Path, required=True)
        if command == "validate-dynamics":
            p.add_argument("--observations", type=Path, required=True)
        if command == "map-segments":
            p.add_argument("--segments", type=Path, required=True)
            p.add_argument("--mapping", type=Path, required=True)
        if command == "allometry":
            p.add_argument("--scales", type=Path, required=True)
            p.add_argument("--provenance", type=Path, required=True)
            p.add_argument(
                "--mass-policy",
                choices=["preserve", "constant_density", "target_total"],
                default="preserve",
            )
            p.add_argument("--target-mass-kg", type=float)
        if command == "fit-capsules":
            p.add_argument("--cache", type=Path, required=True)
            p.add_argument("--trajectory", type=Path, required=True)
            p.add_argument("--assignments", type=Path, required=True)
            p.add_argument("--provenance", type=Path, required=True)
        if command == "map-attachments":
            p.add_argument("--anatomy", type=Path, required=True)
            p.add_argument("--mapping", type=Path, required=True)
            p.add_argument("--registration", type=Path, required=True)
        if command == "simulate-compliant":
            p.add_argument("--parameters", type=Path, required=True)
            p.add_argument("--controls", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "datasets":
            print(json.dumps(dataset_catalog(), indent=2))
            return
        if args.command == "import-architecture":
            if bool(args.mapping) != bool(args.profile_output):
                raise ValueError(
                    "--mapping and --profile-output must be provided together"
                )
            data = import_architecture(args.source, args.input, subject=args.subject)
            profile = (
                architecture_profile(data, _json(args.mapping))
                if args.mapping
                else None
            )
            print(save_json(data, args.output))
            if profile is not None:
                print(save_json(profile, args.profile_output))
            return
        if args.command == "import-attachments":
            print(save_json(import_hand_attachments(args.input), args.output))
            return
        if args.command == "import-observations":
            print(
                save_observations(
                    load_observations(args.input, metadata=args.metadata), args.output
                )
            )
            return
        if args.command == "import-dynamometer":
            from myosuite_soma.physical_data import (
                read_c3d_signals,
                dynamometer_observations,
            )

            print(
                save_observations(
                    dynamometer_observations(
                        read_c3d_signals(args.input), _json(args.config)
                    ),
                    args.output,
                )
            )
            return
        if args.command == "import-segments":
            from myosuite_soma.physical_sources import import_segment_properties

            print(save_json(import_segment_properties(args.input), args.output))
            return
        if args.command == "import-mtu":
            from myosuite_soma.physical_sources import import_mtu_markups

            mapping = _json(args.files)
            if not all(isinstance(value, str) and value for value in mapping.values()):
                raise ValueError("Path-file map values must be nonempty local paths")
            mapping = {
                key: (args.files.resolve().parent / value).resolve()
                for key, value in mapping.items()
            }
            print(
                save_json(
                    import_mtu_markups(
                        mapping,
                        units=args.units,
                        coordinate_system=args.coordinate_system,
                    ),
                    args.output,
                )
            )
            return
        if args.command == "prepare-paths":
            from myosuite_soma.physical_sources import mtu_path_observations
            from myosuite_soma.soma_numeric import inspect_numeric_archive

            paths = _json(args.markups).get("poses")
            if not isinstance(paths, list) or not all(
                isinstance(path, str) and path for path in paths
            ):
                raise ValueError(
                    "Path manifest requires an ordered poses list of local JSON paths"
                )
            markups = [_json(args.markups.resolve().parent / path) for path in paths]
            inspect_numeric_archive(
                args.trajectory, max_bytes=256 * 1024**2, allow_text=True
            )
            with np.load(args.trajectory, allow_pickle=False) as recording:
                data = mtu_path_observations(
                    markups,
                    recording["qpos"],
                    time=recording["time"],
                    recording_ids=recording["recording_ids"].tolist(),
                )
            print(save_json(data, args.output))
            return
        with _model(args) as (source, model):
            if args.command == "build":
                from myosuite_soma.physical_model import (
                    build_physical_model,
                    save_physical_model,
                )

                print(
                    save_physical_model(
                        build_physical_model(
                            source,
                            _json(args.profile),
                            allow_synthetic=args.allow_synthetic,
                        ),
                        args.output,
                    )
                )
            elif args.command in {"fit", "validate"}:
                from myosuite_soma.physical_calibration import (
                    fit_physical_model,
                    validate_physical_model,
                )

                observations = load_observations(
                    args.observations, metadata=args.metadata
                )
                config = _json(args.config)
                if args.command == "validate":
                    print(
                        save_json(
                            validate_physical_model(
                                model,
                                observations,
                                config,
                                allow_synthetic=args.allow_synthetic,
                            ),
                            args.output,
                        )
                    )
                else:
                    from myosuite_soma.physical_model import (
                        build_physical_model,
                        save_physical_model,
                    )

                    fitted = fit_physical_model(
                        model,
                        observations,
                        config,
                        allow_synthetic=args.allow_synthetic,
                    )
                    built = build_physical_model(
                        source, fitted.profile, allow_synthetic=args.allow_synthetic
                    )
                    built.report["calibration"] = fitted.report
                    path = save_physical_model(built, args.output)
                    save_json(fitted.profile, args.output / "fitted-profile.json")
                    print(path)
            elif args.command == "allometry":
                from myosuite_soma.physical_geometry import anthropometric_profile

                print(
                    save_json(
                        anthropometric_profile(
                            model,
                            _json(args.scales),
                            provenance=_json(args.provenance),
                            mass_policy=args.mass_policy,
                            target_mass_kg=args.target_mass_kg,
                        ),
                        args.output,
                    )
                )
            elif args.command == "fit-capsules":
                from myosuite_soma.physical_geometry import (
                    capsule_profile_from_animation,
                )
                from myosuite_soma.soma import load_animation
                from myosuite_soma.soma_cache import AnimationArchive

                animation = (
                    AnimationArchive(args.cache).load()
                    if args.cache.is_dir()
                    else load_animation(args.cache)
                )
                from myosuite_soma.soma_numeric import inspect_numeric_archive
                from myosuite_soma.soma import validate_animation

                inspect_numeric_archive(
                    args.trajectory, max_bytes=256 * 1024**2, allow_text=True
                )
                with np.load(args.trajectory, allow_pickle=False) as recording:
                    poses = {"qpos": recording["qpos"].copy()}
                    if "time" in recording:
                        validate_animation(animation, time=recording["time"])
                print(
                    save_json(
                        capsule_profile_from_animation(
                            animation,
                            model,
                            poses["qpos"],
                            _json(args.assignments),
                            provenance=_json(args.provenance),
                        ),
                        args.output,
                    )
                )
            elif args.command == "map-attachments":
                registration = _json(args.registration)
                if set(registration) - {"rotation", "translation_m", "scale"}:
                    raise ValueError(
                        "Registration uses rotation, translation_m and scale only"
                    )
                print(
                    save_json(
                        attachment_profile(
                            _json(args.anatomy),
                            model,
                            _json(args.mapping),
                            **registration,
                        ),
                        args.output,
                    )
                )
            elif args.command == "map-segments":
                from myosuite_soma.physical_sources import segment_profile

                print(
                    save_json(
                        segment_profile(
                            _json(args.segments), model, _json(args.mapping)
                        ),
                        args.output,
                    )
                )
            elif args.command == "fit-paths":
                from myosuite_soma.physical_paths import fit_path_lengths
                from myosuite_soma.physical_model import (
                    build_physical_model,
                    save_physical_model,
                )

                fitted = fit_path_lengths(
                    model,
                    _json(args.observations),
                    _json(args.config),
                    allow_synthetic=args.allow_synthetic,
                )
                built = build_physical_model(
                    source, fitted.profile, allow_synthetic=args.allow_synthetic
                )
                built.report["path_calibration"] = fitted.report
                print(save_physical_model(built, args.output))
                save_json(fitted.profile, args.output / "fitted-profile.json")
            elif args.command == "validate-dynamics":
                from myosuite_soma.physical_dynamics import validate_inverse_dynamics

                print(
                    save_json(
                        validate_inverse_dynamics(
                            model,
                            _json(args.observations),
                            allow_synthetic=args.allow_synthetic,
                        ),
                        args.output,
                    )
                )
            else:
                print(_simulate(model, args))
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
