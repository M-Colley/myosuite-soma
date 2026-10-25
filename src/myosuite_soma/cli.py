"""Command-line entry points with lazy optional dependencies."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib import metadata
from pathlib import Path

from myosuite_soma import __version__

COMMANDS = {
    "retarget": "soma_retarget",
    "cache": "soma_cache",
    "conditioning": "soma_conditioning",
    "benchmark": "benchmark",
    "physical": "physical",
}


def _doctor() -> None:
    report = {"myosuite-soma": __version__}
    for name in (
        "numpy",
        "scipy",
        "mujoco",
        "torch",
        "osqp",
        "myosuite",
        "usd-core",
        "ezc3d",
    ):
        try:
            report[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report[name] = "not installed"
    report["adapter"] = (
        "Run render to check the optional adapter; see integration/README.md"
    )
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    """Dispatch a package command while preserving the caller's argv.

    Args:
        argv: Explicit arguments, or process arguments when omitted.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="myosuite-soma",
        description="SOMA muscle/skin inference, caches and optional MyoSuite rendering",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            *COMMANDS,
            "render",
            "install-adapter",
            "export-usd",
            "export-glb",
            "doctor",
        ],
    )
    parsed = parser.parse_args(arguments[:1])
    if parsed.command is None:
        parser.print_help()
        return
    rest = arguments[1:]
    if parsed.command == "doctor":
        if rest:
            parser.error("doctor does not take arguments")
        _doctor()
        return
    if parsed.command == "install-adapter":
        from myosuite_soma.adapter import install_adapter

        options = argparse.ArgumentParser(prog="myosuite-soma install-adapter")
        options.add_argument("--myosuite-source", required=True, type=Path)
        options.add_argument("--check", action="store_true")
        args = options.parse_args(rest)
        try:
            print(install_adapter(args.myosuite_source, check_only=args.check))
        except (ValueError, FileNotFoundError) as error:
            options.error(str(error))
        return
    if parsed.command == "export-usd":
        from myosuite_soma.usd import main as export

        export(rest)
        return
    if parsed.command == "export-glb":
        from myosuite_soma.glb_export import main as export_glb

        export_glb(rest)
        return
    if parsed.command == "render":
        from myosuite_soma.adapter import renderer

        try:
            renderer().main(rest)
        except (ImportError, RuntimeError) as error:
            parser.error(str(error))
        return
    module = importlib.import_module(f"myosuite_soma.{COMMANDS[parsed.command]}")
    previous = sys.argv
    try:
        sys.argv = [f"myosuite-soma {parsed.command}", *rest]
        module.main()
    finally:
        sys.argv = previous
