"""Explicit, optional integration with MyoSuite's existing Blender renderer."""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import fields
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

TESTED_MYOSUITE_COMMIT = "c0af9c656de2aa1807924e63d0d5320d77aa1ebd"


def renderer() -> Any:
    """Return the optional renderer, requiring the explicit SOMA adapter.

    Raises:
        ImportError: MyoSuite is unavailable.
        RuntimeError: The renderer lacks the SOMA integration.
    """
    try:
        module = importlib.import_module("myosuite.viz.blender_render")
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == "myosuite":
            raise ImportError(
                "MyoSuite is optional. Install the [myosuite,render,inference] "
                "extras and the adapter described in integration/README.md."
            ) from error
        raise
    expected = {"soma_config", "soma_cache", "soma_cache_format"}
    if not expected <= {field.name for field in fields(module.RenderConfig)}:
        raise RuntimeError(
            "This MyoSuite renderer has no SOMA adapter. Use a compatible source "
            "checkout and run: myosuite-soma install-adapter --myosuite-source PATH. "
            f"The tested base is {TESTED_MYOSUITE_COMMIT}."
        )
    return module


def render_rollout(config: Any, *, export_only: bool = False) -> Any:
    """Render through the adapted upstream pipeline without altering physics.

    Args:
        config: Adapted MyoSuite RenderConfig.
        export_only: Export numeric states and USD without launching Blender.

    Returns:
        The upstream renderer result.
    """
    return renderer().render_rollout(config, export_only=export_only)


def install_adapter(source: Path | str, *, check_only: bool = False) -> str:
    """Check and explicitly apply the packaged patch to a source checkout.

    Args:
        source: Root of a MyoSuite Git checkout.
        check_only: Verify applicability without modifying source files.

    Returns:
        A short completion message.

    Raises:
        ValueError: The directory is invalid or the patch cannot apply cleanly.
        FileNotFoundError: Git is unavailable.
    """
    target = Path(source).resolve()
    if not (target / "myosuite/viz/blender_render.py").is_file():
        raise ValueError(f"Not a MyoSuite source checkout: {target}")
    root = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if root.returncode or Path(root.stdout.strip()).resolve() != target:
        raise ValueError("Pass the root of the MyoSuite Git checkout")
    revision = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode:
        raise ValueError("The MyoSuite source checkout needs a committed HEAD")
    head = revision.stdout.strip()
    if head != TESTED_MYOSUITE_COMMIT:
        raise ValueError(
            f"This adapter requires MyoSuite {TESTED_MYOSUITE_COMMIT}; "
            f"the checkout is at {head}. Source was left unchanged."
        )
    resource = files("myosuite_soma").joinpath("assets/myosuite-adapter.patch")
    with as_file(resource) as patch:
        command = ["git", "-C", str(target), "apply"]
        checked = subprocess.run(
            [*command, "--check", str(patch)],
            check=False,
            capture_output=True,
            text=True,
        )
        if checked.returncode:
            raise ValueError(
                "Adapter cannot apply cleanly; source was left unchanged. "
                "It may already be installed or this checkout may differ from "
                f"the tested base {TESTED_MYOSUITE_COMMIT}.\n{checked.stderr.strip()}"
            )
        if not check_only:
            subprocess.run([*command, str(patch)], check=True, capture_output=True)
    return "Adapter patch check passed" if check_only else "Adapter patch applied"
