"""The opt-in adapter checks and applies only to an explicit source checkout."""

from dataclasses import make_dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from myosuite_soma import adapter


def test_missing_myosuite_explains_optional_adapter(monkeypatch):
    def absent(name):
        assert name == "myosuite.viz.blender_render"
        raise ModuleNotFoundError("No module named myosuite", name="myosuite")

    monkeypatch.setattr(adapter.importlib, "import_module", absent)
    with pytest.raises(ImportError, match="MyoSuite is optional"):
        adapter.renderer()


def test_internal_dependency_failure_is_not_disguised_as_missing_myosuite(monkeypatch):
    failure = ModuleNotFoundError("Missing internal dependency", name="gymnasium")

    def absent(_):
        raise failure

    monkeypatch.setattr(adapter.importlib, "import_module", absent)
    with pytest.raises(ModuleNotFoundError) as stopped:
        adapter.renderer()
    assert stopped.value is failure


def test_unadapted_renderer_is_rejected_and_adapted_calls_delegate(monkeypatch):
    module = SimpleNamespace(RenderConfig=make_dataclass("Old", [("env", str)]))
    monkeypatch.setattr(adapter.importlib, "import_module", lambda _: module)
    with pytest.raises(RuntimeError, match="no SOMA adapter"):
        adapter.renderer()
    module.RenderConfig = make_dataclass(
        "Adapted",
        [(name, str) for name in ("soma_config", "soma_cache", "soma_cache_format")],
    )
    config = object()
    module.render_rollout = lambda value, **kw: (value, kw)
    assert adapter.render_rollout(config, export_only=True) == (
        config,
        {"export_only": True},
    )


def _checkout(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("Git is required to exercise explicit patch installation")
    fixture = Path(__file__).parent / "fixtures/myosuite-base"
    manifest = json.loads((fixture / "manifest.json").read_text())
    target = tmp_path / "checkout"
    target.mkdir()
    for name, expected in manifest["files"].items():
        content = (fixture / (name + ".txt")).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    commands = [
        ["init", "--quiet"],
        ["config", "core.autocrlf", "false"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "canonical source fixture",
        ],
    ]
    for command in commands:
        subprocess.run(
            ["git", "-C", str(target), *command], check=True, capture_output=True
        )
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(adapter, "TESTED_MYOSUITE_COMMIT", head)
    return target, tuple(manifest["files"])


def _snapshot(target, names):
    return {name: (target / name).read_bytes() for name in names}


def test_git_patch_check_is_readonly_apply_is_explicit_and_repeat_is_rejected(
    tmp_path, monkeypatch
):
    target, names = _checkout(tmp_path, monkeypatch)
    before = _snapshot(target, names)
    assert (
        adapter.install_adapter(target, check_only=True) == "Adapter patch check passed"
    )
    assert _snapshot(target, names) == before
    assert adapter.install_adapter(target) == "Adapter patch applied"
    after = _snapshot(target, names)
    assert all(after[name] != before[name] for name in names)
    assert b"myosuite_soma" in after["myosuite/viz/blender_render.py"]
    with pytest.raises(ValueError, match="cannot apply cleanly"):
        adapter.install_adapter(target)
    assert _snapshot(target, names) == after


def test_wrong_git_head_and_conflicting_source_leave_files_unchanged(
    tmp_path, monkeypatch
):
    target, names = _checkout(tmp_path, monkeypatch)
    before = _snapshot(target, names)
    tested = adapter.TESTED_MYOSUITE_COMMIT
    monkeypatch.setattr(adapter, "TESTED_MYOSUITE_COMMIT", "0" * 40)
    with pytest.raises(ValueError):
        adapter.install_adapter(target)
    assert _snapshot(target, names) == before
    monkeypatch.setattr(adapter, "TESTED_MYOSUITE_COMMIT", tested)
    (target / "myosuite/viz/blender_render.py").write_text("incompatible source\n")
    conflict = _snapshot(target, names)
    with pytest.raises(ValueError, match="cannot apply cleanly"):
        adapter.install_adapter(target)
    assert _snapshot(target, names) == conflict


def test_adapter_rejects_noncheckout_and_subdirectory(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Not a MyoSuite source checkout"):
        adapter.install_adapter(tmp_path)
    target, _ = _checkout(tmp_path, monkeypatch)
    nested = target / "nested"
    path = nested / "myosuite/viz/blender_render.py"
    path.parent.mkdir(parents=True)
    path.write_text("placeholder\n")
    with pytest.raises(ValueError, match="root of the MyoSuite Git checkout"):
        adapter.install_adapter(nested)


def test_adapter_reports_unborn_git_head_as_invalid_checkout(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Git is required to exercise explicit patch installation")
    target = tmp_path / "unborn"
    path = target / "myosuite/viz/blender_render.py"
    path.parent.mkdir(parents=True)
    path.write_text("placeholder\n")
    subprocess.run(
        ["git", "-C", str(target), "init", "--quiet"], check=True, capture_output=True
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="commit|HEAD|checkout"):
        adapter.install_adapter(target)
    assert path.read_bytes() == before
