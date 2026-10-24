"""Dispatch, help and caller-state contracts for the standalone entry point."""

import json
import sys
from types import SimpleNamespace

import pytest

from myosuite_soma import __version__, cli


@pytest.mark.parametrize("arguments", [[], ["--help"], ["--version"]])
def test_top_level_help_and_version_need_no_backend_imports(
    arguments, monkeypatch, capsys
):
    previous = sys.argv

    def unexpected_import(name):
        raise AssertionError(f"Help imported a backend: {name}")

    monkeypatch.setattr(cli.importlib, "import_module", unexpected_import)
    if arguments:
        with pytest.raises(SystemExit) as stopped:
            cli.main(arguments)
        assert stopped.value.code == 0
    else:
        cli.main(arguments)
    output = capsys.readouterr().out
    assert (__version__ if arguments == ["--version"] else "myosuite-soma") in output
    assert sys.argv is previous


@pytest.mark.parametrize("command,module_name", list(cli.COMMANDS.items()))
def test_subcommand_help_is_delegated_and_argv_restored(
    command, module_name, monkeypatch
):
    previous = sys.argv
    observed = []

    def main():
        observed.append(sys.argv[:])
        raise SystemExit(0)

    def import_module(name):
        assert name == f"myosuite_soma.{module_name}"
        return SimpleNamespace(main=main)

    monkeypatch.setattr(cli.importlib, "import_module", import_module)
    arguments = [command, "--help"]
    with pytest.raises(SystemExit) as stopped:
        cli.main(arguments)
    assert stopped.value.code == 0
    assert arguments == [command, "--help"]
    assert observed == [[f"myosuite-soma {command}", "--help"]]
    assert sys.argv is previous


def test_dispatch_uses_process_arguments_and_restores_on_failure(monkeypatch):
    previous = ["caller", "cache", "inspect", "file.npz"]
    monkeypatch.setattr(sys, "argv", previous)

    def main():
        assert sys.argv == ["myosuite-soma cache", "inspect", "file.npz"]
        raise RuntimeError("backend failure")

    monkeypatch.setattr(
        cli.importlib, "import_module", lambda _: SimpleNamespace(main=main)
    )
    with pytest.raises(RuntimeError, match="backend failure"):
        cli.main()
    assert sys.argv is previous


def test_export_and_render_delegate_explicit_arguments(monkeypatch):
    from myosuite_soma import adapter, usd

    received = []
    monkeypatch.setattr(usd, "main", lambda argv: received.append(("usd", argv)))
    monkeypatch.setattr(
        adapter,
        "renderer",
        lambda: SimpleNamespace(main=lambda argv: received.append(("render", argv))),
    )
    previous = sys.argv
    cli.main(["export-usd", "--help"])
    cli.main(["render", "--help"])
    assert received == [("usd", ["--help"]), ("render", ["--help"])]
    assert sys.argv is previous


def test_doctor_reports_missing_optional_packages_without_importing_them(
    monkeypatch, capsys
):
    def absent(name):
        raise cli.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cli.metadata, "version", absent)
    cli.main(["doctor"])
    report = json.loads(capsys.readouterr().out)
    assert report["myosuite-soma"] == __version__
    assert report["myosuite"] == report["torch"] == "not installed"
    with pytest.raises(SystemExit) as stopped:
        cli.main(["doctor", "unexpected"])
    assert stopped.value.code == 2


def test_missing_optional_renderer_is_an_actionable_cli_error(monkeypatch, capsys):
    from myosuite_soma import adapter

    def absent():
        raise ImportError("MyoSuite is optional; install the adapter")

    monkeypatch.setattr(adapter, "renderer", absent)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["render", "--help"])
    assert stopped.value.code == 2
    assert "MyoSuite is optional" in capsys.readouterr().err
