"""Smoke-test the example scripts so they never silently rot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_quickstart_runs(tmp_path: Path, capsys: object) -> None:
    import runpy

    cwd = os.getcwd()
    os.chdir(tmp_path)  # write logs/ into the temp dir, not the repo
    try:
        runpy.run_path(str(_EXAMPLES / "quickstart.py"), run_name="__main__")
    finally:
        os.chdir(cwd)
    assert (tmp_path / "logs").exists()


def test_cubepick_case_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import runpy

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        runpy.run_path(str(_EXAMPLES / "cubepick_case.py"), run_name="__main__")
    finally:
        os.chdir(cwd)
    output = capsys.readouterr().out
    assert "status:  success" in output
    assert "scenes:  5   trials: 10" in output
    assert "success_at_end: 1" in output
    assert (tmp_path / "logs" / "cubepick-case").exists()


def test_visualize_eval_log_writes_one_page_dashboard(tmp_path: Path) -> None:
    import runpy

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        runpy.run_path(str(_EXAMPLES / "cubepick_case.py"), run_name="__main__")
    finally:
        os.chdir(cwd)

    log_path = next((tmp_path / "logs" / "cubepick-case").glob("*.json"))
    output_path = tmp_path / "charts.html"
    result = subprocess.run(
        [
            sys.executable,
            str(_EXAMPLES / "visualize_eval_log.py"),
            str(log_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = output_path.read_text(encoding="utf-8")
    assert f"wrote {output_path}" in result.stdout
    assert report.count("<html") == 1
    assert report.count("<svg") == 3
    assert "Success rate" in report
    assert "layout-0" in report
