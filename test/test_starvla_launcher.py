"""Check the launcher with a fake Python executable; no model server is started."""

import json
import os
from pathlib import Path
import subprocess
import sys


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/run_starvla_server.sh"


def test_launcher_help_and_missing_checkpoint(tmp_path):
    syntax = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(["bash", str(LAUNCHER), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "CHECKPOINT" in help_result.stdout
    missing = subprocess.run(
        ["bash", str(LAUNCHER), str(tmp_path / "missing.pt")], capture_output=True, text=True,
    )
    assert missing.returncode != 0


def test_launcher_resolves_relative_path_and_preserves_options(tmp_path):
    checkpoint = tmp_path / "weights with spaces.pt"
    checkpoint.touch()
    python = tmp_path / "fake env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'env': {k: os.environ.get(k) for k in "
        "('STARVLA_ROOT', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', "
        "'PYTHONPATH', 'LD_LIBRARY_PATH')}}))\n"
    )
    python.chmod(0o755)
    env = dict(os.environ, STARVLA_PYTHON=str(python), STARVLA_ROOT="/test/starvla source",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="3",
               PYTHONPATH="/test/existing", LD_LIBRARY_PATH="")
    result = subprocess.run(
        ["bash", str(LAUNCHER), checkpoint.name, "--debug"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["args"] == [
        "-u", "run_server.py", "--model_type", "starvla", "--model_path", str(checkpoint), "--debug",
    ]
    assert output["cwd"] == str(LAUNCHER.parents[1])
    assert output["env"]["STARVLA_ROOT"] == "/test/starvla source"
    assert output["env"]["OMP_NUM_THREADS"] == "2"
    assert output["env"]["OPENBLAS_NUM_THREADS"] == "3"
    assert output["env"]["PYTHONPATH"] == f"{LAUNCHER.parents[1]}:/test/starvla source:/test/existing"
