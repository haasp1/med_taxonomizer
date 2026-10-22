import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE_SCRIPTS = sorted((REPO_ROOT / "scripts").glob("phase_*.py"))


@pytest.mark.parametrize("script", PHASE_SCRIPTS, ids=lambda p: p.name)
def test_phase_script_imports_when_run_directly(script):
    """Phase scripts must resolve `lib` when run as `python scripts/<phase>.py`."""
    result = subprocess.run(
        [sys.executable, str(script.relative_to(REPO_ROOT)), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script",
    ["phase_00_meta_discovery.py", "phase_02_category_discovery.py", "phase_04_labeling.py",
     "phase_04_labeling_cardio_indication.py"],
)
def test_text_reading_phases_accept_input(script):
    source = (REPO_ROOT / "scripts" / script).read_text()
    assert 'parser.add_argument("--input"' in source
    assert "experiments/01_data_preparation" not in source


def test_runner_passes_input_to_text_reading_phases():
    runner = (REPO_ROOT / "scripts" / "run_taxonomy_pipeline.sh").read_text()
    for phase in ("phase_00_meta_discovery.py", "phase_02_category_discovery.py", "phase_04_labeling.py"):
        block = runner.split(phase, 1)[1].split(")", 1)[0]
        assert '--input "$INPUT_DATA"' in block, phase
