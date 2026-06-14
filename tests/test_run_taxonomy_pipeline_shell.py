from pathlib import Path


def test_phase_retry_preserves_failing_command_exit_code():
    script = Path("scripts/run_taxonomy_pipeline.sh").read_text()

    assert "if \"${PHASE_CMD[@]}\"; then" in script
    assert "else\n      local exit_code=$?" in script
    assert "return \"$exit_code\"" in script
