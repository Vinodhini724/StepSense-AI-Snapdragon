"""Dependency-light verification for the StepSense sequence engine."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.io import load_attempt, load_procedure
from core.sequence_engine import evaluate_attempt
_, steps = load_procedure(ROOT / "data" / "procedures" / "lab_safety.json")

expected_status = {
    "correct.json": "Completed",
    "missed_step.json": "Incomplete",
    "wrong_order.json": "Review required",
    "repeated_step.json": "Review required",
}

for filename, status in expected_status.items():
    _, observations = load_attempt(ROOT / "data" / "demo_attempts" / filename)
    result = evaluate_attempt(steps, observations)
    assert result["status"] == status, (filename, result)
    print(f"PASS {filename}: {result['status']} ({result['overall_score']}%)")
