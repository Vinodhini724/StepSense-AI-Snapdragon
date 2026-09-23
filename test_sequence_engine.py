from core.sequence_engine import Observation, Step, evaluate_attempt


STEPS = [Step(str(i), f"Step {i}") for i in range(1, 6)]


def obs(order):
    return [Observation(value, i * 5, (i + 1) * 5, 0.9) for i, value in enumerate(order)]


def test_correct_attempt():
    result = evaluate_attempt(STEPS, obs(["1", "2", "3", "4", "5"]))
    assert result["status"] == "Completed"
    assert result["overall_score"] == 100


def test_missing_step():
    result = evaluate_attempt(STEPS, obs(["1", "2", "4", "5"]))
    assert result["missing"] == ["3"]
    assert result["status"] == "Incomplete"


def test_wrong_order():
    result = evaluate_attempt(STEPS, obs(["1", "3", "2", "4", "5"]))
    assert result["sequence_score"] < 100
    assert result["out_of_order"]


def test_repeated_step():
    result = evaluate_attempt(STEPS, obs(["1", "2", "3", "3", "4", "5"]))
    assert result["repeated"] == ["3"]
    assert result["status"] == "Review required"

