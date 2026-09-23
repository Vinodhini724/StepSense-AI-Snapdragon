from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class Step:
    id: str
    name: str
    prompt: str = ""
    max_duration_seconds: float | None = None


@dataclass(frozen=True)
class Observation:
    step_id: str
    start_seconds: float
    end_seconds: float
    confidence: float = 1.0


def _lcs(a: list[str], b: list[str]) -> list[str]:
    table = [[[] for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
    for i, left in enumerate(a, start=1):
        for j, right in enumerate(b, start=1):
            if left == right:
                table[i][j] = table[i - 1][j - 1] + [left]
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1], key=len)
    return table[-1][-1]


def evaluate_attempt(
    steps: Iterable[Step],
    observations: Iterable[Observation],
    confidence_threshold: float = 0.45,
) -> dict:
    expected_steps = list(steps)
    expected = [step.id for step in expected_steps]
    accepted_obs = [o for o in observations if o.confidence >= confidence_threshold]
    uncertain_obs = [o for o in observations if o.confidence < confidence_threshold]
    observed = [o.step_id for o in accepted_obs]

    expected_counts = Counter(expected)
    observed_counts = Counter(observed)
    missing = [sid for sid in expected if observed_counts[sid] < expected_counts[sid]]
    repeated = sorted([sid for sid, count in observed_counts.items() if count > expected_counts.get(sid, 0)])
    unexpected = sorted([sid for sid in observed_counts if sid not in expected_counts])

    canonical = []
    seen = set()
    for sid in observed:
        if sid in expected_counts and sid not in seen:
            canonical.append(sid)
            seen.add(sid)
    lcs = _lcs(expected, canonical)
    out_of_order = [sid for sid in canonical if sid not in lcs]

    durations = {o.step_id: round(o.end_seconds - o.start_seconds, 2) for o in accepted_obs}
    delayed = []
    for step in expected_steps:
        duration = durations.get(step.id)
        if step.max_duration_seconds and duration and duration > step.max_duration_seconds:
            delayed.append(step.id)

    completion_score = round(100 * (len(expected) - len(set(missing))) / max(1, len(expected)))
    sequence_score = round(100 * len(lcs) / max(1, len(expected)))
    penalty = 5 * len(repeated) + 5 * len(delayed) + 4 * len(uncertain_obs)
    overall_score = max(0, round(0.55 * completion_score + 0.45 * sequence_score - penalty))

    if missing:
        status = "Incomplete"
    elif out_of_order or repeated or unexpected:
        status = "Review required"
    else:
        status = "Completed"

    names = {step.id: step.name for step in expected_steps}
    timeline = []
    for index, obs in enumerate(accepted_obs, start=1):
        timeline.append({
            "order": index,
            "step_id": obs.step_id,
            "step_name": names.get(obs.step_id, obs.step_id.replace('_', ' ').title()),
            "start_seconds": obs.start_seconds,
            "end_seconds": obs.end_seconds,
            "duration_seconds": round(obs.end_seconds - obs.start_seconds, 2),
            "confidence": round(obs.confidence, 3),
        })

    return {
        "status": status,
        "overall_score": overall_score,
        "completion_score": completion_score,
        "sequence_score": sequence_score,
        "expected_order": expected,
        "observed_order": observed,
        "missing": missing,
        "repeated": repeated,
        "unexpected": unexpected,
        "out_of_order": out_of_order,
        "delayed": delayed,
        "uncertain_count": len(uncertain_obs),
        "timeline": timeline,
        "steps": [asdict(step) for step in expected_steps],
    }


def feedback_messages(result: dict) -> list[str]:
    names = {step["id"]: step["name"] for step in result["steps"]}
    messages = []
    if result["missing"]:
        messages.append("Missed: " + ", ".join(names.get(x, x) for x in result["missing"]))
    if result["out_of_order"]:
        messages.append("Wrong order: " + ", ".join(names.get(x, x) for x in result["out_of_order"]))
    if result["repeated"]:
        messages.append("Repeated: " + ", ".join(names.get(x, x) for x in result["repeated"]))
    if result["delayed"]:
        messages.append("Long duration: " + ", ".join(names.get(x, x) for x in result["delayed"]))
    if result["uncertain_count"]:
        messages.append(f'{result["uncertain_count"]} low-confidence event(s) require trainer review')
    if not messages:
        messages.append("All required steps were completed in the approved order.")
    return messages

