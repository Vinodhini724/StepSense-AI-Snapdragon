from __future__ import annotations

import json
from pathlib import Path

from .sequence_engine import Observation, Step


def load_procedure(path: str | Path) -> tuple[dict, list[Step]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, [Step(**step) for step in payload["steps"]]


def load_attempt(path: str | Path) -> tuple[dict, list[Observation]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, [Observation(**item) for item in payload["observations"]]

