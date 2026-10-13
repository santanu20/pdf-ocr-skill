"""Events — transformation pipeline event log.

Per proposal item #11: "Build a transformation ledger. Every normalization
should record: operation, input, output, rule, confidence."

This module is the EVENT-LEVEL log (one event per transformation pass).
Complements ledger.py (which is the ENTRY-LEVEL log per individual change).
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class EventLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class PipelineEvent:
    """One event in the transformation pipeline."""

    ts: float = field(default_factory=time.time)
    level: str = "INFO"
    page: Optional[int] = None
    op: str = ""  # function/transform name
    rule_id: str = ""  # rule category
    input_preview: str = ""
    output_preview: str = ""
    confidence: float = 1.0
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# In-memory event log (per process)
_EVENT_LOG: list[PipelineEvent] = []


def log_event(
    level: str,
    op: str,
    rule_id: str = "",
    input_val: Any = "",
    output_val: Any = "",
    confidence: float = 1.0,
    page: Optional[int] = None,
    duration_s: float = 0.0,
    metadata: Optional[dict] = None,
) -> PipelineEvent:
    """Log a pipeline event. Returns the event for chaining."""
    ev = PipelineEvent(
        level=level,
        page=page,
        op=op,
        rule_id=rule_id,
        input_preview=str(input_val)[:120] if input_val else "",
        output_preview=str(output_val)[:120] if output_val else "",
        confidence=confidence,
        duration_s=duration_s,
        metadata=metadata or {},
    )
    _EVENT_LOG.append(ev)
    return ev


def get_events() -> list[PipelineEvent]:
    return list(_EVENT_LOG)


def clear_events() -> None:
    _EVENT_LOG.clear()


def flush_events_jsonl(path) -> None:
    """Write all events to JSONL file."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for ev in _EVENT_LOG:
            f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")


def event_summary() -> dict:
    """Rollup: counts by op + level."""
    by_op = {}
    by_level = {}
    for ev in _EVENT_LOG:
        by_op[ev.op] = by_op.get(ev.op, 0) + 1
        by_level[ev.level] = by_level.get(ev.level, 0) + 1
    return {
        "total_events": len(_EVENT_LOG),
        "by_op": by_op,
        "by_level": by_level,
    }


__all__ = [
    "EventLevel",
    "PipelineEvent",
    "log_event",
    "get_events",
    "clear_events",
    "flush_events_jsonl",
    "event_summary",
]
