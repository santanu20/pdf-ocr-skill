"""Transformation Ledger — auditable diff log for every normalization.

Per architecture proposal item #11 (2026-08-12):
  "Every normalization should record: {operation, input, output, rule,
   confidence}. This gives you reproducibility — you can reconstruct
   exactly how the final document was produced."

Design:
  - Per-document ledger (singleton per skill invocation)
  - Decorator `@ledger_op(rule_id, confidence)` for any text-transform fn
  - Manual `record()` for inline transformations (e.g. regex subs)
  - Sidecar `.ledger.jsonl` next to output (one entry per transformation)
  - Backward-compat: if disabled, decorators are no-ops

Each entry shape:
  {
    "ts": 1723445600.123,
    "page": 7,
    "op": "fix_latex_math_output",
    "rule_id": "LATEX_CMD_CONFUSION",
    "input": "\\sort{x}",
    "output": "\\sqrt{x}",
    "confidence": 0.97,
    "context_before": "...",
    "context_after": "..."
  }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from functools import wraps
from pathlib import Path
from typing import Callable, Optional
import logging

log = logging.getLogger(__name__)


_LEDGER: Optional["Ledger"] = None


@dataclass
class LedgerEntry:
    ts: float
    op: str
    rule_id: str
    input_snippet: str
    output_snippet: str
    confidence: float
    page: Optional[int] = None
    context_before: str = ""
    context_after: str = ""


class Ledger:
    """Per-document transformation ledger.

    Use as singleton via get_ledger(). Reset per document via reset().
    """

    def __init__(self, output_path: Optional[Path] = None):
        self._entries: list[LedgerEntry] = []
        self._output_path = output_path
        self._current_page: Optional[int] = None

    def set_page(self, page_num: Optional[int]) -> None:
        """Set current page context for subsequent entries."""
        self._current_page = page_num

    def set_output_path(self, path: Path) -> None:
        self._output_path = path

    def record(
        self,
        op: str,
        rule_id: str,
        input_val: str,
        output_val: str,
        confidence: float = 1.0,
        page: Optional[int] = None,
        context: str = "",
    ) -> None:
        """Record a transformation. Strings clipped to 200 chars for compactness."""
        if input_val == output_val:
            return  # no-op — skip
        snippet_in = (
            input_val[:200] if isinstance(input_val, str) else str(input_val)[:200]
        )
        snippet_out = (
            output_val[:200] if isinstance(output_val, str) else str(output_val)[:200]
        )
        entry = LedgerEntry(
            ts=time.time(),
            op=op,
            rule_id=rule_id,
            input_snippet=snippet_in,
            output_snippet=snippet_out,
            confidence=confidence,
            page=page if page is not None else self._current_page,
            context_before=context[:80] if context else "",
            context_after="",  # filled by decorator version if needed
        )
        self._entries.append(entry)

    def flush(self) -> None:
        """Write all entries to sidecar JSONL if output_path set."""
        if not self._output_path or not self._entries:
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def by_op(self) -> dict:
        out = {}
        for e in self._entries:
            out[e.op] = out.get(e.op, 0) + 1
        return out


def get_ledger() -> Ledger:
    """Singleton ledger. Created on first access."""
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = Ledger()
    return _LEDGER


def reset_ledger(output_path: Optional[Path] = None) -> Ledger:
    """Reset and optionally set output path. Call once per document."""
    global _LEDGER
    _LEDGER = Ledger(output_path)
    return _LEDGER


def ledger_op(
    rule_id: str,
    confidence: float = 1.0,
    op_name: Optional[str] = None,
) -> Callable:
    """Decorator: log input→output of a text-transform function.

    The wrapped function MUST take `text: str` as first arg and return `str`.
    Records only if output != input (skips no-ops).

    Usage:
        @ledger_op("LATEX.CMD_CONFUSION", 0.95)
        def fix_latex_math_output(text: str) -> str:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        actual_op = op_name or fn.__name__

        @wraps(fn)
        def wrapper(text, *args, **kwargs):
            out = fn(text, *args, **kwargs)
            try:
                get_ledger().record(
                    op=actual_op,
                    rule_id=rule_id,
                    input_val=text,
                    output_val=out,
                    confidence=confidence,
                )
            except Exception as e:
                log.debug("ledger record failed (transform unaffected): %s", e)
            return out

        return wrapper

    return decorator


def manual_record(
    op: str,
    rule_id: str,
    input_val: str,
    output_val: str,
    confidence: float = 1.0,
) -> None:
    """For inline transformations (e.g. regex subs inside larger fn).

    Usage:
        before = text
        text = re.sub(r"\\\\sort", r"\\\\sqrt", text)
        manual_record("fix_latex", "CMD_CONFUSION", before, text, 0.97)
    """
    try:
        get_ledger().record(op, rule_id, input_val, output_val, confidence)
    except Exception as e:
        log.debug("manual_record: Exception degraded: %s", e)


__all__ = [
    "Ledger",
    "LedgerEntry",
    "get_ledger",
    "reset_ledger",
    "ledger_op",
    "manual_record",
]
