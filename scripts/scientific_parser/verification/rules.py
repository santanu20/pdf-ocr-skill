"""Rules-based verification — composable rule engine."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Callable


@dataclass
class RuleResult:
    rule_name: str
    passed: bool
    severity: str = "warn"  # info|warn|error
    message: str = ""


class Rule:
    """One verification rule."""

    def __init__(
        self, name: str, fn: Callable[[Any], tuple[bool, str]], severity: str = "warn"
    ):
        self.name = name
        self.fn = fn
        self.severity = severity

    def check(self, value: Any) -> RuleResult:
        try:
            ok, msg = self.fn(value)
            return RuleResult(self.name, ok, self.severity, msg)
        except Exception as e:
            return RuleResult(self.name, False, "error", f"rule crashed: {e}")


class RuleEngine:
    """Run a list of rules against a value. Aggregate results."""

    def __init__(self, rules: Optional[list] = None):
        self.rules: list[Rule] = rules or []

    def add(self, rule: Rule) -> None:
        self.rules.append(rule)

    def run(self, value: Any) -> list[RuleResult]:
        return [r.check(value) for r in self.rules]

    def run_aggregate(self, value: Any) -> dict:
        results = self.run(value)
        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "errors": sum(1 for r in results if r.severity == "error" and not r.passed),
            "warnings": sum(
                1 for r in results if r.severity == "warn" and not r.passed
            ),
            "results": [
                {
                    "rule": r.rule_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                }
                for r in results
            ],
        }


# === Built-in rules ===================================================== #


def rule_min_length(min_chars: int = 10) -> Rule:
    """Text must be at least N chars."""

    def fn(text):
        if not isinstance(text, str):
            return False, "not a string"
        return len(text) >= min_chars, f"length {len(text)} < {min_chars}"

    return Rule("min_length", fn, "warn")


def rule_no_repetition(threshold: float = 0.5) -> Rule:
    """Text must not have heavy character repetition."""
    import re

    def fn(text):
        if not isinstance(text, str) or not text:
            return True, ""
        long_runs = len(re.findall(r"(.)\1{9,}", text))
        return long_runs == 0, f"{long_runs} long char runs detected"

    return Rule("no_repetition", fn, "error")


def rule_no_cjk() -> Rule:
    """Text should not contain CJK characters (for English scientific docs)."""
    import re

    def fn(text):
        if not isinstance(text, str):
            return True, ""
        cjk = re.findall(r"[\u3400-\u9fff]", text)
        return len(cjk) == 0, f"{len(cjk)} CJK chars present"

    return Rule("no_cjk", fn, "error")


def rule_katex_valid() -> Rule:
    """All math blocks must render via pykatex."""
    import re

    def fn(text):
        if not isinstance(text, str) or "$" not in text:
            return True, ""
        try:
            import pykatex
        except ImportError:
            return True, ""
        blocks = re.findall(r"\$\$.+?\$\$", text, re.DOTALL)
        blocks += re.findall(r"(?<!\$)\$([^$\n]{1,200})\$(?!\$)", text)
        for b in blocks:
            body = re.sub(r"^\${1,2}|\${1,2}$", "", b.strip())
            try:
                pykatex.renderToString(body, strict="error")
            except Exception:
                return False, f"KaTeX parse fail: {body[:60]}"
        return True, ""

    return Rule("katex_valid", fn, "warn")


def default_rule_engine() -> RuleEngine:
    """Engine pre-loaded with common rules."""
    return RuleEngine(
        [
            rule_min_length(10),
            rule_no_repetition(),
            rule_no_cjk(),
            rule_katex_valid(),
        ]
    )


__all__ = [
    "RuleResult",
    "Rule",
    "RuleEngine",
    "rule_min_length",
    "rule_no_repetition",
    "rule_no_cjk",
    "rule_katex_valid",
    "default_rule_engine",
]
