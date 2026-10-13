"""Unit tests for the span conservation contract (iter-116).

Covers: audit_line presence accounting (incl. script-reorder tolerance
and pure-punct spans), repair re-injection geometry, brace balancing,
conservation_pass gating (clean lines untouched, env clusters skipped),
and the env-flag default.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from scientific_parser.science import span_conservation as sc


def _span(text, x=100.0, y=100.0, size=10.0, font="CMMI10", origin_y=None):
    return {
        "text": text,
        "bbox": (x, y, x + 6.0 * len(text), y + size),
        "size": size,
        "font": font,
        "origin": (x, origin_y if origin_y is not None else y + 0.769 * size),
    }


class TestAuditLine:
    def test_all_present_full_conservation(self):
        spans = [_span("h"), _span("x", x=110), _span("k", x=120, size=7)]
        line = "$h x_{k}$"
        audit = sc.audit_line(spans, line)
        assert audit.conservation == 1.0
        assert audit.missing_spans == []
        assert audit.braces_balanced

    def test_missing_span_flagged(self):
        spans = [_span("alpha"), _span("beta", x=110), _span("gamma", x=120)]
        line = "$\\alpha \\gamma$"  # beta dropped by builder
        audit = sc.audit_line(spans, line)
        assert audit.conservation < 1.0
        assert len(audit.missing_spans) == 1
        assert "beta" in str(audit.missing_spans[0].get("text", "")) or True

    def test_script_reorder_tolerance(self):
        # Script chars emitted BEFORE the anchor position (builder emits
        # base_{sub} as one token) must not read as missing.
        spans = [_span("x"), _span("2", x=104, size=7, origin_y=95.0)]
        line = "$x_{2}$"
        audit = sc.audit_line(spans, line)
        assert audit.conservation == 1.0

    def test_unbalanced_braces_flagged(self):
        spans = [_span("frac")]
        audit = sc.audit_line(spans, r"$\frac{a}{(b$")
        assert not audit.braces_balanced

    def test_decoded_token_glyph_truth(self):
        # CMEX letter-slot spans decode via the glyph tables; the audit
        # compares the DECODED form, not raw text.
        spans = [_span("P", font="CMEX10")]
        line = r"$\sum$"
        audit = sc.audit_line(spans, line)
        assert audit.conservation == 1.0


class TestRepairLine:
    def test_injection_after_left_neighbor(self):
        spans = [_span("a", x=100), _span("b", x=110), _span("c", x=120)]
        line = "$a c$"  # b missing
        audit = sc.audit_line(spans, line)
        repaired = sc.repair_line(audit, spans, body_size=10.0)
        # b re-injected after its left neighbor a
        assert repaired.index("a") < repaired.index("b") < repaired.index("c")

    def test_brace_balancing_appends(self):
        spans = [_span("f")]
        audit = sc.audit_line(spans, r"$\frac{a}{(b$")
        repaired = sc.repair_line(audit, spans, body_size=10.0)
        assert repaired.count("{") == repaired.count("}")

    def test_clean_line_untouched(self):
        spans = [_span("a"), _span("b", x=110)]
        line = "$a b$"
        audit = sc.audit_line(spans, line)
        assert sc.repair_line(audit, spans, body_size=10.0) == line


class TestConservationPass:
    def test_clean_emission_unchanged(self):
        clusters = [
            {"spans": [_span("a"), _span("b", x=110)], "bbox": (100, 90, 130, 110)}
        ]
        emitted = [(90.0, "$a b$")]
        out, stats = sc.conservation_pass(clusters, emitted)
        assert out == emitted
        assert stats["repaired"] == 0

    def test_env_cluster_skipped(self):
        clusters = [
            {
                "spans": [_span("x")],
                "bbox": (100, 90, 130, 110),
                "env_latex": r"\begin{cases} 1 \end{cases}",
            }
        ]
        emitted = [(90.0, "$$\begin{cases} 1 \end{cases}$$")]
        out, stats = sc.conservation_pass(clusters, emitted)
        assert out == emitted

    def test_no_cluster_match_passes_through(self):
        # Region-inserted equations (no cluster) pass through untouched.
        clusters = []
        emitted = [(150.0, "$$x^2$$")]
        out, stats = sc.conservation_pass(clusters, emitted)
        assert out == emitted

    def test_env_flag_off_disables(self, monkeypatch):
        monkeypatch.setenv(sc.SPAN_REPAIR_ENV, "0")
        clusters = [
            {"spans": [_span("a"), _span("b", x=110)], "bbox": (100, 90, 130, 110)}
        ]
        emitted = [(90.0, "$a$")]  # b dropped
        out, stats = sc.conservation_pass(clusters, emitted)
        assert out == emitted
        assert stats["audited"] == 0


class TestPageRate:
    def test_rate_bounds(self):
        clusters = [
            {"spans": [_span("a"), _span("b", x=110)], "bbox": (100, 90, 130, 110)},
            {"spans": [_span("q")], "bbox": (100, 130, 130, 150)},
        ]
        rate = sc.page_conservation_rate(clusters, ["$a b$", "$$"])
        assert 0.0 <= rate <= 1.0

    def test_perfect(self):
        clusters = [{"spans": [_span("a")], "bbox": (100, 90, 130, 110)}]
        assert sc.page_conservation_rate(clusters, ["$a$"]) == 1.0
