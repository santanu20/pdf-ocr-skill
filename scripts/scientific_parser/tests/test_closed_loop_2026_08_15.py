"""Tests for 2026-08-15 closed-loop + gate additions.

Covers:
  - _symbol_multiset / _symbols_agree (equation repair symbol gate)
  - looks_like_chat_response (VL chat-mode detector)
  - born-digital router gating (signals 7b/8 suppressed)
  - retry executor SPECIALIST_MODEL / COLUMN_RECONSTRUCT closure
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scientific_parser.science.equation_cascade import (  # noqa: E402
    _symbol_multiset,
    _symbols_agree,
)
from scientific_parser.core.confidence_helpers import (  # noqa: E402
    _compute_page_confidence,
    looks_like_chat_response,
)


class TestSymbolMultiset:
    def test_unicode_and_latex_both_counted(self):
        # ∑ as unicode + \alpha as command → 2 symbol kinds
        ms = _symbol_multiset(r"\sum_{i=1}^{n} \alpha_i + ∫")
        assert "∑" in ms and "α" in ms and "∫" in ms

    def test_command_word_boundary(self):
        # \epsilon ≠ \varepsilon (distinct glyphs)
        ms_eps = _symbol_multiset(r"\epsilon")
        ms_vareps = _symbol_multiset(r"\varepsilon")
        assert set(ms_eps) == {"ϵ"}
        assert set(ms_vareps) == {"ε"}

    def test_prose_has_no_symbols(self):
        assert not _symbol_multiset("the quick brown fox 12345")


class TestSymbolsAgree:
    def test_identical_passes(self):
        c = r"\sum_{i} \alpha_i \geq \beta"
        assert _symbols_agree(c, c.replace("i", "j"))

    def test_sum_to_prod_swap_rejected(self):
        # alnum retention would pass; symbol gate must catch ∑↔∏
        c = r"\sum_{i=1}^{n} x_i^2 + y"
        r = r"\prod_{i=1}^{n} x_i^2 + y"
        assert not _symbols_agree(c, r)

    def test_symbol_fix_allowed(self):
        # repair fixes ≤4 broken symbols
        c = r"\alpa + \beta \geq \gamma"
        r = r"\alpha + \beta \geq \gamma"
        assert _symbols_agree(c, r)

    def test_different_equation_rejected(self):
        c = r"\alpha \beta \gamma \delta \pi \sigma"
        r = r"\int \oint \nabla \forall \exists \oplus"
        assert not _symbols_agree(c, r)

    def test_no_symbols_vacuous_pass(self):
        assert _symbols_agree("abc", "xyz")


class TestChatModeDetector:
    def test_garbage_preambles(self):
        for s in (
            "The user wants me to extract the table",
            "I will format the data into a table",
            "Based on the image, this is a table of contents",
            "Sure, here is the transcription.",
            "I can see a list of names.",
        ):
            assert looks_like_chat_response(s), s

    def test_clean_transcription_passes(self):
        for s in (
            "Annual Wheat News Letter Vol. 49.",
            "Table 4. Top wheat cultivars planted in Kansas.",
            "$$E = mc^2$$",
        ):
            assert not looks_like_chat_response(s), s

    def test_confidence_penalty_drops_below_floor(self):
        garb = "The user wants me to extract the contact information. " * 5
        assert _compute_page_confidence("ocr", [], garb) < 0.55


class TestBornDigitalRouterGate:
    def _cls(self):
        from scientific_parser.routing.page_classifier import should_route_to_ocr

        return should_route_to_ocr

    def test_table_cells_one_per_line_suppressed_on_born_digital(self):
        route = self._cls()
        text = "\n".join(["Name", "Phone", "Room", "Smith J", "4115", "EA.G.13"] * 10)
        di_bd = {"born_digital": True, "pages_with_fonts": {1}, "invisible_pages": set()}
        r, why = route(text, doc_info=di_bd)
        assert not r, why
        # Same text WITHOUT doc_info (router must not assume born-digital):
        # layout signals may fire — the gate keys strictly on doc_info.
        r_naive, _ = route(text, doc_info=None)
        assert isinstance(r_naive, bool)

    def test_strong_signals_still_fire_on_born_digital(self):
        route = self._cls()
        garbled = "x" * 50 + "\ufffd" * 50  # U+FFFD flood
        di = {"born_digital": True, "pages_with_fonts": {1}, "invisible_pages": set()}
        r, why = route(garbled, doc_info=di)
        assert r and any("broken_encoding" in w for w in why)


class TestRetryClosure:
    def test_column_reconstruct_dispatches(self, tmp_path):
        from scientific_parser.verification.retry_actions import (
            RetryAction,
            RetryActionType,
        )
        from scientific_parser.verification.retry_executor import execute_action

        # Build a 2-block PDF via fitz so xy-cut has >=4 blocks... use a
        # real bench PDF page instead (fixture-free): skip if unavailable.
        bench_pdf = Path(__file__).resolve().parents[3] / "bench/bench_data/pdfs/multi_column"
        pdfs = sorted(bench_pdf.glob("*.pdf")) if bench_pdf.exists() else []
        if not pdfs:
            pytest.skip("no bench multi_column PDFs available")
        import fitz

        doc = fitz.open(pdfs[0])
        page = next((i for i in range(len(doc)) if len(doc[i].get_text().split()) > 80), 0)
        doc.close()
        action = RetryAction(
            action=RetryActionType.COLUMN_RECONSTRUCT,
            params={},
            rationale="test",
        )
        out = execute_action(action, {"pdf_path": str(pdfs[0]), "page_num": page + 1})
        # Outcome may be success (text) or graceful failure — never raise,
        # and on success must carry reordered text.
        assert out.success is False or (out.output_text or "").strip()

    def test_specialist_unknown_name_graceful(self, tmp_path):
        from scientific_parser.verification.retry_actions import (
            RetryAction,
            RetryActionType,
        )
        from scientific_parser.verification.retry_executor import execute_action
        from PIL import Image

        img = tmp_path / "p.png"
        Image.new("RGB", (100, 100), "white").save(img)
        action = RetryAction(
            action=RetryActionType.SPECIALIST_MODEL,
            params={"specialist": "nonexistent_specialist"},
            rationale="test",
        )
        out = execute_action(action, {"img_path": str(img), "region_bbox": (10, 10, 90, 90)})
        assert not out.success
        assert "unknown specialist" in out.message
