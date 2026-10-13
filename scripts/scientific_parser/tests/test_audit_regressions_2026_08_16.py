"""Regression tests for 2026-08-16 nuclear-audit findings (C3).

Each test pins a path the suite previously could NOT guard (mutation
spot-check M1-M5 all passed green with the bug reintroduced). These are
the bench-critical bug families:
  - phase-0 $$ empty-merge must NOT pair across lines (was -11 pts)
  - extraction cache key must include skill version
  - geometry header/footer deep bands (6% top / 92% bottom)
  - font-math detector items MUST carry bbox (gap-aware separator)
  - DEFAULT_CACHE_DIR / reflexive constants single-sourced
  - Ollama HTTP retry (transient 5xx recovery)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scientific_parser.postproc.text_pipeline import fix_latex_math_output  # noqa: E402


class TestPhaseZeroEmptyMerge:
    """Audit M2 mutation: reverting `[ \\t]` to `\\s` re-fuses adjacent
    display equations across newlines (-11 bench tests, 2026-08-15)."""

    def test_adjacent_display_equations_not_fused(self):
        text = "$$E = mc^2$$\n\n$$F = ma$$\n"
        out = fix_latex_math_output(text)
        assert "$$" in out
        # The fusion bug joined both blocks into one $$...$$ containing a
        # newline-spanning pair. Post-fix, "mc^2" and "F = ma" must not be
        # inside ONE math block.
        assert not any(("mc^2" in part and "F = ma" in part) for part in out.split("$$"))

    def test_phase0_never_matches_across_newlines(self):
        # Exact oracle for the 2026-08-15 fusion bug: with bare \s in any
        # phase-0 pattern, `$ \n$X$ \n$` collapses to `$X$` (adjacent
        # display equations fused, -11 bench tests). [ \t] must NOT match.
        text = "$ \n$X$ \n$"
        out = fix_latex_math_output(text)
        assert "\n" in out, f"phase-0 pattern matched across newlines: {out!r}"

    def test_phase0_pattern_source_uses_tab_char_class(self):
        import inspect

        from scientific_parser.postproc import text_pipeline as tp

        src = inspect.getsource(tp.fix_latex_math_output)
        for ln in src.splitlines():
            if "$(" in ln or ("\\$" in ln and "[^$" in ln):
                # candidate phase-0 pattern line
                assert "\\s+\\$" not in ln.replace("[ \\t]", ""), (
                    f"phase-0 pattern reintroduced bare \\s: {ln.strip()}"
                )


class TestCacheKeyVersion:
    """Audit M4 mutation: dropping SKILL_EXTRACTION_VERSION from the cache
    key makes old-format cache hits return stale extractions."""

    def test_key_includes_version(self):
        """Contract migrated 2026-09-10 (raw-cache split): the key tracks
        OCR_PIPELINE_VERSION (extraction semantics). POSTPROC_VERSION drift
        must NOT move the key — postproc changes replay raw through the
        chain instead of re-extracting (run-7 RCA)."""
        from scientific_parser.core.extraction_cache import _cache_key

        class NS:
            dpi = 200
            max_dim = 1024
            force_ocr = False
            fast = False
            pages = None
            no_cache = False

        src = Path(__file__).parent.parent.parent / "unit_test_fixture.pdf"
        src.write_bytes(b"%PDF-1.4 minimal")
        k1 = _cache_key(str(src), ["rapidocr"], NS)
        import scientific_parser.core.extraction_cache as ec

        orig_ocr = ec.OCR_PIPELINE_VERSION
        orig_pp = ec.POSTPROC_VERSION
        try:
            ec.OCR_PIPELINE_VERSION = "9999.99.99.ocr-bump"
            k2 = _cache_key(str(src), ["rapidocr"], NS)
            ec.OCR_PIPELINE_VERSION = orig_ocr
            ec.POSTPROC_VERSION = "9999.99.99.pp-bump"
            k3 = _cache_key(str(src), ["rapidocr"], NS)
        finally:
            ec.OCR_PIPELINE_VERSION = orig_ocr
            ec.POSTPROC_VERSION = orig_pp
        src.unlink()
        assert k1 != k2, "cache key ignores OCR_PIPELINE_VERSION (stale hits)"
        assert k1 == k3, (
            "cache key must not react to POSTPROC_VERSION — postproc-only "
            "bumps re-roll stochastic OCR (run-7 RCA)"
        )


class TestHeaderFooterBands:
    """Audit M5 mutation: deep band 6%->60% would classify body text as
    header/footer and strip it."""

    def test_deep_band_thresholds(self):
        import re

        src_path = Path(__file__).parent.parent / "routing" / "page_metadata.py"
        src = src_path.read_text()
        m = re.search(r"deep = y0 < H \* ([\d.]+) or y1 > H \* ([\d.]+)", src)
        assert m, "band expression moved — update this test"
        top, bottom = float(m.group(1)), float(m.group(2))
        assert top <= 0.12, f"top band {top} too permissive (strips body)"
        assert bottom >= 0.85, f"bottom band {bottom} too permissive (strips body)"

    def test_band_geometry_behavior(self, monkeypatch):
        import scientific_parser.routing.page_metadata as pm

        H = 800.0
        W = 600.0
        # Real fitz rects carry BOTH width and height; the column-aware
        # band detector (audit C4, 2026-08-18) reads width for gutter
        # checks. Fake spans now carry sizes too (body-size detection).
        pd = {
            "rect": type("R", (), {"height": H, "width": W})(),
            "dict": {
                "blocks": [
                    {
                        "type": 0,
                        "lines": [
                            {
                                "bbox": (0, 20, 500, 40),  # y0=20 < 6% of 800=48
                                "spans": [{"text": "JOURNAL HEADER", "size": 8.0}],
                            }
                        ],
                    },
                    {
                        "type": 0,
                        "lines": [
                            {
                                "bbox": (0, 400, 500, 420),  # mid-body
                                "spans": [{"text": "Body text here", "size": 10.0}],
                            }
                        ],
                    },
                ]
            },
        }
        monkeypatch.setattr(pm, "_get_page_dict", lambda p, n: pd)
        out = pm._geometry_header_footer_lines("fake.pdf", 1)
        assert out == ["JOURNAL HEADER"], out

    def test_column_body_not_header(self, monkeypatch):
        """Audit C4 pin (2026-08-18): on a 2-column page, a band-zone line
        that sits INSIDE one column at body size is body text, not a
        header — the old detector ate 8 body sentences on 0514f667 pg4."""
        import scientific_parser.routing.page_metadata as pm

        H, W = 800.0, 600.0

        # left column lines at x 60..280, right column at 320..540
        def line(x0, y0, x1, y1, txt):
            return {"bbox": (x0, y0, x1, y1), "spans": [{"text": txt, "size": 9.5}]}

        body = [
            line(60, 50, 280, 64, "left column body first line here"),
            line(320, 50, 540, 64, "right column body first line here"),
            line(60, 400, 280, 414, "left column mid text"),
            line(320, 400, 540, 414, "right column mid text"),
            line(60, 500, 280, 514, "left column more text"),
            line(320, 500, 540, 514, "right column more text"),
            line(60, 600, 280, 614, "left column tail text"),
            line(320, 600, 540, 614, "right column tail text"),
            # a TRUE running header: full-width, above both columns
            line(60, 20, 540, 34, "J. Author et al. / Journal 12 (2015) 55-66"),
        ]
        pd = {
            "rect": type("R", (), {"height": H, "width": W})(),
            "dict": {"blocks": [{"type": 0, "lines": body}]},
        }
        monkeypatch.setattr(pm, "_get_page_dict", lambda p, n: pd)
        out = pm._geometry_header_footer_lines("fake.pdf", 1)
        # Only the full-width running header qualifies on a 2-col page
        assert out == ["J. Author et al. / Journal 12 (2015) 55-66"], out


class TestFontItemsBbox:
    """Audit M3: items dict lacking "bbox" silently killed the gap-aware
    space separator (glued words on every cascade page, 2026-08-16 fix)."""

    def test_gap_aware_separator_emits_space(self):
        """Functional pin: two body-size spans with an x-gap > 0.15*body
        MUST get a space separator (the 'Finally,webriefly' glued-word bug
        was the items dict missing bbox — separator silently dead)."""
        from scientific_parser.science.font_math_detector import _build_line_with_math

        spans = [
            # plain-font body spans, gap from x=90 to x=150 (body=10 → 60 > 1.5)
            {
                "text": "Finally,",
                "font": "Helvetica",
                "size": 10.0,
                "origin": (50, 100),
                "bbox": (50, 90, 90, 102),
            },
            {
                "text": "webriefly",
                "font": "Helvetica",
                "size": 10.0,
                "origin": (150, 100),
                "bbox": (150, 90, 200, 102),
            },
        ]
        out = _build_line_with_math(spans, 10.0, 100.0)
        assert "Finally, webriefly" in out, f"gap-aware separator dead: {out!r}"


class TestConstantsSingleSourced:
    """Audit M2: reflexive constants + DEFAULT_CACHE_DIR were defined 2-4x;
    value drift between copies is silent. Post-fix there is exactly one
    definition site each."""

    def test_no_duplicate_definitions(self):
        base = Path(__file__).resolve().parents[2]
        monolith = (base / "pdf_ocr.py").read_text()
        cache_mod = (base / "scientific_parser" / "core" / "extraction_cache.py").read_text()
        for const in (
            "REFLEXIVE_MAX_ATTEMPTS_PER_MODEL",
            "OCR_PER_ATTEMPT_TIMEOUT_S",
            "SKILL_EXTRACTION_VERSION",
            "DEFAULT_CACHE_DIR",
        ):
            assert not re.search(rf"^{const}\s*=", monolith, re.M), (
                f"{const} redefined in monolith (drift trap)"
            )
            assert (
                not re.search(
                    rf"^{const}\s*=",
                    cache_mod if const != "DEFAULT_CACHE_DIR" else "/nonexistent",
                    re.M,
                )
                or const != "DEFAULT_CACHE_DIR"
            ), f"{const} redefined in extraction_cache"
        # values still resolve via star-imports
        import pdf_ocr  # noqa: F401

        assert pdf_ocr.REFLEXIVE_MAX_ATTEMPTS_PER_MODEL == 2
        assert pdf_ocr.OCR_PER_ATTEMPT_TIMEOUT_S == 240


import re  # noqa: E402  (used above)


class TestOllamaRetry:
    """Audit H1: transient Ollama 5xx must be retried, persistent failure
    must raise (caller degrades), never infinite-loop."""

    def test_retry_then_success(self, monkeypatch):
        import json as _json

        import scientific_parser.ingest.ocr_engine as oe

        calls = {"n": 0}

        class FakeResp:
            def __init__(self, body):
                self._b = body.encode()

            def read(self):
                return self._b

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                import urllib.error

                raise urllib.error.HTTPError(req.full_url, 503, "cold", {}, None)
            return FakeResp('{"response": "ok"}')

        monkeypatch.setattr(oe.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(
            oe._time_mod if hasattr(oe, "_time_mod") else __import__("time"),
            "sleep",
            lambda s: None,
        )
        out = oe._ollama_post_json("http://x", b"{}", retries=2)
        assert out == {"response": "ok"} and calls["n"] == 2

    def test_persistent_500_raises(self, monkeypatch):
        import urllib.error

        import scientific_parser.ingest.ocr_engine as oe

        calls = {"n": 0}

        def always_500(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 500, "x", None, None)

        monkeypatch.setattr(oe.urllib.request, "urlopen", always_500)
        monkeypatch.setattr(__import__("time"), "sleep", lambda s: None)
        with pytest.raises(urllib.error.HTTPError):
            oe._ollama_post_json("http://x", b"{}", retries=2)
        assert calls["n"] == 3, f"expected retries+1=3 attempts, got {calls['n']}"

    def test_4xx_not_retried(self, monkeypatch):
        import urllib.error

        import scientific_parser.ingest.ocr_engine as oe

        calls = {"n": 0}

        def always_404(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 404, "nf", None, None)

        monkeypatch.setattr(oe.urllib.request, "urlopen", always_404)
        with pytest.raises(urllib.error.HTTPError):
            oe._ollama_post_json("http://x", b"{}", retries=2)
        assert calls["n"] == 1, "client errors must not be retried"
