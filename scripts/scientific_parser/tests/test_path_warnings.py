"""path_warnings contracts — collector behavior (unit).

Reconstruction check (2026-08-18): the module was rebuilt from call sites
after a harness reset; these tests pin the API shape the callers rely on.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PDF_OCR_NO_SKILL_VENV"] = "1"

from scientific_parser.core import path_warnings as pw


def setup_function():
    pw.reset_path_warnings()


def test_record_and_get_roundtrip():
    pw.record_path_warning(pw.VL_FALLBACK_USED, "vl used", page=3)
    recs = pw.get_path_warnings()
    assert len(recs) == 1
    assert recs[0]["kind"] == pw.VL_FALLBACK_USED
    assert recs[0]["page"] == 3


def test_summary_empty_when_clean():
    pw.reset_path_warnings()
    assert pw.format_path_warnings_summary() == ""


def test_cache_replay_format():
    out = pw.format_path_warning_records(
        [{"kind": "pdftotext_unavailable", "message": "gone", "page": 1}]
    )
    assert "pdftotext_unavailable" in out and "p1" in out


def test_born_digital_flag():
    pw.warn_born_digital_ocr_routes(
        {1: ("fast", []), 2: ("ocr", ["x"])}, {"born_digital": True}, is_image=False
    )
    recs = pw.get_path_warnings()
    assert recs and recs[0]["pages"] == [2]


# ── Gap-wiring coverage (2026-08-18 audit — every remaining silent site) ── #


def test_gap_codes_defined():
    for code in (
        "TABLE_MODEL_FALLBACK",
        "STAGE_SKIPPED",
        "PAGE_COUNT_FALLBACK",
        "PAGE_RENDER_FALLBACK",
        "SCRIPT_EXTRACTION_FALLBACK",
        "OCR_PROMPT_FALLBACK",
        "SPECIALIST_PROVIDERS_MISSING",
        "CLI_CONFIG_IGNORED",
        "FONT_MATH_CASCADE_FAILED",
        "OCR_ENGINE_LEGACY_API",
    ):
        assert isinstance(getattr(pw, code, None), str), code


def test_ocr_engine_legacy_api_records(monkeypatch):
    from scientific_parser.ingest import ocr_engine as oe

    # v1.4.4 tuple shape triggers the legacy-API warning
    out = oe._rapidocr_output_to_result(
        ([((0, 0, 1, 1), "hello", 0.9), ((0, 1, 2, 2), "world", 0.8)], None, None)
    )
    assert out[0] == "hello\nworld"
    recs = pw.get_path_warnings()
    assert any(r["kind"] == "ocr_engine_legacy_api" for r in recs)


# ── Registry consistency guard (2026-08-19, iter-57b root-cause pin) ────── #
# Incident class: a WIP edit REPLACED live constants instead of inserting
# (FIGURE_TABLE_REROUTE ate OCR_ENGINE_LEGACY_API + EQUATION_MODEL_FALLBACK),
# and 7 older codes were never added to __all__. Deferred function-local
# imports let the corruption ship silently until the rare path fired.
# These guards make every drift direction fail at test time.


def test_registry_all_matches_defined_constants():
    """__all__ must contain EXACTLY the UPPER_SNAKE code constants."""
    import re

    src = open(pw.__file__, encoding="utf-8").read()
    consts = set(re.findall(r"^([A-Z][A-Z0-9_]+)\s*=\s*\"[a-z0-9_]+\"", src, re.M))
    exported_upper = {n for n in pw.__all__ if n.isupper()}
    assert exported_upper == consts, (
        f"registry drift: missing-from-__all__={consts - exported_upper} "
        f"extra-in-__all__={exported_upper - consts}"
    )


def test_every_repo_import_of_codes_resolves():
    """Every name imported from path_warnings anywhere in scripts/ must
    exist in the module — a deleted constant must fail HERE, not on some
    rare runtime path."""
    import glob
    import re

    bad = []
    for f in glob.glob(
        os.path.join(os.path.dirname(pw.__file__), "..", "**", "*.py"),
        recursive=True,
    ):
        f = os.path.abspath(f)
        if f == os.path.abspath(pw.__file__):
            continue
        text = open(f, encoding="utf-8", errors="ignore").read()
        for m in re.finditer(
            r"from scientific_parser\.core\.path_warnings import \(([^)]*)\)",
            text,
        ):
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1)):
                if not hasattr(pw, name):
                    bad.append(f"{os.path.basename(f)}: {name}")
    assert not bad, f"unresolvable path_warnings imports: {bad}"


def test_no_deferred_path_warnings_imports():
    """All path_warnings imports must be top-level — a missing constant
    then fails at module import (instant), not when a rare fallback path
    first executes (silent shipping)."""
    import glob

    for f in glob.glob(
        os.path.join(os.path.dirname(pw.__file__), "..", "**", "*.py"),
        recursive=True,
    ):
        if os.path.basename(f).startswith("test_"):
            continue  # tests legitimately quote the pattern textually
        for i, line in enumerate(open(f, encoding="utf-8", errors="ignore"), 1):
            if line.startswith((" ", "\t")) and "path_warnings import" in line:
                raise AssertionError(
                    f"deferred import at {os.path.basename(f)}:{i} — hoist to module top"
                )


def test_skill_md_documents_every_code():
    """SKILL.md §Path Warnings must list every code value (doc sync)."""
    import re

    skill = open(
        os.path.join(os.path.dirname(pw.__file__), "..", "..", "..", "SKILL.md"),
        encoding="utf-8",
        errors="ignore",
    ).read()
    m = re.search(r"## Path[^\n]*\n(.*?)(?=\n## |\Z)", skill, re.S)
    assert m, "SKILL.md Path Warnings section missing"
    doc_values = set(re.findall(r"`?([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`?", m.group(1)))
    src = open(pw.__file__, encoding="utf-8").read()
    values = set(re.findall(r"^[A-Z][A-Z0-9_]+\s*=\s*\"([a-z0-9_]+)\"", src, re.M))
    missing = values - doc_values
    assert not missing, f"codes undocumented in SKILL.md: {sorted(missing)}"


def test_page_render_fallback_records(monkeypatch, tmp_path):
    from scientific_parser.ingest import ocr_engine as oe

    # fitz render blows up → pdftoppm path taken → warning recorded
    import types

    monkeypatch.setitem(sys.modules, "fitz", None)  # force ImportError
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        (tmp_path / "page-1.png").write_bytes(b"png")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(oe.subprocess, "run", fake_run)
    out = oe.pdf_page_to_png(str(tmp_path / "x.pdf"), 1, str(tmp_path / "page"))
    assert out.endswith("page-1.png")
    recs = pw.get_path_warnings()
    assert any(r["kind"] == "page_render_fallback" for r in recs)


def test_font_analysis_index_mismatch_records():
    from scientific_parser.ingest import font_analysis as fa

    class _TP:  # textpage whose ranges disagree with count
        def count_chars(self):
            return 5

        def get_text_range(self):
            return "hi"

    class _Page:
        def get_textpage(self):
            return _TP()

    class _Doc:
        def get_page(self, i):
            return _Page()

        def close(self):
            pass

    monkey_docs = {"_pdfium": types.SimpleNamespace(PdfDocument=lambda p: _Doc())}
    # PERF-1: _pdfium is no longer a wildcard-bound font_analysis global;
    # _extract_scripts_pypdfium reads it from shared (PEP 562 lazy) —
    # inject there so the call-time attr access sees the fake.
    import shared as _shared_mod

    saved = {k: getattr(_shared_mod, k) for k in monkey_docs}
    for k, v in monkey_docs.items():
        setattr(_shared_mod, k, v)
    try:
        out = fa._extract_scripts_pypdfium("fake.pdf", 1)
        assert out == "hi"
        recs = pw.get_path_warnings()
        assert any(r["kind"] == "script_extraction_fallback" for r in recs)
    finally:
        for k, v in saved.items():
            setattr(_shared_mod, k, v)


import types  # noqa: E402  (used by monkeypatch tests above)
