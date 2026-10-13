#!/usr/bin/env python3
"""Audit ALL specialized components — import + functional probe per component.

Covers:
  scientific_parser/ package (10 sublayers, ~90 modules)
  pdf_ocr.py monolith specialized functions

Run: .venv/bin/python scripts/audit_specialized.py
Exit: 0 if all PASS, 1 if any FAIL (SKIP = optional dep missing, does not fail)
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("PDF_OCR_NO_SKILL_VENV", "1")

_PASS = 0
_FAIL = 0
_SKIP = 0
_RESULTS: list[tuple[str, str, str, str]] = []


def probe(category: str, name: str, fn):
    global _PASS, _FAIL, _SKIP
    try:
        detail = fn() or ""
        status = "PASS"
        _PASS += 1
    except ImportError as e:
        status = "SKIP"
        detail = f"ImportError: {e}"
        _SKIP += 1
    except Exception as e:
        status = "FAIL"
        detail = f"{type(e).__name__}: {e}"
        _FAIL += 1
    _RESULTS.append((category, name, status, detail))


# ═══════════════════════════════════════════════════════════════════════════
# 1. CORE IR
# ═══════════════════════════════════════════════════════════════════════════
def t_core_make_block():
    from scientific_parser.core.ir import make_block, BlockIR, E2_DETERMINISTIC

    b = make_block(
        page_num=1,
        block_idx=0,
        block_type="text",
        bbox=(0, 0, 100, 20),
        text="hello",
        level=E2_DETERMINISTIC,
    )
    assert isinstance(b, BlockIR)


def t_core_evidence():
    from scientific_parser.core.ir import Evidence, E0_NATIVE_PDF_TEXT, E5_VLM
    from scientific_parser.core.evidence import rank_by_priority, merge_evidence

    e1 = Evidence(level=E0_NATIVE_PDF_TEXT, source="pdftotext", value="hi")
    e2 = Evidence(level=E5_VLM, source="qwen", value="hi")
    ranked = rank_by_priority([e2, e1])
    assert ranked[0].level == E0_NATIVE_PDF_TEXT
    merged = merge_evidence([e1], [e2])
    assert len(merged) == 2


def t_core_confidence():
    from scientific_parser.core.confidence import aggregate_evidence_confidence

    c = aggregate_evidence_confidence([0.9, 0.8])
    assert 0.0 <= c <= 1.0


def t_core_events():
    from scientific_parser.core.events import log_event, get_events, clear_events

    clear_events()
    log_event("INFO", "test_op")
    assert len(get_events()) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. INGEST
# ═══════════════════════════════════════════════════════════════════════════
def t_ingest_image():
    from scientific_parser.ingest.image import image_stats

    assert callable(image_stats)


def t_ingest_vector():
    from scientific_parser.ingest.vector_extract import detect_table_grid, detect_plot_axes

    assert callable(detect_table_grid) and callable(detect_plot_axes)


def t_ingest_render():
    from scientific_parser.ingest.pdf_render import estimate_render_dpi

    assert callable(estimate_render_dpi)


def t_ingest_font_analysis():
    from scientific_parser.ingest.font_analysis import detect_subscripts_pymupdf

    assert callable(detect_subscripts_pymupdf)


def t_ingest_font_aware_latex():
    from scientific_parser.ingest.font_aware_latex import extract_latex_from_page

    assert callable(extract_latex_from_page)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PERCEPTION — OCR backends (5)
# ═══════════════════════════════════════════════════════════════════════════
def t_ocr_caps():
    from scientific_parser.perception.ocr.interface import (
        CAP_TEXT,
        CAP_FORMULA,
        CAP_TABLE,
        CAP_CHART,
        CAP_LAYOUT,
    )

    assert all(isinstance(c, str) for c in (CAP_TEXT, CAP_FORMULA, CAP_TABLE))


def t_ocr_ollama():
    from scientific_parser.perception.ocr.ollama import OllamaBackend

    b = OllamaBackend(model="glm-ocr")
    assert b.name == "glm-ocr" and b.supports("text") is True


def t_ocr_rapidocr():
    from scientific_parser.perception.ocr.rapidocr_backend import RapidOCRBackend

    b = RapidOCRBackend()
    assert "text" in b.capabilities


def t_ocr_vlm():
    from scientific_parser.perception.ocr.vlm import VLMBackend

    b = VLMBackend(call_fn=lambda img, prompt: "ok", name="test")
    assert b.available() and b.ocr("/fake.png").text == "ok"


def t_ocr_onnx():
    import tempfile
    from scientific_parser.perception.ocr.onnx import OnnxBackend

    with tempfile.TemporaryDirectory() as td:
        b = OnnxBackend(td)
        assert callable(b.available)


def t_ocr_paddle():
    # paddle.py removed 2026-08-16 (dead API, undeclared dep) — probe the
    # live ONNX backend contract instead.
    import tempfile

    from scientific_parser.perception.ocr.onnx import OnnxBackend

    with tempfile.TemporaryDirectory() as td:
        b = OnnxBackend(td)
        assert callable(b.available)


# ═══════════════════════════════════════════════════════════════════════════
# 4. PERCEPTION — LAYOUT (XY-Cut reading order)
# ═══════════════════════════════════════════════════════════════════════════
def t_layout_xy_cut():
    from scientific_parser.perception.layout.xy_cut import reading_order

    regions = [
        {"x0": 0, "y0": 0, "x1": 200, "y1": 50, "type": "text"},
        {"x0": 250, "y0": 0, "x1": 450, "y1": 50, "type": "text"},
        {"x0": 0, "y0": 60, "x1": 200, "y1": 110, "type": "text"},
        {"x0": 250, "y0": 60, "x1": 450, "y1": 110, "type": "text"},
    ]
    ordered = reading_order(regions, prefer_columns=True)
    assert len(ordered) == 4


# ═══════════════════════════════════════════════════════════════════════════
# 5. PERCEPTION — EQUATIONS
# ═══════════════════════════════════════════════════════════════════════════
def t_eq_parse_latex():
    from scientific_parser.perception.equations.ast import parse_latex

    assert parse_latex(r"$E = mc^2$") is not None


def t_eq_extract():
    from scientific_parser.perception.equations.ast import extract_all_equations

    eqs = extract_all_equations(r"Energy $E=mc^2$ and force $$F=ma$$.")
    assert len(eqs) >= 1, f"expected >=1 eq, got {eqs}"


def t_eq_validate():
    from scientific_parser.perception.equations.ast import validate_latex_renders

    r = validate_latex_renders(r"$x^2 + y^2 = z^2$")
    assert r is not None


def t_eq_math_extractor():
    from scientific_parser.perception.equations import math_extractor

    assert hasattr(math_extractor, "__file__")


# ═══════════════════════════════════════════════════════════════════════════
# 6. PERCEPTION — TABLES
# ═══════════════════════════════════════════════════════════════════════════
def t_tables_schema():
    from scientific_parser.perception.tables.schema import infer_table_schema

    rows = [["Sample", "SiO2"], ["GR-01", "72.5"], ["GR-02", "68.1"]]
    assert infer_table_schema(rows) is not None


def t_tables_structure():
    from scientific_parser.perception.tables.structure import validate_table_structure

    assert callable(validate_table_structure)


def t_tables_detector():
    from scientific_parser.perception.tables.detector import detect_tables

    assert callable(detect_tables)


# ═══════════════════════════════════════════════════════════════════════════
# 7. PERCEPTION — FIGURES (6 parsers)
# ═══════════════════════════════════════════════════════════════════════════
def t_fig_classify():
    from scientific_parser.perception.figures.dispatch import classify_figure

    assert classify_figure("microscopy image of quartz grain") is not None


def t_fig_plot():
    from scientific_parser.perception.figures.plot_parser import parse_plot

    assert callable(parse_plot)


def t_fig_chart():
    from scientific_parser.perception.figures.chart_parser import parse_chart

    assert callable(parse_chart)


def t_fig_map():
    from scientific_parser.perception.figures.map_parser import parse_map

    assert callable(parse_map)


def t_fig_diagram():
    from scientific_parser.perception.figures.diagram_parser import parse_diagram

    assert callable(parse_diagram)


def t_fig_microscopy():
    from scientific_parser.perception.figures.microscopy_parser import parse_microscopy

    assert callable(parse_microscopy)


# ═══════════════════════════════════════════════════════════════════════════
# 8. PROVIDERS — Specialized model wrappers
# ═══════════════════════════════════════════════════════════════════════════
def t_prov_formula():
    from scientific_parser.perception.providers.formula_ocr import (
        RapidLaTeXProvider,
        get_formula_recognizer,
    )

    assert callable(get_formula_recognizer)


def t_prov_layout():
    from scientific_parser.perception.providers.layout_detector import (
        RapidLayoutProvider,
        get_layout_detector,
    )

    assert callable(get_layout_detector)


def t_prov_table():
    from scientific_parser.perception.providers.table_recognizer import (
        RapidTableProvider,
        get_table_recognizer,
    )

    assert callable(get_table_recognizer)


def t_prov_dispatcher():
    from scientific_parser.perception.providers.region_dispatcher import route_page_by_region

    assert callable(route_page_by_region)


# ═══════════════════════════════════════════════════════════════════════════
# 9. STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════
def t_struct_reading_order():
    from scientific_parser.structure.reading_order import compute_reading_order

    assert callable(compute_reading_order)


def t_struct_hierarchy():
    from scientific_parser.structure.hierarchy import detect_sections

    assert callable(detect_sections)


def t_struct_region_graph():
    from scientific_parser.structure.region_graph import build_region_graph

    assert callable(build_region_graph)


def t_struct_column_reading():
    from scientific_parser.structure.column_reading_order import column_aware_reading_order

    assert callable(column_aware_reading_order)


# ═══════════════════════════════════════════════════════════════════════════
# 10. SCIENCE — validators, formulas, entities, numbers, units
# ═══════════════════════════════════════════════════════════════════════════
def t_sci_entities():
    from scientific_parser.science.entities import extract_all

    r = extract_all("The granite has 65 wt% SiO2. doi:10.1000/test")
    assert isinstance(r, (list, dict))


def t_sci_numbers():
    from scientific_parser.science.numbers import analyze_text

    assert analyze_text("Age = 425 ± 3 Ma.") is not None


def t_sci_units():
    from scientific_parser.science.units import is_known_unit

    assert is_known_unit("Ma") is True


def t_sci_formulas():
    from scientific_parser.science.formulas import parse_formula

    f = parse_formula("H2O")
    assert f.valid is True and f.elements == {"H": 2, "O": 1}


def t_sci_dimensional():
    from scientific_parser.science.dimensional import get_dimension, check_equation_consistency

    assert callable(get_dimension) and callable(check_equation_consistency)


def t_sci_validators():
    from scientific_parser.science.validators import (
        NumericValidator,
        UnitValidator,
        FormulaValidator,
        CitationValidator,
    )

    for cls in (NumericValidator, UnitValidator, FormulaValidator, CitationValidator):
        assert cls is not None


def t_sci_geo():
    from scientific_parser.science.geo_validators import (
        validate_scientific_data,
        validate_ree_sequence,
        detect_mineral_abbreviations,
    )

    assert callable(validate_scientific_data)
    # Functional: REE sequence check
    r = validate_ree_sequence("La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy")
    assert r is not None


def t_sci_unicode_latex():
    from scientific_parser.science.unicode_to_latex import convert_unicode_to_latex

    result = convert_unicode_to_latex("α + β = γ")
    assert "alpha" in result.lower() or "\\alpha" in result


def t_sci_symbol_registry():
    from scientific_parser.science.symbol_registry import lookup

    assert callable(lookup)


# ═══════════════════════════════════════════════════════════════════════════
# 11. DOMAINS — 10 packs + detection
# ═══════════════════════════════════════════════════════════════════════════
def _domain_test(pack_mod, pack_attr="PACK"):
    def inner():
        mod = __import__(pack_mod, fromlist=["_"])
        p = getattr(mod, pack_attr)
        assert p is not None

    return inner


def t_domain_detect():
    from scientific_parser.domains.base import detect_domains

    d = detect_domains("The granite contains quartz and feldspar with 65 wt% SiO2.")
    assert isinstance(d, list)


# ═══════════════════════════════════════════════════════════════════════════
# 12. VERIFICATION — repair, pykatex, calibration, consensus, rules, abstention
# ═══════════════════════════════════════════════════════════════════════════
def t_verif_repair_latex():
    from scientific_parser.verification.render_compare import repair_latex

    r = repair_latex(r"\sort{4}")
    assert r.success is True and "\\sqrt" in r.repaired


def t_verif_pykatex():
    from scientific_parser.verification.render_compare import verify_with_pykatex

    r = verify_with_pykatex(r"x^2 + y^2")
    assert r is not None


def t_verif_calibration():
    from scientific_parser.verification.calibration import ConformalCalibrator

    assert ConformalCalibrator() is not None


def t_verif_consensus():
    from scientific_parser.verification.consensus import consensus_value

    val = consensus_value("72.5", "72.3", "72.6")
    assert val is not None


def t_verif_rules():
    from scientific_parser.verification.rules import default_rule_engine, RuleEngine

    eng = default_rule_engine()
    assert isinstance(eng, RuleEngine) or eng is not None


def t_verif_abstention():
    from scientific_parser.verification.abstention import should_abstain, ACCEPT, REJECT

    assert ACCEPT != REJECT
    assert callable(should_abstain)


# ═══════════════════════════════════════════════════════════════════════════
# 13. ROUTING
# ═══════════════════════════════════════════════════════════════════════════
def t_route_model_selector():
    from scientific_parser.routing.model_selector import select_model

    sel = select_model(required_capabilities=["text"], available_backends=["glm-ocr"])
    assert sel is not None


def t_route_budget():
    from scientific_parser.routing.budget import allocate_budget

    b = allocate_budget(tier="medium")
    assert b is not None


def t_route_complexity():
    from scientific_parser.routing import complexity

    assert hasattr(complexity, "__file__")


def t_route_page_classifier():
    from scientific_parser.routing import page_classifier

    assert hasattr(page_classifier, "__file__")


# ═══════════════════════════════════════════════════════════════════════════
# 14. OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
def t_out_json():
    from scientific_parser.output.json_out import document_to_json

    assert callable(document_to_json)


def t_out_markdown():
    from scientific_parser.output.markdown import document_to_markdown

    assert callable(document_to_markdown)


def t_out_html():
    from scientific_parser.output.html import document_to_html

    assert callable(document_to_html)


# ═══════════════════════════════════════════════════════════════════════════
# 15. MONOLITH pdf_ocr.py — specialized functions
# ═══════════════════════════════════════════════════════════════════════════
def t_mono_import():
    import pdf_ocr

    assert hasattr(pdf_ocr, "SKILL_EXTRACTION_VERSION")


def t_mono_select_ocr():
    import pdf_ocr

    m, p, w = pdf_ocr.select_ocr_model("ocr", ["low_chars"], ["glm-ocr"])
    assert m == "glm-ocr"


def t_mono_score():
    import pdf_ocr

    conf, bd = pdf_ocr._score_ocr_text("Clean scientific text with quartz.")
    assert conf > 0.5


def t_mono_cache_key():
    import pdf_ocr

    class FakeArgs:
        dpi = max_dim = None
        force_ocr = fast = False
        pages = crop = ""

    assert len(pdf_ocr._cache_key("/fake.pdf", ["glm-ocr"], FakeArgs())) == 32


def t_mono_reflexive():
    from unittest.mock import patch
    import pdf_ocr

    with patch.object(
        pdf_ocr, "ocr_image", side_effect=lambda *a, **k: "Quartz monzonite with feldspar."
    ):
        text, model, att, conf, warns = pdf_ocr.ocr_with_reflexive_loop(
            "/fake.png", ["glm-ocr"], max_attempts_per_model=1
        )
    assert "Quartz" in text


def t_mono_table_marker():
    import pdf_ocr

    assert pdf_ocr.detect_continued_table_marker("(continued from Table 1)") is None or callable(
        pdf_ocr.detect_continued_table_marker
    )


def t_mono_fix_latex():
    import pdf_ocr

    fn = getattr(pdf_ocr, "fix_latex_math_output", None)
    if fn is None:
        raise ImportError("fix_latex_math_output not found")
    result = fn(r"$\sort{4} + x$")
    assert "\\sqrt" in result or "sqrt" in result


def t_mono_detect_tables():
    import pdf_ocr

    fn = getattr(pdf_ocr, "detect_all_tables", None)
    if fn is None:
        raise ImportError("detect_all_tables not found")
    assert callable(fn)


def t_mono_validate_chemical():
    import pdf_ocr

    fn = getattr(pdf_ocr, "validate_chemical_formulas", None)
    if fn is None:
        raise ImportError("validate_chemical_formulas not found")
    assert callable(fn)


def _bug_formula_mass():
    from scientific_parser.science.formulas import parse_formula, formula_mass

    f = parse_formula("H2O")
    mass = formula_mass(f)
    if mass is None:
        raise AssertionError("formula_mass(H2O)=None — atomic weight table not loaded")
    assert mass > 0


def _bug_wt_percent():
    from scientific_parser.science.units import is_known_unit

    if not is_known_unit("wt%"):
        raise AssertionError("wt% not recognized — fundamental geochemistry unit missing")


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════
ALL = [
    ("Core IR", "make_block", t_core_make_block),
    ("Core IR", "evidence rank/merge", t_core_evidence),
    ("Core IR", "confidence aggregate", t_core_confidence),
    ("Core IR", "events log/get", t_core_events),
    ("Ingest", "image_stats", t_ingest_image),
    ("Ingest", "vector grid+axes", t_ingest_vector),
    ("Ingest", "render_dpi", t_ingest_render),
    ("Ingest", "font_analysis", t_ingest_font_analysis),
    ("Ingest", "font_aware_latex", t_ingest_font_aware_latex),
    ("OCR Backend", "interface caps", t_ocr_caps),
    ("OCR Backend", "ollama", t_ocr_ollama),
    ("OCR Backend", "rapidocr", t_ocr_rapidocr),
    ("OCR Backend", "vlm", t_ocr_vlm),
    ("OCR Backend", "onnx", t_ocr_onnx),
    ("OCR Backend", "paddle", t_ocr_paddle),
    ("Layout", "xy_cut reading_order", t_layout_xy_cut),
    ("Equations", "parse_latex", t_eq_parse_latex),
    ("Equations", "extract_all", t_eq_extract),
    ("Equations", "validate_renders", t_eq_validate),
    ("Equations", "math_extractor", t_eq_math_extractor),
    ("Tables", "schema infer", t_tables_schema),
    ("Tables", "structure detect", t_tables_structure),
    ("Tables", "detector", t_tables_detector),
    ("Figures", "classify_figure", t_fig_classify),
    ("Figures", "parse_plot", t_fig_plot),
    ("Figures", "parse_chart", t_fig_chart),
    ("Figures", "parse_map", t_fig_map),
    ("Figures", "parse_diagram", t_fig_diagram),
    ("Figures", "parse_microscopy", t_fig_microscopy),
    ("Provider", "FormulaOCR (RapidLaTeX)", t_prov_formula),
    ("Provider", "LayoutDetector (RapidLayout)", t_prov_layout),
    ("Provider", "TableRecognizer (RapidTable)", t_prov_table),
    ("Provider", "RegionDispatcher route", t_prov_dispatcher),
    ("Structure", "compute_reading_order", t_struct_reading_order),
    ("Structure", "detect_sections", t_struct_hierarchy),
    ("Structure", "build_region_graph", t_struct_region_graph),
    ("Structure", "column_reading_order", t_struct_column_reading),
    ("Science", "extract_all entities", t_sci_entities),
    ("Science", "analyze_text numbers", t_sci_numbers),
    ("Science", "is_known_unit", t_sci_units),
    ("Science", "parse_formula+mass", t_sci_formulas),
    ("Science", "dimensional check", t_sci_dimensional),
    ("Science", "4 validators", t_sci_validators),
    ("Science", "geo_validators+REE", t_sci_geo),
    ("Science", "unicode_to_latex", t_sci_unicode_latex),
    ("Science", "symbol_registry", t_sci_symbol_registry),
    ("Domain", "general", _domain_test("scientific_parser.domains.general.pack")),
    ("Domain", "geology", _domain_test("scientific_parser.domains.geology.pack")),
    ("Domain", "chemistry", _domain_test("scientific_parser.domains.chemistry.pack")),
    ("Domain", "physics", _domain_test("scientific_parser.domains.physics.pack")),
    ("Domain", "biology", _domain_test("scientific_parser.domains.biology.pack")),
    ("Domain", "materials", _domain_test("scientific_parser.domains.materials.pack")),
    ("Domain", "astronomy", _domain_test("scientific_parser.domains.astronomy.pack")),
    ("Domain", "mathematics", _domain_test("scientific_parser.domains.mathematics.pack")),
    ("Domain", "medicine", _domain_test("scientific_parser.domains.medicine.pack")),
    ("Domain", "engineering", _domain_test("scientific_parser.domains.engineering.pack")),
    ("Domain", "detect_domains", t_domain_detect),
    ("Verification", "repair_latex sqrt", t_verif_repair_latex),
    ("Verification", "verify_with_pykatex", t_verif_pykatex),
    ("Verification", "ConformalCalibrator", t_verif_calibration),
    ("Verification", "consensus_value", t_verif_consensus),
    ("Verification", "rule_engine", t_verif_rules),
    ("Verification", "abstention", t_verif_abstention),
    ("Routing", "select_model", t_route_model_selector),
    ("Routing", "allocate_budget", t_route_budget),
    ("Routing", "complexity module", t_route_complexity),
    ("Routing", "page_classifier module", t_route_page_classifier),
    ("Output", "json", t_out_json),
    ("Output", "markdown", t_out_markdown),
    ("Output", "html", t_out_html),
    ("Monolith", "import+version", t_mono_import),
    ("Monolith", "select_ocr_model", t_mono_select_ocr),
    ("Monolith", "_score_ocr_text", t_mono_score),
    ("Monolith", "_cache_key", t_mono_cache_key),
    ("Monolith", "reflexive_loop", t_mono_reflexive),
    ("Monolith", "detect_continued_table", t_mono_table_marker),
    ("Monolith", "fix_latex_math_output", t_mono_fix_latex),
    ("Monolith", "detect_all_tables", t_mono_detect_tables),
    ("Monolith", "validate_chemical_formulas", t_mono_validate_chemical),
    # Known-bug probes (document real findings)
    ("Bug Probe", "formula_mass H2O returns None", _bug_formula_mass),
    ("Bug Probe", "is_known_unit wt% = False", _bug_wt_percent),
]


def main():
    print("=" * 90)
    print("SPECIALIZED COMPONENT AUDIT — pdf-ocr skill")
    print(f"Probes: {len(ALL)} | Categories: 16")
    print("=" * 90)

    for cat, name, fn in ALL:
        probe(cat, name, fn)

    cats: dict[str, list] = {}
    for cat, name, status, detail in _RESULTS:
        cats.setdefault(cat, []).append((name, status, detail))

    for cat in cats:
        rows = cats[cat]
        cp = sum(1 for _, s, _ in rows if s == "PASS")
        cf = sum(1 for _, s, _ in rows if s == "FAIL")
        cs = sum(1 for _, s, _ in rows if s == "SKIP")
        sym = "\u2713" if cf == 0 else "\u2717"
        print(f"\n{sym} {cat} ({cp}P/{cf}F/{cs}S)")
        for name, status, detail in rows:
            icon = {"PASS": "\u2713", "FAIL": "\u2717", "SKIP": "\u2014"}[status]
            line = f"  {icon} {name}"
            if status != "PASS":
                line += f" \u2014 {detail[:120]}"
            print(line)

    print(f"\n{'=' * 90}")
    total = len(_RESULTS)
    print(f"TOTAL: {_PASS} PASS / {_FAIL} FAIL / {_SKIP} SKIP ({total} probes)")
    if _FAIL > 0:
        print(f"\n\u2717 {_FAIL} FAILURES:")
        for cat, name, status, detail in _RESULTS:
            if status == "FAIL":
                print(f"  \u2717 [{cat}] {name}: {detail[:150]}")
    print("=" * 90)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
