"""scientific_parser — top-level package.

Universal scientific document parsing with pluggable domain intelligence.
Dual-mode: standalone library + pdf-ocr skill backend.

Layers (proposal item #32):
  L1 Perception  — OCR, layout, tables, equations, figures, PDF vectors
  L2 Structure   — reading order, hierarchy, region graph
  L3 Normalization — Unicode, numbers, units, spacing
  L4 Semantics   — entities, measurements, formulas, relations
  L5 Verification — rules, consensus, calibration, abstention
"""

from __future__ import annotations

# === Core IR ============================================================ #
from .core.ir import (
    DocumentIR,
    PageIR,
    BlockIR,
    Evidence,
    VerificationResult,
    make_block,
    E0_NATIVE_PDF_TEXT,
    E1_NATIVE_PDF_GEOM,
    E2_DETERMINISTIC,
    E3_CLASSICAL_OCR,
    E4_SPECIALIST_OCR,
    E5_VLM,
    E6_SEMANTIC,
    EVIDENCE_PRIORITY,
)
from .core.evidence import (
    EvidenceResolver,
    DEFAULT_RESOLVER,
    EVIDENCE_SOURCE_RELIABILITY,
    rank_by_priority,
    merge_evidence,
)
from .core.provenance import Provenance, make_provenance
from .core.confidence import (
    TemperatureScaler,
    IsotonicMapper,
    aggregate_evidence_confidence,
    document_confidence_summary,
)
from .core.events import (
    EventLevel,
    PipelineEvent,
    log_event,
    get_events,
    clear_events,
    flush_events_jsonl,
    event_summary,
)

# === Ingest ============================================================= #
from .ingest.pdf_native import PDFNativeResult, extract_page_native, get_page_count
from .ingest.pdf_render import render_page, render_region, estimate_render_dpi
from .ingest.image import normalize_image, to_grayscale, autocontrast, image_stats
from .ingest.vector_extract import (
    VectorPath,
    TableGrid,
    PlotAxes,
    extract_vector_paths,
    detect_table_grid,
    detect_plot_axes,
)

# === Perception ======================================================== #
from .perception.ocr.interface import (
    OCRBackend,
    OCRResult,
    CAP_TEXT,
    CAP_MULTILINGUAL,
    CAP_FORMULA,
    CAP_TABLE,
    CAP_CHART,
    CAP_LAYOUT,
    CAP_HANDWRITING,
    CAP_VERTICAL_TEXT,
)
from .perception.ocr.ollama import OllamaBackend
from .perception.ocr.rapidocr_backend import RapidOCRBackend
from .perception.ocr.onnx import OnnxBackend
from .perception.ocr.vlm import VLMBackend

from .perception.layout.xy_cut import reading_order
from .perception.equations.ast import (
    EquationAST,
    parse_latex,
    extract_all_equations,
    validate_latex_renders,
)
from .perception.tables.schema import (
    ColumnSchema,
    infer_table_schema,
    infer_column_schema,
)
from .perception.figures.dispatch import (
    FigureClassification,
    classify_figure,
    FIG_PHOTO,
    FIG_PLOT,
    FIG_CHART,
    FIG_DIAGRAM,
    FIG_MAP,
    FIG_SCHEMATIC,
    FIG_MICROSCOPY,
    FIG_WORKFLOW,
    FIG_UNKNOWN,
)
from .perception.figures.plot_parser import PlotData, parse_plot
from .perception.figures.chart_parser import ChartData, parse_chart
from .perception.figures.map_parser import MapData, parse_map
from .perception.figures.diagram_parser import DiagramData, parse_diagram
from .perception.figures.microscopy_parser import MicroscopyData, parse_microscopy

# === Structure ========================================================= #
from .structure.reading_order import ReadingOrderResult, compute_reading_order
from .structure.hierarchy import SectionNode, detect_sections, build_tree
from .structure.region_graph import (
    RegionNode,
    build_region_graph,
    find_caption_for_figure,
)

# === Science =========================================================== #
from .science.entities import (
    Entity,
    extract_quantities,
    extract_dates,
    extract_persons,
    extract_citations,
    extract_dois,
    extract_all,
)
from .science.numbers import NumericFailure, NumericFinding, analyze_text, annotate_text
from .science.units import (
    get_registry,
    parse_quantity,
    check_plausible,
    normalize_unit_text,
    is_known_unit,
)
from .science.formulas import (
    ChemicalFormula,
    parse_formula,
    is_mineral_formula,
    formula_mass,
    find_formulas_in_text,
)
from .science.equations import (
    EquationAST as EqAST,
    parse_latex as parse_eq,
    extract_all_equations as extract_eqs,
    validate_latex_renders as validate_eq,
    dimensional_check,
)
from .science.validators import (
    NumericValidator,
    UnitValidator,
    FormulaValidator,
    CitationValidator,
)

# === Domains =========================================================== #
from .domains.base import (
    DomainPack,
    Validator,
    register as register_domain,
    detect_domains,
    active_lexicon,
    active_validators,
    all_packs,
)
from .domains.general.pack import GeneralPack, PACK as GENERAL_PACK
from .domains.geology.pack import GeologyPack, PACK as GEOLOGY_PACK
from .domains.chemistry.pack import ChemistryPack
from .domains.physics.pack import PhysicsPack
from .domains.biology.pack import BiologyPack
from .domains.materials.pack import MaterialsPack
from .domains.astronomy.pack import AstronomyPack
from .domains.mathematics.pack import MathPack
from .domains.medicine.pack import MedicinePack
from .domains.engineering.pack import EngineeringPack

# === Verification ====================================================== #
from .verification.render_compare import (
    RepairResult,
    repair_latex,
    repair_all_equations,
    verify_with_pykatex,
)
from .verification.calibration import (
    CandidateSet,
    ConformalCalibrator,
    numeric_nonconformity,
    ocr_nonconformity,
)
from .verification.rules import (
    RuleResult,
    Rule,
    RuleEngine,
    rule_min_length,
    rule_no_repetition,
    rule_no_cjk,
    rule_katex_valid,
    default_rule_engine,
)
from .verification.consensus import consensus_value, compare_texts
from .verification.abstention import (
    ACCEPT,
    REVIEW,
    REJECT,
    TriageResult,
    triage,
    annotate_with_triage,
    should_abstain,
)

# === Routing =========================================================== #
from .routing.model_selector import ModelSelection, select_model, list_capabilities
from .routing.budget import ComputeBudget, allocate_budget, summarize_budget
from .routing.complexity import (
    TIER_TRIVIAL,
    TIER_SIMPLE,
    TIER_MEDIUM,
    TIER_HARD,
    PageComplexity,
    assess_page_complexity,
    recommend_model,
    recommend_dpi,
)

# === Output ============================================================ #
from .output.json_out import document_to_json, write_json, entities_to_json
from .output.markdown import document_to_markdown, write_markdown
from .output.html import document_to_html, write_html

# === Benchmark ========================================================= #
from .benchmark.datasets import BenchDataset, load_olmocr_bench, list_tests, list_pdfs
from .benchmark.metrics import (
    char_error_rate,
    f1_token,
    equation_accuracy,
    numeric_accuracy,
)
import logging

log = logging.getLogger(__name__)
# benchmark/evaluator.py REMOVED 2026-08-15: its F1≥0.95 pass rule is NOT
# olmOCR-bench semantics — a landmine for anyone trusting it as a bench
# score. The authoritative scorer is olmocr.bench.benchmark.


def _autoregister():
    """Register all built-in domain packs."""
    try:
        from .domains.base import register, all_packs

        if not all_packs():
            register(GENERAL_PACK)
            register(GEOLOGY_PACK)
    except Exception as e:
        log.debug("domain pack registration degraded: %s", e)


_autoregister()


__all__ = [
    # Core
    "DocumentIR",
    "PageIR",
    "BlockIR",
    "Evidence",
    "make_block",
    "E0_NATIVE_PDF_TEXT",
    "E1_NATIVE_PDF_GEOM",
    "E2_DETERMINISTIC",
    "E3_CLASSICAL_OCR",
    "E4_SPECIALIST_OCR",
    "E5_VLM",
    "E6_SEMANTIC",
    "EVIDENCE_PRIORITY",
    "EvidenceResolver",
    "DEFAULT_RESOLVER",
    "EVIDENCE_SOURCE_RELIABILITY",
    "rank_by_priority",
    "merge_evidence",
    "VerificationResult",
    "Provenance",
    "make_provenance",
    "TemperatureScaler",
    "IsotonicMapper",
    "aggregate_evidence_confidence",
    "document_confidence_summary",
    "EventLevel",
    "PipelineEvent",
    "log_event",
    "get_events",
    "clear_events",
    "flush_events_jsonl",
    "event_summary",
    # Ingest
    "PDFNativeResult",
    "extract_page_native",
    "get_page_count",
    "render_page",
    "render_region",
    "estimate_render_dpi",
    "normalize_image",
    "to_grayscale",
    "autocontrast",
    "image_stats",
    "VectorPath",
    "TableGrid",
    "PlotAxes",
    "extract_vector_paths",
    "detect_table_grid",
    "detect_plot_axes",
    # Perception
    "OCRBackend",
    "OCRResult",
    "CAP_TEXT",
    "CAP_MULTILINGUAL",
    "CAP_FORMULA",
    "CAP_TABLE",
    "CAP_CHART",
    "CAP_LAYOUT",
    "CAP_HANDWRITING",
    "CAP_VERTICAL_TEXT",
    "OllamaBackend",
    "RapidOCRBackend",
    "OnnxBackend",
    "VLMBackend",
    "reading_order",
    "EquationAST",
    "parse_latex",
    "extract_all_equations",
    "validate_latex_renders",
    "ColumnSchema",
    "infer_table_schema",
    "infer_column_schema",
    "FigureClassification",
    "classify_figure",
    "FIG_PHOTO",
    "FIG_PLOT",
    "FIG_CHART",
    "FIG_DIAGRAM",
    "FIG_MAP",
    "FIG_SCHEMATIC",
    "FIG_MICROSCOPY",
    "FIG_WORKFLOW",
    "FIG_UNKNOWN",
    "PlotData",
    "parse_plot",
    "ChartData",
    "parse_chart",
    "MapData",
    "parse_map",
    "DiagramData",
    "parse_diagram",
    "MicroscopyData",
    "parse_microscopy",
    # Structure
    "ReadingOrderResult",
    "compute_reading_order",
    "SectionNode",
    "detect_sections",
    "build_tree",
    "RegionNode",
    "build_region_graph",
    "find_caption_for_figure",
    # Science
    "Entity",
    "extract_quantities",
    "extract_dates",
    "extract_persons",
    "extract_citations",
    "extract_dois",
    "extract_all",
    "NumericFailure",
    "NumericFinding",
    "analyze_text",
    "annotate_text",
    "parse_quantity",
    "check_plausible",
    "normalize_unit_text",
    "is_known_unit",
    "ChemicalFormula",
    "parse_formula",
    "is_mineral_formula",
    "formula_mass",
    "find_formulas_in_text",
    "dimensional_check",
    "NumericValidator",
    "UnitValidator",
    "FormulaValidator",
    "CitationValidator",
    # Domains
    "DomainPack",
    "Validator",
    "register_domain",
    "detect_domains",
    "active_lexicon",
    "active_validators",
    "all_packs",
    "GeneralPack",
    "GeologyPack",
    "ChemistryPack",
    "PhysicsPack",
    "BiologyPack",
    "MaterialsPack",
    "AstronomyPack",
    "MathPack",
    "MedicinePack",
    "EngineeringPack",
    # Verification
    "RepairResult",
    "repair_latex",
    "repair_all_equations",
    "verify_with_pykatex",
    "CandidateSet",
    "ConformalCalibrator",
    "numeric_nonconformity",
    "ocr_nonconformity",
    "RuleResult",
    "Rule",
    "RuleEngine",
    "rule_min_length",
    "rule_no_repetition",
    "rule_no_cjk",
    "rule_katex_valid",
    "default_rule_engine",
    "consensus_value",
    "compare_texts",
    "ACCEPT",
    "REVIEW",
    "REJECT",
    "TriageResult",
    "triage",
    "annotate_with_triage",
    "should_abstain",
    # Routing
    "ModelSelection",
    "select_model",
    "list_capabilities",
    "ComputeBudget",
    "allocate_budget",
    "summarize_budget",
    "TIER_TRIVIAL",
    "TIER_SIMPLE",
    "TIER_MEDIUM",
    "TIER_HARD",
    "PageComplexity",
    "assess_page_complexity",
    "recommend_model",
    "recommend_dpi",
    # Output
    "document_to_json",
    "write_json",
    "entities_to_json",
    "document_to_markdown",
    "write_markdown",
    "document_to_html",
    "write_html",
    # Benchmark
    "BenchDataset",
    "load_olmocr_bench",
    "list_tests",
    "list_pdfs",
    "char_error_rate",
    "f1_token",
    "equation_accuracy",
    "numeric_accuracy",
]
