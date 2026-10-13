"""Postproc - unified post-processor registry + cleanup functions.

Extracted from scripts/pdf_ocr.py during Phase R1 refactoring (2026-08-12).
Pure move - no behavioral changes. All names re-exported by pdf_ocr.py.
"""

from __future__ import annotations

import re
import os
import sys
import html as _htmlmod
import logging
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state

# `import *` skips underscore names: without this explicit import,
# _get_page_dict raised NameError inside detect_section_hierarchy and
# reconstruct_footnotes, silently killing both features (caught by their
# try/except, stderr log.warning only). EXP-6c revival fix.
from scientific_parser.routing.page_metadata import _get_page_dict

_REGISTRY_CACHE = None  # lazy-init singleton for _get_registry()
_CONSTS_LOADED = False


def _load_consts():
    """Lazy-load ALL cross-module names from pdf_ocr at first call.

    Uses vars() to pull everything — avoids enumerating 100+ individual names.
    Idempotent via _CONSTS_LOADED flag.
    """
    global _CONSTS_LOADED
    if _CONSTS_LOADED:
        return
    p = _pdf_ocr()
    g = globals()
    # Pull ALL public + private names from pdf_ocr into our globals.
    # Only overwrite names NOT already defined locally (avoid clobbering
    # our own functions/constants with re-imports of ourselves).
    own_keys = set(g.keys())
    for k, v in vars(p).items():
        if k not in own_keys and not k.startswith("__"):
            g[k] = v
    _CONSTS_LOADED = True


def _pdf_ocr():
    import sys

    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


# Old _load_consts removed — using vars()-based version at line 20 instead.


def restore_degree_symbol(text):
    """Restore lost degree symbol in temperature expressions.

    Two source-PDF encoding patterns produce this loss:
      1. Font subsetting: °C encoded as '8C' (the degree sign maps to glyph
         index 8 in some font subsets). Affects born-digital AND OCR'd PDFs.
      2. Encoding loss: °C → U+FFFD (replacement char) '�C'. Affects PDFs
         with broken CMaps.

    Patterns matched (case-sensitive, requires digit prefix to avoid math
    false positives like "equation 8C"):
      - "650 8C"        → "650 °C"  (digit + space + 8C)
      - "50 �C"         → "50 °C"   (replacement char variant)
      - "(6508C)"       → "(650°C)" (no space variant, in parens)

    Does NOT match:
      - "equation 8C"   (no preceding digit)
      - "1.8C"          (decimal not followed by space — likely math expr)
      - "C8"            (reversed)
    """
    if not text:
        return text
    # Pattern: digit (optional space) + 8 or U+FFFD (optional space) + C/K/F + word boundary
    # The \s* between 8/FFD and C/K/F handles "500 � C" (space between FFD and C).
    # The digit prefix is the key context guard (avoids "equation 8C").
    text = re.sub(
        r"(\d)\s*[8\uFFFD]\s*([CKF])\b",
        r"\1 °\2",
        text,
    )
    return text


# Geochemistry/mineralogy starter dictionary for OCR correction.
# Sources: IMA minerals (top 500 by frequency), common geological terms,
# chemistry element names, petrology/geochemistry jargon.
# Format: SymSpell-compatible (term frequency=1).
_GEOLOGY_DICT_TERMS = {
    # Mineral names (high-frequency IMA list)
    "quartz",
    "feldspar",
    "plagioclase",
    "alkali",
    "albite",
    "anorthite",
    "orthoclase",
    "microcline",
    "sanidine",
    "olivine",
    "forsterite",
    "fayalite",
    "pyroxene",
    "clinopyroxene",
    "orthopyroxene",
    "enstatite",
    "ferrosilite",
    "diopside",
    "hedenbergite",
    "augite",
    "pigeonite",
    "spodumene",
    "jadeite",
    "amphibole",
    "hornblende",
    "tremolite",
    "actinolite",
    "glaucophane",
    "riebeckite",
    "crocidolite",
    "anthophyllite",
    "cumingtonite",
    "gedrite",
    "mica",
    "muscovite",
    "biotite",
    "phlogopite",
    "lepidolite",
    "paragonite",
    "chlorite",
    "clinochlore",
    "chamosite",
    "serpentine",
    "antigorite",
    "lizardite",
    "chrysotile",
    "talc",
    "pyrophyllite",
    "kaolinite",
    "illite",
    "smectite",
    "montmorillonite",
    "vermiculite",
    "garnet",
    "almandine",
    "pyrope",
    "spessartine",
    "grossular",
    "andradite",
    "uvarovite",
    "epidote",
    "zoisite",
    "clinozoisite",
    "piedmontite",
    "allanite",
    "sillimanite",
    "kyanite",
    "andalusite",
    "topaz",
    "staurolite",
    "chloritoid",
    "cordierite",
    "spinel",
    "magnetite",
    "hercynite",
    "chromite",
    "titanite",
    "sphene",
    "perovskite",
    "zircon",
    "monazite",
    "xenotime",
    "apatite",
    "fluorapatite",
    "chlorapatite",
    "hydroxylapatite",
    "calcite",
    "dolomite",
    "magnesite",
    "siderite",
    "rhodochrosite",
    "smithsonite",
    "aragonite",
    "strontianite",
    "witherite",
    "barite",
    "celestine",
    "gypsum",
    "anhydrite",
    "halite",
    "sylvite",
    "fluorite",
    "pyrite",
    "pyrrhotite",
    "chalcopyrite",
    "bornite",
    "chalcocite",
    "covellite",
    "galena",
    "sphalerite",
    "wurtzite",
    "cinnabar",
    "realgar",
    "orpiment",
    "stibnite",
    "bismuthinite",
    "molybdenite",
    "pentlandite",
    "violarite",
    "millerite",
    "niccolite",
    "skutterudite",
    "smaltite",
    "cobaltite",
    "arsenopyrite",
    "marcasite",
    "loellingite",
    "safflorite",
    "rutile",
    "anatase",
    "brookite",
    "ilmenite",
    "hematite",
    "goethite",
    "limonite",
    "lepidocrocite",
    "maghemite",
    "corundum",
    "cassiterite",
    "cuprite",
    "tenorite",
    "zincite",
    "brucite",
    "gibbsite",
    "boehmite",
    "diaspore",
    "manganite",
    "psilomelane",
    "pyrolusite",
    "cryptomelane",
    "hollandite",
    "romanechite",
    "todorokite",
    "birnessite",
    # Rock names
    "granite",
    "granodiorite",
    "tonalite",
    "diorite",
    "gabbro",
    "norite",
    "anorthosite",
    "peridotite",
    "dunite",
    "harzburgite",
    "lherzolite",
    "wehrlite",
    "pyroxenite",
    "komatiite",
    "basalt",
    "andesite",
    "dacite",
    "rhyolite",
    "trachyte",
    "phonolite",
    "pumice",
    "obsidian",
    "tuff",
    "ignimbrite",
    "pegmatite",
    "aplite",
    "porphyry",
    "sandstone",
    "shale",
    "limestone",
    "dolostone",
    "mudstone",
    "siltstone",
    "conglomerate",
    "breccia",
    "chert",
    "flint",
    "evaporite",
    "schist",
    "gneiss",
    "slate",
    "phyllite",
    "quartzite",
    "marble",
    "skarn",
    "hornfels",
    "mylonite",
    "migmatite",
    "eclogite",
    "blueschist",
    "greenschist",
    "amphibolite",
    "granulite",
    "serpentinite",
    "soapstone",
    # Geochemistry terms
    "geochemistry",
    "geothermometry",
    "geobarometry",
    "thermobarometry",
    "partitioning",
    "coefficient",
    "distribution",
    "fractionation",
    "fractional",
    "crystallization",
    "differentiation",
    "assimilation",
    "contamination",
    "weathering",
    "erosion",
    "deposition",
    "diagenesis",
    "lithification",
    "metamorphism",
    "metasomatism",
    "alteration",
    "hydrothermal",
    "metasomatic",
    "magmatic",
    "volcanic",
    "plutonic",
    "igneous",
    "aqueous",
    "fluid",
    "melt",
    "solution",
    "solubility",
    "saturation",
    "undersaturated",
    "saturated",
    "oversaturated",
    "volatile",
    "volatiles",
    "incompatible",
    "compatible",
    "refractory",
    "lithophile",
    "siderophile",
    "chalcophile",
    "atmophile",
    "rare",
    "earth",
    "elements",
    "trace",
    "isotope",
    "isotopes",
    "isotopic",
    "radiogenic",
    "decay",
    "radioactive",
    "geochronology",
    "dating",
    "cosmogenic",
    "nuclide",
    "concentration",
    "abundance",
    "enrichment",
    "depletion",
    "anomaly",
    "anomalies",
    "anomalous",
    "normalized",
    "chondrite",
    "primitive",
    "mantle",
    "depleted",
    "enriched",
    "lithosphere",
    "asthenosphere",
    "mesosphere",
    "crust",
    "core",
    "conduction",
    "convection",
    "advection",
    "diffusion",
    "percolation",
    "infiltration",
    "reaction",
    "transport",
    "kinetics",
    "thermodynamics",
    "equilibrium",
    "disequilibrium",
    "stoichiometry",
    "stoichiometric",
    "valence",
    "oxidation",
    "reduction",
    "redox",
    "buffer",
    "fugacity",
    "oxygen",
    "sulfur",
    "activity",
    "chemical",
    "potential",
    "composition",
    "compositional",
    "oxide",
    "mole",
    # Process / geology terms
    "petrology",
    "petrogenesis",
    "petrogenetic",
    "tectonic",
    "tectonics",
    "subduction",
    "collision",
    "rifting",
    "extension",
    "compression",
    "uplift",
    "subsidence",
    "deformation",
    "strain",
    "stress",
    "fault",
    "fracture",
    "joint",
    "fold",
    "foliation",
    "lineation",
    "fabric",
    "shear",
    "kinematic",
    "dynamic",
    "rheology",
    "viscosity",
    "plasticity",
    "elasticity",
    "brittle",
    "ductile",
    "friction",
    "pressure",
    "temperature",
    "depth",
    "lithology",
    "stratigraphy",
    "stratigraphic",
    "sedimentology",
    "depositional",
    "facies",
    "sequence",
    "formation",
    "unconformity",
    "correlation",
    # Common institutions / publishing
    "University",
    "Institute",
    "Research",
    "Department",
    "Laboratory",
    "School",
    "Sciences",
    "Science",
    "Technology",
    "Engineering",
    "Physics",
    "Chemistry",
    "Biology",
    "Geology",
    "Geophysics",
    "Geochemistry",
    "Mineralogy",
    "Petrology",
    "Earth",
    "Planetary",
    "Environmental",
    "Climate",
    "Atmospheric",
    "Oceanic",
    "Received",
    "Accepted",
    "Revised",
    "Abstract",
    "Introduction",
    "Methods",
    "Methodology",
    "Results",
    "Discussion",
    "Conclusion",
    "Conclusions",
    "Acknowledgements",
    "References",
    "Bibliography",
    "Appendix",
    "Figure",
    "Table",
    "Equation",
    # Common German / umlaut terms (frequent OCR error sources)
    "Zürich",
    "für",
    "Tatsumi",
    "Mössbauer",
    "Mößbauer",
    "Düsseldorf",
    "München",
    "Häfner",
    "Lüders",
    "Müller",
    "Schmidt",
    "Weber",
    "Meyer",
    "Wagner",
    "Becker",
    "Schneider",
    "Hoffmann",
    "Fischer",
    "Walter",
}

# Lazily-initialized SymSpell instance (built once per process)

# Cross-page table-continuation state (merge_cross_page_tables). Module
# init is LOAD-BEARING: the function reads the global before any branch
# assigns it when its FIRST call carries a non-empty table list, so a
# missing init raises NameError that the defensive wrapper used to
# swallow at debug level — order-dependent silent table loss (measured
# 2026-09-15: 53d3f304_pg3 table present in harness batch, dropped in
# fresh-process runs; init missing since the 8cbb51f4 refactor era).
_OPEN_TABLES_STATE: list = []


def correct_ocr_errors(text, page_source=None):
    """Conservative SymSpell-based OCR error correction.

    Only runs on OCR'd pages (page_source='ocr'). For each word:
      1. If word is in geology dict → keep as-is (no change)
      2. If not, look up SymSpell suggestions at edit-distance ≤1
      3. Apply ONLY if EXACTLY ONE closest suggestion exists

    Conservative guards (avoid false corrections):
      - Word length 4-20 (don't touch "I", "a", "an"; don't touch identifiers)
      - Preserve capitalization pattern (don't lowercase ALLCAPS acronyms)
      - Don't touch words containing digits (sample IDs, isotopes)
      - Don't touch words with hyphens/apostrophes (U-Pb, O'Hara)
    """
    if not text or page_source != "ocr":
        return text
    ss = _get_symspell()
    if ss is None:
        return text
    try:
        from symspellpy import Verbosity
    except ImportError:
        return text

    # Tokenize alphabetic words 4-20 chars (skip digits/hyphens/apostrophes)
    word_pattern = re.compile(r"\b[A-Za-z]{4,20}\b")

    def _correct_word(match):
        word = match.group(0)
        # Skip ALLCAPS acronyms (≥3 chars uppercase) — preserve as-is
        if len(word) >= 3 and word.isupper():
            return word
        lower = word.lower()
        # Fast-path: word already in geology dict
        if lower in _GEOLOGY_DICT_TERMS or word in _GEOLOGY_DICT_TERMS:
            return word
        # SymSpell lookup at distance ≤1 (conservative — only single-char errors)
        suggestions = list(ss.lookup(lower, Verbosity.CLOSEST, max_edit_distance=1))
        # Filter out distance-0 (already correct)
        candidates = [s for s in suggestions if s.distance > 0]
        if not candidates:
            return word
        # Rank by (distance ASC, count DESC) and apply only if top candidate
        # is clearly dominant (count ≥ 100x the runner-up, OR only 1 candidate)
        candidates.sort(key=lambda s: (s.distance, -s.count))
        top = candidates[0]
        if len(candidates) == 1:
            best = top
        elif candidates[0].distance < candidates[1].distance:
            best = top  # closer distance wins
        elif top.count >= 100 * candidates[1].count:
            best = top  # 100x more frequent → confident
        else:
            return word  # ambiguous — don't correct
        # Apply correction preserving capitalization
        corrected = best.term
        if word[0].isupper():
            corrected = corrected.capitalize()
        return corrected

    return word_pattern.sub(_correct_word, text)


def fix_text_encoding(text):
    """Fix broken Unicode via ftfy. Marker uses ftfy to clean text from both
    pdftotext and OCR. Common fixes:
    - mojibake (Ã© â é), smart quotes, ligatures (ï¬ â fi)
    - lossy recoding artifacts from PDFs saved with wrong encoding
    """
    _load_consts()
    if not text:
        return text
    return _ftfy_fix_text(text)


# EXP-1.5: control-char noise scrub set (see _fix_text_encoding_full).
_CTRL_SCRUB_RE_PIPELINE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _fix_text_encoding_full(text):
    """Order-100 text hygiene: control-char scrub + ftfy.

    EXP-1.5 (2026-09-16, routing audit): symbol fonts on born-digital pages
    emit 0x07-0x1f glyph codes (bullets, diagram labels) interleaved with
    real prose. The router no longer treats them as an OCR trigger when the
    layer holds >=200 alphanumerics, so every fast-path text source (by_page
    pdftotext, fitz reorders, pdfplumber crops) must be scrubbed HERE — the
    registry is the single choke-point downstream of all ingestion paths.
    \t\n\r preserved; \x0b/\x0c included (unlike _PRIVATE_CHAR_RE).
    """
    _load_consts()
    if not text:
        return text
    if _CTRL_SCRUB_RE_PIPELINE.search(text):
        text = _CTRL_SCRUB_RE_PIPELINE.sub("", text)
    return _ftfy_fix_text(text)


# Marker's drop_repeated_text — drops OCR hallucination duplicates.
# VLMs sometimes emit the same paragraph 2-4 times in a row (we observed this
# on Craig 1973 and Spear 2002 OCR output). Heuristic: if a non-trivial line
# (>40 chars) appears consecutively ≥2 times, keep only the first.


def drop_repeated_text(text):
    """Remove OCR hallucination duplicates: identical long lines anywhere on page.

    Marker SOTA (drop_repeated_text). Conservative: only drops EXACT
    duplicates ≥40 chars. Tracks all seen long-lines and drops subsequent
    matches — catches both consecutive dupes (typical) and alternating
    dupes (e.g. Figure 7/8 paragraphs repeating in cycle, observed in
    OCR output on Spear 2002).
    """
    lines = text.split("\n")
    out = []
    seen_long = set()
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= REPEAT_LINE_MIN_CHARS:
            if stripped in seen_long:
                # Drop duplicate (also drop a trailing blank line if present)
                if out and not out[-1].strip():
                    out.pop()
                continue
            seen_long.add(stripped)
        out.append(line)
    return "\n".join(out)


# Bold-duplicate char detection. Some PDFs render bold by drawing the same
# character twice at slightly offset positions; the text layer then contains
# "hheelllloo" instead of "hello". Marker detects this via char-position
# analysis. We use a simpler heuristic: consecutive duplicate chars that
# resolve to a real word when de-duplicated.
# Source: Marker providers/pdf.py _remove_duplicate_chars


def remove_bold_duplicate_chars(text):
    """Fix bold-rendered-as-duplicate characters: 'hheelllloo' → 'hello'.

    Marker SOTA (_remove_duplicate_chars). Heuristic: for each word, if
    halving its length by taking every other char produces a "more normal"
    word (lower char-pair repetition), do so. Conservative: only applied
    to words ≥4 chars where the original has high pair-repetition and the
    halved version doesn't.
    """
    import re as _re

    def fix_word(w):
        if len(w) < DUPLICATE_CHAR_MIN_LEN:  # need ≥4 chars to consider
            return w
        if len(w) % 2 != 0:
            return w
        # Take every other char
        halved = w[::2]
        # Check: original has high pair-repetition (each char appears twice
        # consecutively), halved doesn't
        orig_pairs = sum(1 for i in range(0, len(w) - 1, 2) if w[i] == w[i + 1])
        halved_pairs = sum(1 for i in range(len(halved) - 1) if halved[i] == halved[i + 1])
        # Strong signal: ≥75% pairs in original, ≤25% in halved
        if orig_pairs >= len(w) // 2 * 0.75 and halved_pairs <= len(halved) // 4:
            return halved
        return w

    # Apply per word (preserve whitespace)
    return _re.sub(r"\S+", lambda m: fix_word(m.group(0)), text)


def merge_cross_page_tables(page_num, tables_on_page):
    """Merge tables that span page boundaries.

    A table on page N+1 is a CONTINUATION of a table on page N if:
      1. Same X-extent (within 10pt)
      2. Same column count
      3. Page N had a table (any table, typically the last)

    After page N: _OPEN_TABLES_STATE = [last_table_on_page_N].
    On page N+1: if first table matches last_open's X+cols, merge rows.

    Returns: list of tables to emit on current page (merged if continuation).
    """
    global _OPEN_TABLES_STATE
    if not tables_on_page:
        # Page has no tables — close any open table, emit it on prev page's output
        # (already emitted; just clear state)
        _OPEN_TABLES_STATE = []
        return []
    first = tables_on_page[0]
    rest = tables_on_page[1:]
    if not _OPEN_TABLES_STATE:
        # No open table from previous page
        result = list(tables_on_page)
    else:
        last_open = _OPEN_TABLES_STATE[-1]
        open_bbox = last_open["bbox"]
        first_bbox = first["bbox"]
        x_match = abs(open_bbox[0] - first_bbox[0]) < 10 and abs(open_bbox[2] - first_bbox[2]) < 10
        open_cols = max(len(r) for r in last_open["rows"]) if last_open["rows"] else 0
        first_cols = max(len(r) for r in first["rows"]) if first["rows"] else 0
        col_match = open_cols == first_cols and open_cols > 0
        if x_match and col_match:
            # Continuation — merge first's rows into last_open
            last_open["rows"].extend(first["rows"])
            last_open["bbox"] = (
                min(open_bbox[0], first_bbox[0]),
                open_bbox[1],
                max(open_bbox[2], first_bbox[2]),
                first_bbox[3],
            )
            last_open["markdown"] = build_table_markdown(
                last_open["rows"], last_open.get("strategy", "merged")
            )
            result = [last_open] + rest
        else:
            # No continuation — close open table (already emitted on prev page),
            # emit this page's tables fresh
            result = list(tables_on_page)
    # Track last table on this page as open for next page
    _OPEN_TABLES_STATE = [result[-1]] if result else []
    return result


def detect_all_tables(pdf_path, page_num, ocr_text=None, raster_only=False):
    """Multi-strategy table detector — closes all 6 gaps.

    Runs four strategies, dedups by bbox overlap >50%:

      1. **pdfplumber text-strategy** (decimal-bearing tables)
      2. **lines_strict** (ruled-line tables via PyMuPDF)
      3. **text-only pattern** (tables WITHOUT decimals)
      4. **OCR-text parse** (markdown tables from glm-ocr output)

    Each table's markdown is built via build_table_markdown which handles
    multi-row header merging (collapses 2-3 header rows into one combined row).

    For OCR'd pages, pass ocr_text= to also scan the model's output for
    pipe-tables it may have emitted (these are validated + multi-row merged).

    raster_only=True (det48) scopes detection to raster-embedded candidates:
    the caller established the page's ONLY table evidence is a table-ish
    image (layout_gate.raster_only_table_evidence), so text/geometry
    strategies are skipped — they see no table signal and only emit
    positional/whitespace false positives on prose around the image
    (measured odlbench det47: 0056/0066/0098/0152 gained 3-11 junk pipes).

    Returns: list of {"bbox", "markdown", "rows", "strategy"} dicts.
    """
    all_tables = []
    # 2026-08-16 iter-8 (tables loop): these strategies were referenced but
    # NEVER imported here — every call raised NameError, swallowed at
    # debug level → postproc table detection was silently dead (measured:
    # 117 'No tables found' bench tests; pdfplumber found 2 ruled tables
    # the pipeline never emitted).
    from scientific_parser.perception.tables.detector import (  # noqa: E402
        build_table_markdown,
        detect_tables_camelot,
        detect_tables_in_text,
        detect_tables_lines_strict,
        detect_tables_pdfplumber,
        detect_text_only_tables,
    )
    from scientific_parser.perception.tables.structure import _dedup_tables  # noqa: E402

    if not raster_only:
        try:
            all_tables.extend(detect_tables_pdfplumber(pdf_path, page_num))
        except Exception as e:
            log.warning("pdfplumber strategy failed p%s: %s", page_num, e)
        try:
            all_tables.extend(detect_tables_lines_strict(pdf_path, page_num))
        except Exception as e:
            log.warning("lines_strict strategy failed p%s: %s", page_num, e)
        try:
            all_tables.extend(
                detect_text_only_tables(
                    pdf_path, page_num, prior_bboxes=[t["bbox"] for t in all_tables]
                )
            )
        except Exception as e:
            log.warning("text_only strategy failed p%s: %s", page_num, e)
        # Strategy 4b: positional word-coordinate reconstruction — the only
        # strategy that recovers CELL-PER-LINE layers (each cell its own text
        # line; geometry lives in word x/y). 57/188 bench pages had no table
        # from any other strategy (2026-08-15).
        try:
            from scientific_parser.perception.tables.detector import (
                detect_tables_positional,
            )

            all_tables.extend(detect_tables_positional(pdf_path, page_num))
        except Exception as e:
            log.warning("positional strategy failed p%s: %s", page_num, e)
        # Strategy 4c: whitespace-alignment borderless detection (img2table
        # port) — LAST-RESORT ONLY: participates only where NO other strategy
        # already produced a table. Measured A/B over the 34 firing PDFs with
        # gold tests: unconditional participation = NET −8 (26076dc 12→1,
        # 1a67d1 4→1 — its glued rows displaced better camelot/lines tables);
        # the wins (0cda549 +3, 7e85030 +3, 53d3f304 family) all sit on pages
        # where no other strategy covered the region.
        try:
            from scientific_parser.perception.tables.detector import (
                detect_tables_whitespace,
            )

            _ws_tables = detect_tables_whitespace(pdf_path, page_num)
            for _wt in _ws_tables:
                _bx = _wt["bbox"]
                _covered_good = False
                for _ot in all_tables:
                    # TENTATIVE even-pitch text_only tables (iter-115)
                    # don't count as coverage — they admit NUMERIC grids
                    # outside that detector's contract; the whitespace
                    # specialist retains priority over them.
                    if _ot.get("even_pitch"):
                        continue
                    # A text_only_pattern table carrying FEWER rows than the
                    # whitespace candidate is weak coverage (fragmented
                    # clusters; iter-115 b10f6b pg1: a 6-row fragment
                    # suppressed the correct 13-row whitespace roster) — let
                    # whitespace compete; dedup still keeps the higher-row
                    # table.
                    if _ot.get("strategy") == "text_only_pattern" and len(
                        _ot.get("rows", [])
                    ) < len(_wt.get("rows", [])):
                        continue
                    _ob = _ot["bbox"]
                    _ix = max(0.0, min(_bx[2], _ob[2]) - max(_bx[0], _ob[0]))
                    _iy = max(0.0, min(_bx[3], _ob[3]) - max(_bx[1], _ob[1]))
                    if _ix * _iy > 0.4 * max(1.0, (_bx[2] - _bx[0]) * (_bx[3] - _bx[1])):
                        # Overlapping table is trustworthy when its cells are
                        # neither mojibake-polluted nor glued (multi-word header
                        # cells spanning >2 columns of content). Measured A/B:
                        # 26076dc camelot table (clean per-column headers) must
                        # WIN; 53d3f304 pdfplumber table (glued + (cid:) cells)
                        # must LOSE.
                        if not _is_glued_or_mojibake(_ot):
                            _covered_good = True
                            break
                if not _covered_good:
                    all_tables.append(_wt)
        except Exception as e:
            log.warning("whitespace strategy failed p%s: %s", page_num, e)
        # Strategy 4d: rowspan-label value-band tables (det42, 9c389720
        # Table S3 class) — sparse section labels governing center-aligned
        # value runs defeat every per-row multi-cell collector. LAST-RESORT:
        # participates ONLY on pages where no other strategy produced any
        # table (chatty-list contract — measured: looser overlap gating let
        # prose-label carves onto pages that already had positional tables,
        # 26076dc/b5d9db35 garbage emissions).
        try:
            from scientific_parser.perception.tables.detector import (
                detect_tables_rowspan_labels,
            )

            if not all_tables:
                all_tables.extend(detect_tables_rowspan_labels(pdf_path, page_num))
        except Exception as e:
            log.warning("rowspan_labels strategy failed p%s: %s", page_num, e)
    # Strategy 4e: raster-embedded tables (det43, 3a91fad8/587a1f5a
    # class) — whole tables typeset as raster images carry no (or
    # partial) text layer, so every text-based strategy misses. OCR the
    # region with rapidocr and rebuild the grid from box geometry. Same
    # last-resort chatty-list contract as 4d: fires ONLY when no other
    # strategy produced a table for the page. Under raster_only=True
    # (det48) this is the PRIMARY strategy by construction — every text
    # strategy above was skipped.
    try:
        from scientific_parser.perception.tables.detector import (
            detect_tables_raster,
        )

        if not all_tables:
            all_tables.extend(detect_tables_raster(pdf_path, page_num))
    except Exception as e:
        log.warning("raster strategy failed p%s: %s", page_num, e)
    if ocr_text:
        try:
            ocr_tables = detect_tables_in_text(ocr_text)
            all_tables.extend(ocr_tables)
            # VL CHATTY-LIST RECOVERY (iter-71, 4db371ae): the model
            # transcribed the table as narration ('Column N:' + numbered
            # 'Data Rows:') with no pipe table. Convert it — but only
            # when nothing else produced a table for the page (the
            # chatty grid is the last resort, never a displacement).
            if not ocr_tables and not all_tables:
                from scientific_parser.perception.tables.detector import (
                    detect_chatty_list_tables,
                )

                all_tables.extend(detect_chatty_list_tables(ocr_text))
        except Exception as e:
            log.warning("ocr_text strategy failed p%s: %s", page_num, e)
    # Strategy 5+6: camelot (lattice + stream) — run always, dedup handles
    # overlap. Skipped under raster_only: camelot reads the TEXT layer,
    # which raster-only pages lack by definition (det48).
    if not raster_only:
        try:
            all_tables.extend(detect_tables_camelot(pdf_path, page_num))
        except Exception as e:
            log.warning("camelot strategy failed p%s: %s", page_num, e)
    deduped, suppressed = _dedup_tables(all_tables, return_suppressed=True)

    # BAND-CONFINED TABLE FILTER (iter-61): a "table" whose bbox lies
    # ENTIRELY in the top 10% or bottom 10% of the page is a running
    # header/footer block (masthead left/right line pairs read as 2-col
    # tables — measured 96ac3eb8: ASPET masthead y41-61 = 5-8% H re-
    # emitted as a pipe table AFTER the header stripper had already
    # removed the lines from body text; 7 absent-tests). Position alone:
    # a real data table never lives wholly inside a margin band.
    # Zero-area bboxes are EXEMPT (iter-71): an OCR-text table
    # (ocr_chatty_list, detect_tables_in_text) carries NO position
    # information — a position filter cannot judge it (the (0,0,0,0)
    # convention silently killed every OCR-text table here; the masthead
    # case had real y-extent, so it stays filtered).
    try:
        import fitz as _fz

        with _fz.open(pdf_path) as _doc:
            _H = _doc[page_num - 1].rect.height
        if _H > 0:
            deduped = [
                t
                for t in deduped
                if (t["bbox"][2] - t["bbox"][0]) * (t["bbox"][3] - t["bbox"][1]) <= 0
                or not (t["bbox"][3] <= _H * 0.10 or t["bbox"][1] >= _H * 0.90)
            ]
    except Exception as e:
        log.debug("band-table filter skipped: %s", e)

    # Strategy 7: line-grid re-carve for GLUED-NUMERIC tables (iter-58).
    # The TableAgent char-geometry grid (modal-K column cuts from raw char
    # x0) re-extracts the table bbox; it replaces the strategy's rows ONLY
    # when every gate holds — measured A/B over the tables sample:
    #   1. pathology present: >=1 cell carries >=2 standalone numeric
    #      tokens (values from different columns glued into one cell).
    #   2. pathology fixed: strictly fewer such cells after re-carve.
    #   3. new grid column-consistent (modal_ratio >= 0.8).
    #   4. row-count ratio <= 1.25 — stacked-cell layouts (53d3f304)
    #      double rows and mangle parens; block (iter44 pin fixture).
    #   5. shred-free (no adjacent cells rejoin into a layer word).
    #   6. numeric preservation: every old numeric token survives.
    #   7. ADJACENCY PRESERVATION (added after first G1 reject): every
    #      old-grid pair (label, single-numeric-value) that sits adjacent
    #      must stay adjacent somewhere in the new grid — the olmOCR table
    # scorer requires the value's immediate-left cell to be the label.
    #      Modal-K bucketing splits multi-word labels ('TAM 107') across
    #      two buckets, leaving an empty cell between label and value
    #      (0953927d 'Trego-HWWW' | '' | '9.6' flipped one gold test).
    _num_tok = re.compile(r"\b\d+\.\d+%?\b|\b\d+%?\b")

    _bare_decimal = re.compile(r"\d+\.\d+")

    def _glued_numeric_cells(cells):
        """Cells whose numerics indicate CROSS-COLUMN glue, not label
        content. Paren groups are config annotations ('(k=4, ℓ=5)'),
        not column values — strip them before counting (iter-87:
        '(k=4, ℓ=5)' labels scored glued on BOTH sides of the recarve,
        freezing the gate at 3>=3). A text-heavy cell (>=3 words)
        carrying a bare decimal is also glue residue ('discrete
        decoding 0.79 unbalanced ...')."""
        n = 0
        for row in cells:
            for c in row:
                txt = re.sub(r"\([^)]*\)", " ", str(c).replace("\n", " "))
                toks = _num_tok.findall(txt)
                words = [w for w in txt.split() if any(ch.isalpha() for ch in w)]
                if len(toks) >= 2 or (len(words) >= 3 and _bare_decimal.search(txt)):
                    n += 1
        return n

    def _label_value_pairs(cells):
        """Adjacent (immediate-left, cell) pairs where the right cell is a
        single numeric token and the left cell is a non-empty TEXTUAL
        label. A left cell carrying a bare decimal is glue residue
        ('discrete decoding 0.79 unbalanced'), not a semantic label —
        its pair is an artifact and must not veto a recarve (iter-87)."""
        pairs = set()
        for row in cells:
            vals = [str(c).strip() for c in row]
            for i in range(1, len(vals)):
                v, l = vals[i], vals[i - 1]
                if v and l and _num_tok.fullmatch(v) and not _bare_decimal.search(l):
                    pairs.add((l, v))
        return pairs

    try:
        from scientific_parser.perception.tables.table_agent import TableAgent

        _ag = TableAgent(pdf_path, page_num, 150.0)
        _sc = 150.0 / 72.0
        for _t in deduped:
            _glue_old = _glued_numeric_cells(_t.get("rows", []))
            # RAGGED trigger (iter-87): sub-line bucket slicing can leave
            # label-fragment rows with a different cell count than the
            # data rows (f45a188e: header 8 cells vs data 10, a label
            # fragment + its value glued into one cell). No glued-numeric
            # pair exists, but the carve is still inconsistent — let the
            # char-grid recarve (column-major) compete under the same
            # protective gates.
            _ragged_old = len({len(r) for r in _t.get("rows", []) if r}) > 1
            if _glue_old == 0 and not _ragged_old:
                continue  # no pathology — leave proven carving alone
            _b = _t.get("bbox")
            if not _b:
                continue
            _pack = _ag._grid_candidate((_b[0] * _sc, _b[1] * _sc, _b[2] * _sc, _b[3] * _sc))
            if not _pack:
                continue
            _grid, _k, _mr = _pack
            if _mr < 0.8 or _ag._is_layer_shredded(_grid):
                continue
            if len(_grid) > 1.25 * max(1, len(_t.get("rows", []))):
                continue  # sub-line split — wrong tool for stacked cells
            _ragged_new = len({len(r) for r in _grid if r}) > 1
            if _glue_old > 0 and _glued_numeric_cells(_grid) >= _glue_old:
                continue  # didn't fix the pathology
            if _glue_old == 0 and _ragged_new:
                continue  # ragged-triggered: recarve must be consistent
            _old_txt = " ".join(str(c) for r in _t["rows"] for c in r if c)
            _old_nums = set(_num_tok.findall(_old_txt))
            _new_txt = " ".join(str(c) for r in _grid for c in r if c)
            if not all(n in _new_txt for n in _old_nums):
                continue  # value loss — never
            _old_pairs = _label_value_pairs(_t["rows"])
            _lost_pairs = _old_pairs - _label_value_pairs(_grid)
            if _lost_pairs:
                continue  # breaks already-correct label→value adjacency
            _t["rows"] = _grid
            _t["strategy"] = _t.get("strategy", "unknown") + "+linegrid"
            _t["recarved"] = True
    except Exception as e:
        log.warning("linegrid recarve failed p%s: %s", page_num, e)
    # LABEL-WRAP SPLIT (iter-113, 904a1b4e pg141): a wrapped row label's
    # continuation word lands in the first value column and glues to its
    # number ('anos 31,0') when a strategy buckets the short continuation
    # line into the value column. Signature: data row whose first
    # non-empty cell after the label reads '<lowercase-word> <number>'
    # while every later non-empty cell is a pure number — move the word
    # back onto the label.
    _lw_num = re.compile(r"^[-+\d.,%€$‰]+$")
    _lw_pat = re.compile(r"^([a-zà-ÿ]{2,})\s+([-+\d.,%€$‰]+)$")
    for t in deduped:
        for row in t.get("rows", []):
            ne = [(i, c) for i, c in enumerate(row) if str(c).strip()]
            if len(ne) < 3:
                continue
            li, label = ne[0]
            vi, val = ne[1]
            m = _lw_pat.match(str(val).strip())
            if (
                m
                and vi > li
                and all(_lw_num.match(str(c).strip()) for _j, c in ne[2:])
                and not str(label).rstrip().endswith((".", ":"))
            ):
                row[li] = f"{str(label).rstrip()} {m.group(1)}"
                row[vi] = m.group(2)
    # Rebuild markdown for all tables via multi-row header builder.
    # iter-119a: page header-line words per table bbox feed the fused-
    # header split (text always from the layer; structure from geometry).
    _pg_words = None
    if pdf_path:
        try:
            import fitz as _fz

            _pg = _fz.open(pdf_path)
            _p_i = (page_num or 1) - 1
            if 0 <= _p_i < len(_pg):
                _pg_words = _pg[_p_i].get_text("words")
            _pg.close()
        except Exception:
            _pg_words = None
    for t in deduped:
        _hw = None
        if _pg_words and t.get("bbox"):
            try:
                bx0, by0, bx1, _by1 = t["bbox"]
                _hw = [
                    (w[0], w[2], w[4])
                    for w in _pg_words
                    if w[1] >= by0 - 2
                    and w[3] <= by0 + 28
                    and w[0] >= bx0 - 6
                    and w[2] <= bx1 + 6
                    and w[4].strip()
                ]
            except Exception:
                _hw = None
        # P2.1 header-band recovery: when the grid starts at DATA, the
        # typeset header line above the bbox never entered any strategy's
        # rows — pull it in as the header row(s) before the build so both
        # the md emission and the HTML twin carry it (scorer top_heading
        # probes are column-anchored on row 0).
        if _pg_words and t.get("bbox") and t.get("rows"):
            try:
                from scientific_parser.perception.tables.detector import (
                    recover_header_band,
                )

                _new_rows, _rw = recover_header_band(t["rows"], t["bbox"], _pg_words)
                if _rw is not None:
                    t["rows"] = _new_rows
                    t["header_band_recovered"] = True
                    if _rw:
                        # the recovered line's own words are the split
                        # oracle for the fused header cells
                        _hw = [w for ln in _rw for w in ln]
            except Exception as e:
                log.debug("header-band recovery skipped: %s", e)
        t["markdown"] = build_table_markdown(
            t["rows"], t.get("strategy", "unknown"), header_words=_hw
        )
        # HTML twin preserves multi-row headers for checkers that parse
        # <thead>/<th> (olmOCR bench HTML path) — markdown keeps row 0 only.
        try:
            from scientific_parser.perception.tables.detector import build_table_html

            t["html"] = build_table_html(t["rows"]) if t["markdown"] else ""
        except Exception:
            t["html"] = ""
    # VALIDITY FALLBACK (iter-67, b8d3ad0f): a winner that builds EMPTY
    # markdown means the region would emit NOTHING even though a valid
    # suppressed alternative exists (dedup's more-rows-wins kept a 12-row
    # ragged pdfplumber carve over the 11-row VALID positional grid; the
    # winner then died on inconsistent_column_count — 5 gold 'No tables
    # found'). Reinstate the best suppressed alternative that builds.
    final_tables = []
    for t in deduped:
        if t["markdown"]:
            final_tables.append(t)
            continue
        _tb = t["bbox"]
        _cands = [
            s
            for s in suppressed
            if s is not t
            and not s.get("markdown")
            and _bbox_overlap_area(_tb, s["bbox"]) > 0.6 * max(1.0, _bbox_area(_tb))
        ]
        for s in _cands:
            s["markdown"] = build_table_markdown(s["rows"], s.get("strategy", "unknown"))
        _cands = [s for s in _cands if s["markdown"]]
        if _cands:
            best = max(_cands, key=lambda s: len(s["rows"]))
            try:
                from scientific_parser.perception.tables.detector import build_table_html

                best["html"] = build_table_html(best["rows"])
            except Exception:
                best["html"] = ""
            final_tables.append(best)
            log.info(
                "table fallback: reinstated %s carve (%d rows) over empty-markdown winner",
                best.get("strategy"),
                len(best["rows"]),
            )
    # IMG2TABLE LAST RESORT (R16, 2026-09-13): fires ONLY when every strategy
    # above found nothing — the measured no-table-found population (36 tests/9
    # PDFs at floor; img2table's OpenCV borderless detector recovers +5:
    # 7e850300 x3 + c6673ff6 x2, splice-probe 2026-09-13). Zero arbitration
    # risk by construction (empty finals). Graceful skip on any failure.
    # det48: skipped under raster_only — img2table reads the RASTER itself;
    # on raster-only pages the raster_ocr strategy (4e) is the primary and
    # img2table only adds chart/figure-label false grids (odlbench 0056).
    if not final_tables and not raster_only:
        try:
            from img2table.document import PDF as _I2T_PDF

            _i2t = _I2T_PDF(pdf_path, pages=[page_num - 1])
            _detected = (
                _i2t.extract_tables(implicit_rows=True, borderless_tables=True, min_confidence=50)
                or {}
            )
            for _tbl in _detected.get(page_num - 1, []):
                _rows = []
                for _r in _tbl.df.itertuples(index=False):
                    _rows.append(
                        [
                            ""
                            if v is None or (isinstance(v, float) and v != v)
                            else str(v).replace("|", "\\|").replace("\n", " ").strip()
                            for v in _r
                        ]
                    )
                # WORD-BACK FILL (2026-09-14): scanned pages (d896f9ae pg4
                # family) detect the grid but img2table's born-digital
                # self-extraction returns None cells. Render the page at
                # img2table's internal scale (derived dynamically), read
                # words with rapidocr (deterministic ONNX), and assign
                # them to cell bboxes by center containment. Probe: 246/
                # 370 cells filled, all 5 gold cell-values present.
                _ne = sum(1 for r in _rows for c in r if c)
                _tot = sum(len(r) for r in _rows)
                if _tot and _ne / _tot < 0.2:
                    try:
                        import tempfile as _tf

                        import fitz as _fz

                        _doc = _fz.open(pdf_path)
                        _pg = _doc[page_num - 1]
                        _ih, _iw = _i2t.images[0].shape[:2]
                        _scale = _iw / max(1.0, _pg.rect.width)
                        _pix = _pg.get_pixmap(matrix=_fz.Matrix(_scale, _scale))
                        with _tf.NamedTemporaryFile(suffix=".png") as _tmpf:
                            _pix.save(_tmpf.name)
                            from rapidocr import RapidOCR as _ROCR

                            _res = _ROCR()(_tmpf.name)
                        _words = []
                        for _box, _txt, _conf in zip(_res.boxes, _res.txts, _res.scores):
                            _xs = [p[0] for p in _box]
                            _ys = [p[1] for p in _box]
                            _words.append((min(_xs), min(_ys), max(_xs), max(_ys), _txt))
                        _doc.close()

                        def _wb_cell(_cell):
                            _b = _cell.bbox
                            _hit = [
                                w
                                for w in _words
                                if _b.x1 <= (w[0] + w[2]) / 2 <= _b.x2
                                and _b.y1 <= (w[1] + w[3]) / 2 <= _b.y2
                            ]
                            _hit.sort(key=lambda w: (round(w[1] / 10), w[0]))
                            return " ".join(w[4] for w in _hit).replace("|", "\\|").strip()

                        _wb_rows = [
                            [_wb_cell(c) for c in _cells] for _r, _cells in _tbl.content.items()
                        ]
                        _wb_rows = [r for r in _wb_rows if any(c for c in r)]
                        _wb_ne = sum(1 for r in _wb_rows for c in r if c)
                        _wb_tot = sum(len(r) for r in _wb_rows)
                        if _wb_tot and _wb_ne / _wb_tot > (_ne / _tot if _tot else 0.0):
                            _rows = _wb_rows
                    except Exception:
                        log.debug("img2table word-back unavailable", exc_info=True)
                _rows = [r for r in _rows if any(c for c in r)]
                if len(_rows) < 2:
                    continue
                # PROSE GUARD (det47, run-52 census: odlbench 033/034/036
                # prose-shatter FPs — justified sentence fragments carved
                # into a 3-col grid). Predicate + measured keep-wins in
                # looks_like_prose_shatter (7e850300 '10.9888' numeric,
                # c6673ff6 'A/RES/66/2' code-like).
                try:
                    from scientific_parser.perception.tables.detector import (
                        looks_like_prose_shatter,
                    )

                    if looks_like_prose_shatter(_rows):
                        continue
                except Exception:
                    log.debug("prose guard unavailable", exc_info=True)
                _md = build_table_markdown(_rows, "img2table_lastresort")
                if not _md:
                    continue
                try:
                    from scientific_parser.perception.tables.detector import build_table_html

                    _html = build_table_html(_rows)
                except Exception:
                    _html = ""
                final_tables.append(
                    {
                        "rows": _rows,
                        "markdown": _md,
                        "html": _html,
                        "strategy": "img2table_lastresort",
                        # Pixel-space bbox is not page-point comparable and
                        # no arbitration competitors exist (finals were
                        # empty) — full-page region is the honest bound.
                        "bbox": (0.0, 0.0, 612.0, 792.0),
                    }
                )
                log.info(
                    "img2table last-resort: %dx%d grid on page %d",
                    len(_rows),
                    len(_rows[0]) if _rows else 0,
                    page_num,
                )
        except Exception:
            log.debug("img2table last-resort unavailable", exc_info=True)

    # EXP-3 cell x-clip: geometry-split jammed cells ('9 M-PSI/06' in one
    # cell, 033c1efd pg5 class) — sibling-row column consensus, moves only
    # into EMPTY neighbor cells, strategy-agnostic. Emissions rebuilt for
    # changed tables.
    try:
        from scientific_parser.perception.tables.detector import (
            split_jammed_cells,
        )

        _rows_before = {id(t): t.get("rows") for t in final_tables if isinstance(t, dict)}
        split_jammed_cells(pdf_path, page_num, final_tables)
        for t in final_tables:
            if (
                isinstance(t, dict)
                and _rows_before.get(id(t)) is not None
                and _rows_before[id(t)] is not t.get("rows")
            ):
                t["markdown"] = build_table_markdown(t["rows"], t.get("strategy", "unknown"))
                try:
                    from scientific_parser.perception.tables.detector import (
                        build_table_html,
                    )

                    t["html"] = build_table_html(t["rows"]) if t["markdown"] else ""
                except Exception:
                    t["html"] = ""
    except Exception:
        log.debug("cell x-clip pass failed", exc_info=True)

    # Cell-level glyph map (EXP det16): the xval crosscheck verifies a
    # systematic frm->to glyph substitution on the fitz LAYER words; the
    # flowed-text pass already applies it to body text, but the tables
    # lane re-reads the same layer — so cells inside final tables keep
    # the corruption. take_page_glyph_map() pops the map the producer
    # stored for this page (single-shot, mtime-keyed) and we re-apply
    # the identical surgical substitution per cell. No map (None) or
    # zero applications = exact no-op.
    try:
        from scientific_parser.routing.glyph_map import (
            surgical_word_sub,
            take_page_glyph_map,
        )

        _gm = take_page_glyph_map(pdf_path, page_num)
        if _gm and final_tables:
            _frm, _to, _words = _gm
            _changed = 0
            for t in final_tables:
                if not isinstance(t, dict):
                    continue
                rows = t.get("rows")
                if not rows:
                    continue
                _t_applied = 0
                for row in rows:
                    if not isinstance(row, list):
                        continue
                    for ci, cell in enumerate(row):
                        if not (isinstance(cell, str) and _frm and _frm in cell.lower()):
                            continue
                            new_cell, n = surgical_word_sub(
                                cell, [_words] if isinstance(_words, str) else _words, _frm, _to
                            )
                            if n:
                                row[ci] = new_cell
                                _t_applied += n
                if _t_applied:
                    t["markdown"] = build_table_markdown(t["rows"], t.get("strategy", "unknown"))
                    try:
                        from scientific_parser.perception.tables.detector import (
                            build_table_html,
                        )

                        t["html"] = build_table_html(t["rows"]) if t["markdown"] else ""
                    except Exception:
                        t["html"] = ""
                    t["strategy"] = (t.get("strategy") or "unknown") + "+glyph_map"
                    _changed += 1
            if _changed:
                log.info(
                    "[glyph_map] applied verified substitution to cells of %d table(s) on page %s",
                    _changed,
                    page_num,
                )
    except Exception:
        log.debug("cell glyph-map pass failed", exc_info=True)
    # LEADER-PAGE-LIST VETO (det47, run-52): TOC/index leader lists
    # (['1. Changing Practices in Disaster Management' | '7']) survive
    # every grid guard because they ARE clean 2-col grids — but GT keeps
    # them as plain text, so a converted table is pure NID deletion
    # damage (odlbench doc-016: 29-row TOC as positional FP). Veto shape
    # measured with zero false-veto exposure (see helper docstring).
    try:
        from scientific_parser.perception.tables.detector import (
            looks_like_leader_page_list,
        )

        _pre = len(final_tables)
        final_tables = [
            t for t in final_tables if not looks_like_leader_page_list(t.get("rows") or [])
        ]
        if len(final_tables) < _pre:
            log.info(
                "leader-page-list veto: dropped %d table(s) on page %s",
                _pre - len(final_tables),
                page_num,
            )
    except Exception:
        log.debug("leader veto unavailable", exc_info=True)
    # EQUATION-ROWS VETO (det49, run-56 census): aligned equations carved
    # into grids emit as pipe tables, which strips the source line from
    # page text so the inline Unicode->LaTeX lane never runs — GT LaTeX
    # probes then find nothing (arxiv class-A: up to 267 tests, 84 PDFs).
    # Stats/spec tables survive: digit-attached operators and checkmarks
    # are not flagged (see helper docstring for measured kill rates).
    try:
        from scientific_parser.perception.tables.detector import (
            looks_like_equation_rows,
        )

        _pre = len(final_tables)
        final_tables = [
            t for t in final_tables if not looks_like_equation_rows(t.get("rows") or [])
        ]
        if len(final_tables) < _pre:
            log.info(
                "equation-rows veto: dropped %d table(s) on page %s",
                _pre - len(final_tables),
                page_num,
            )
    except Exception:
        log.debug("equation veto unavailable", exc_info=True)
    return final_tables


def _bbox_area(b):
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))


def _bbox_overlap_area(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def repair_encoding_decimals(text):
    """Repair European decimal notation in extracted text.

    Some European journal PDFs use custom font encodings where the decimal
    point renders as degree sign (°), middle dot (·), multiplication (×),
    or replacement character (U+FFFD). This function fixes these in
    numeric contexts: digit + artifact + digit → digit.digit.

    Runs on ALL text (not just table cells) because these artifacts appear
    in body text, captions, and abstracts too.
    """
    return re.sub(
        r"(\d)[\u00b0\ufffd\u00b7\u00d7](\d)",
        r"\1.\2",
        text,
    )


# P0-3: Display equation detection for layout-aware OCR
# =============================================================================

# Math operator characters that indicate equation blocks


# mask_math_regions/unmask_math_regions live in shared.py (one guard,
# two call sites: preserve_typography + math_extractor.replace_math_in_text).


def preserve_typography(pdf_path, page_num, text):
    """Wrap formatted text spans with markdown markers.

    Detects via fitz font flags:
      italic → *text*  (species names, journal titles)
      bold → **text**  (headings, emphasis)
      monospace → `text`  (code, data)
      superscript → $^{text}$  (isotope mass numbers, exponents)

    Conservative: only wraps spans ≥8 chars that appear exactly once in text
    to avoid false-positive substring matches.

    **Phase 1 — within-line merging**: PDFs often split a single bold/italic
    run into multiple adjacent spans (font subsetting, kerning adjustments).
    Without merging, the output is fragmented: ``**The** **MgO** **of**
    **lavas is**``. With merging: ``**The MgO of lavas is**``.

    **Phase 2 — cross-line merging** (NEW): bold runs that span line breaks
    (common in scientific abstracts) get merged into single **...** blocks.
    Triggered when consecutive lines have (a) same flag signature, (b) Y-gap
    < 1.5x median line height. Eliminates fragmentation across paragraphs.
    """
    import fitz

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]

        # Collect per-span data: (text, flags, font, x0, y0, x1, y1, line_id)
        # line_id = unique per line, used for cross-line merge.
        all_spans = []
        line_id = 0
        line_y_centers = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                line_id += 1
                line_bbox = line.get("bbox", [0, 0, 0, 0])
                line_y_centers.append((line_bbox[1] + line_bbox[3]) / 2)
                for span in line.get("spans", []):
                    s_text = span.get("text", "")
                    if not s_text.strip():
                        continue
                    sbbox = span.get("bbox", [0, 0, 0, 0])
                    all_spans.append(
                        {
                            "text": s_text,
                            "flags": span.get("flags", 0),
                            "font": span.get("font", ""),
                            "x0": sbbox[0],
                            "y0": sbbox[1],
                            "x1": sbbox[2],
                            "y1": sbbox[3],
                            "line_id": line_id,
                            "line_y": (sbbox[1] + sbbox[3]) / 2,
                        }
                    )

        if not all_spans:
            doc.close()
            return text

        # Compute median line height for cross-line merge threshold
        sorted_ys = sorted(line_y_centers)
        if len(sorted_ys) >= 2:
            line_heights = [
                sorted_ys[i + 1] - sorted_ys[i]
                for i in range(len(sorted_ys) - 1)
                if 0 < sorted_ys[i + 1] - sorted_ys[i] < 50
            ]
            median_lh = sorted(line_heights)[len(line_heights) // 2] if line_heights else 12.0
        else:
            median_lh = 12.0
        y_merge_threshold = median_lh * 1.5

        # Phase 1: within-line merging — group consecutive spans in same
        # line with identical (flags, font) signature.
        line_groups = {}  # line_id -> list of (sig, merged_text)
        for span in all_spans:
            lid = span["line_id"]
            sig = (span["flags"], span["font"])
            if lid not in line_groups:
                line_groups[lid] = []
            if line_groups[lid] and line_groups[lid][-1][0] == sig:
                # Same signature as previous — append to current group
                line_groups[lid][-1][1] += span["text"]
            else:
                line_groups[lid].append([sig, span["text"]])

        # Build per-line formatted runs (sig, merged_text, line_y)
        # Mono-wrap gate: when most of the page is set in a mono-flagged font
        # (old books typeset in Courier-class faces), backticking every line is
        # wrong AND score-destructive (bench normalizer strips bold/italic
        # markers but NOT backticks). Only wrap mono when it's a minority —
        # real code/data spans inside prose.
        total_chars = sum(len(s["text"]) for s in all_spans)
        mono_chars = 0
        for span in all_spans:
            f_, fo_ = span["flags"], span["font"]
            if bool(f_ & 8) or "courier" in fo_.lower() or "mono" in fo_.lower():
                mono_chars += len(span["text"])
        mono_minority = total_chars == 0 or mono_chars / total_chars < 0.25

        runs = []
        for lid in sorted(line_groups.keys()):
            # line_y from first span in this line
            line_spans = [s for s in all_spans if s["line_id"] == lid]
            if not line_spans:
                continue
            line_y = line_spans[0]["line_y"]
            for sig, merged_text in line_groups[lid]:
                stripped = merged_text.strip()
                if len(stripped) < MIN_SPAN_CHARS:
                    continue
                flags, font = sig
                is_italic = bool(flags & 2) or "italic" in font.lower() or "oblique" in font.lower()
                is_bold = bool(flags & 16) or "bold" in font.lower() or "black" in font.lower()
                is_mono = bool(flags & 8) or "courier" in font.lower() or "mono" in font.lower()
                if is_mono and not mono_minority:
                    is_mono = False
                if not (is_italic or is_bold or is_mono):
                    continue
                # Determine format type
                if is_mono:
                    fmt = "mono"
                elif is_bold and is_italic:
                    fmt = "bold_italic"
                elif is_bold:
                    fmt = "bold"
                elif is_italic:
                    fmt = "italic"
                else:
                    continue
                runs.append(
                    {
                        "text": stripped,
                        "fmt": fmt,
                        "line_id": lid,
                        "line_y": line_y,
                    }
                )

        # Phase 2: cross-line merging — adjacent runs with same fmt AND
        # Y-gap < 1.5x median line height get merged.
        merged_runs = []
        for run in runs:
            if (
                merged_runs
                and merged_runs[-1]["fmt"] == run["fmt"]
                and abs(run["line_y"] - merged_runs[-1]["line_y"]) <= y_merge_threshold
            ):
                # Merge into previous run
                merged_runs[-1]["text"] += " " + run["text"]
                merged_runs[-1]["line_y"] = run["line_y"]
            else:
                merged_runs.append(dict(run))

        # Build wrapped text per merged run
        formatted = []
        for run in merged_runs:
            t = run["text"]
            fmt = run["fmt"]
            if fmt == "sup":
                wrapped = f"$^{{{t}}}$"
            elif fmt == "mono":
                wrapped = f"`{t}`"
            elif fmt == "bold_italic":
                wrapped = f"***{t}***"
            elif fmt == "bold":
                wrapped = f"**{t}**"
            elif fmt == "italic":
                wrapped = f"*{t}*"
            else:
                continue
            if wrapped != t:
                formatted.append((t, wrapped))

        doc.close()

        # Apply wraps conservatively — only if span appears exactly once.
        # Sort longest-first so longer matches win over shorter substrings.
        # MATH MASK (2026-08-16, flip-diff 2503.06703_pg4): a math-font
        # sub/superscript span ('k,m' in CMMI7) matches the sup-wrap
        # geometry test; wrapping it inside existing $...$ corrupts the
        # delimiter pairing and downstream cleaners drop the equation
        # tail (measured: '$\bm{\varphi}_{$k,m $' → '\mathbb{C}^{N}'
        # lost). Mask $...$/$$...$$ regions before ANY wrap, restore after.
        formatted.sort(key=lambda x: len(x[0]), reverse=True)
        for original, wrapped in formatted:
            count = text.count(original)
            if count == 1:
                masked, spans_kept = mask_math_regions(text)
                # Heading lines are already promoted ('### 2.3 ...') —
                # injecting ** markers inside them corrupts the heading
                # ('### 2.3 **Hidden Data Display**'). Mask whole lines
                # starting with '#', same discipline as math regions.
                _lines = masked.split("\n")
                _hl = [(i, ln) for i, ln in enumerate(_lines) if ln.lstrip().startswith("#")]
                _sent = ["\x02HLM%d\x03" % i for i, _ in _hl]
                for (i, ln), s in zip(_hl, _sent):
                    _lines[i] = s
                masked = "\n".join(_lines)
                if original in masked:
                    masked = masked.replace(original, wrapped, 1)
                for (i, ln), s in zip(_hl, _sent):
                    masked = masked.replace(s, ln, 1)
                text = unmask_math_regions(masked, spans_kept)

        return text
    except Exception as e:
        log.warning("post-processor failed: %s", e)
        return text


# =============================================================================
# P2-10: Intelligent dehyphenation
# =============================================================================

# Hyphenated compounds that must NOT be joined (geochemistry whitelist)
PRESERVE_HYPHENS = {
    # Element-element compounds
    "Fe-bearing",
    "Fe-rich",
    "Fe-poor",
    "Fe-free",
    "U-Pb",
    "Sm-Nd",
    "K-Ar",
    "Ar-Ar",
    "Rb-Sr",
    "Lu-Hf",
    "Re-Os",
    "La-Ce",
    "Pb-Pb",
    "Mg-Fe",
    "Ca-Mg",
    "Si-Al",
    "Na-K",
    "C-O-H",
    "Fe-Ti",
    "Cr-Ni",
    # Mineral descriptors
    "low-K",
    "high-K",
    "medium-K",
    "high-Mg",
    "low-Mg",
    "high-Al",
    "low-Al",
    "Mg-rich",
    "K-rich",
    "Si-rich",
    # Temperature/pressure
    "high-T",
    "low-T",
    "high-P",
    "low-P",
    "T-P",
    "P-T",
    # Size fractions
    "coarse-grained",
    "fine-grained",
    "medium-grained",
    # Common geochemistry terms
    "H2O-saturated",
    "fluid-saturated",
    "non-conservative",
    "non-ideal",
    # ICP-MS type instruments
    "LA-ICP-MS",
    "HR-ICP-MS",
    "MC-ICP-MS",
    "SIMS",
    "TIMS",
    "EPMA",
}


_SUP_DIGITS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
}
_SUB_DIGITS = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
}


def _lone_scripts_to_unicode(text):
    """Convert math blocks that are ONLY a single-digit sub/superscript
    (`$^{2}$`, `$_{3}$`, optional leading minus) to unicode glyphs (², ₂).

    These are prose typography — unit superscripts (kg/m², cm³), isotope
    subscripts — and the bench gold keeps them as unicode verbatim.
    Multi-glyph scripts (x^{m-1}, P_{\\Delta_n}) stay LaTeX.
    """

    def _sup(m):
        d = m.group(1).lstrip("-")
        return ("⁻" if m.group(1).startswith("-") else "") + _SUP_DIGITS[d]

    def _sub(m):
        d = m.group(1).lstrip("-")
        return ("₋" if m.group(1).startswith("-") else "") + _SUB_DIGITS[d]

    text = re.sub(r"\$\^\{(-?\d)\}\$", _sup, text)
    text = re.sub(r"\$_\{(-?\d)\}\$", _sub, text)
    return text


def _normalize_typographic_dashes(t):
    """U+2212/U+2013 → '-'; U+2014 → '-' only when digit-adjacent.

    Letter-joined em-dashes ('Process—A') are document typography kept
    verbatim (gold expects them, 028649 iter69 pin); digit-adjacent ones
    are range notation ('1973—79') normalized for exact-match recall.
    """
    t = t.replace("\u2212", "-").replace("\u2013", "-")
    return re.sub(
        r"\u2014",
        lambda m: (
            "-"
            if (t[max(0, m.start() - 1)].isdigit() or t[m.end() : m.end() + 1].isdigit())
            else m.group(0)
        ),
        t,
    )


def dehyphenate(text):
    """Join words broken across lines by hyphenation.

    PDF extraction often breaks words at line endings:
      meta-\nmorphism → metamorphism
      frac-\ntionation → fractionation

    But PRESERVES legitimate hyphenated compounds:
      Fe-bearing, U-Pb, Sm-Nd, K-Ar

    Algorithm:
    1. Find pattern: lowercase-\n lowercase (word broken at line end)
    2. Check if the joined word is in PRESERVE_HYPHENS → keep hyphen
    3. Otherwise join: remove hyphen and newline
    """

    def _join_hyphen(match):
        prefix = match.group(1)
        suffix = match.group(2)
        joined = prefix + "-" + suffix
        # Check whitelist
        if joined in PRESERVE_HYPHENS:
            return joined
        # Check if it looks like an element-symbol compound (A-Bb pattern)
        if re.match(r"^[A-Z][a-z]?-[A-Z]", joined):
            return joined  # Preserve element compounds
        # COMPOUND GATE (iter-85, 013a3686): when BOTH parts are English
        # words and the fused form is not, the hyphen is a real compound
        # ('racist-'+'free' → 'racist-free'), not a line-wrap break —
        # gold keeps 'racist-free'. Genuine hyphenation breaks have a
        # non-word fragment ('frac'+'tionation', 'identi'+'fied') and
        # still fuse. The regex captures only the prefix's LAST letter —
        # read the full word back from the enclosing text ('co-'+'operation'
        # must not read prefix='o', a dictionary word).
        mstart = match.start(1)
        wstart = mstart
        while wstart > 0 and (
            text[wstart - 1].isalpha() or text[wstart - 1] in "\u00c6\u00e6\ufffd"
        ):
            wstart -= 1
        full_prefix = text[wstart:mstart] + prefix
        try:
            _wl = _english_wordlist()
        except Exception:
            _wl = None
        if (
            _wl is not None
            and full_prefix.lower() in _wl
            and suffix.lower() in _wl
            and (full_prefix + suffix).lower() not in _wl
        ):
            return prefix + "-" + suffix
        # Join: remove hyphen
        return prefix + suffix

    # Pattern: word ending with hyphen + newline + word starting with
    # lowercase. Trailing spaces after the hyphen allowed ('par- \n'
    # pdftotext padding, 027c3905 8d); one OPTIONAL blank line between
    # (column reorder emitters insert a blank line between blocks — a
    # word wrapped across the block boundary 'con-'/'cerning' stays
    # broken otherwise, 0514f667 order-test 9c). Tail must be alphabetic;
    # a CAPITAL tail (Boundary-\nLayer) is a real compound.
    text = re.sub(
        r"([a-z\uFFFD\u00C6\u00E6])-[ \t]*\n(?:[ \t]*\n)?[ \t]*([a-z\uFFFD]+)",
        _join_hyphen,
        text,
    )
    # PAREN-GLUE (iter-76, 027c3905 87): a line ending with a word whose
    # next line starts '(' is a wrapped percent/citation ('Saudi Arabia
    # \n(53%)') — the source has NO space there; gold matches at
    # max_l_dist 0 require 'Arabia(53%)' glued. DIGIT-CONTENT ONLY
    # (iter-85): gold census 87:2 favors a SPACE when the paren content
    # starts with a letter ('enhancements (yellow lines)',
    # 'substances (green TNT)', 081875a1 p16); percents/citations are
    # digit-led.
    text = re.sub(r"([A-Za-z0-9\]\)])[ \t]*\n\((?=\d)", r"\1(", text)
    return text


# =============================================================================
# P2-9: Chemical/mineral domain normalization
# =============================================================================

# Unicode subscript/superscript digit maps


def normalize_chemical_notation(text):
    """Convert ASCII chemical formulas to Unicode subscript notation.

    SiO2 → SiO₂, Fe2+ → Fe²⁺, Al2O3 → Al₂O₃

    Conservative: only converts well-known oxide formulas and
    explicit oxidation state notations. Leaves unknown formulas unchanged.
    """
    # Convert known oxide formulas
    for ascii_form, unicode_form in OXIDE_FORMULAS.items():
        # Use word boundary to avoid partial matches
        text = re.sub(
            r"\b" + re.escape(ascii_form) + r"\b",
            unicode_form,
            text,
        )

    # Convert oxidation states: Fe2+ → Fe²⁺, Fe3+ → Fe³⁺, S6+ → S⁶⁺
    def _fix_oxidation(match):
        elem = match.group(1)
        charge = match.group(2)
        sign = match.group(3) or ""
        if charge in ("2", "3", "4", "5", "6", "7", "1"):
            charge_unicode = charge.translate(SUPERSCRIPT_DIGITS)
            sign_unicode = sign.translate(SUPERSCRIPT_DIGITS) if sign else ""
            return f"{elem}{charge_unicode}{sign_unicode}"
        return match.group(0)

    text = re.sub(
        r"\b([A-Z][a-z]?)(\d)([+-]?)(?=\s|[,;.\)\]\}]|$)",
        _fix_oxidation,
        text,
    )

    # Preserve Mg# exactly (don't let other processors corrupt it)
    # Already handled by the word-boundary regex above

    return text


def repair_citation_splits(text):
    """Repair citations split across line breaks.

    PDF extraction can break: 'Johnson [12]-\net al.' into separate fragments.
    This function rejoins citation brackets with their context.
    """
    # Rejoin: word-\n[NN] → word [NN]
    text = re.sub(
        r"(\w)-\n\[(\d+)\]",
        r"\1 [\2]",
        text,
    )
    # Rejoin: [NN]-\nword → [NN] word
    text = re.sub(
        r"\[(\d+)\]-\n(\w)",
        r"[\1] \2",
        text,
    )
    return text


def preserve_dois(text):
    """Ensure DOI strings are not broken or corrupted.

    Joins DOIs split across lines: 10.1234/abc-\ndef → 10.1234/abcdef
    Wraps DOIs in markdown link format for machine readability.
    """
    # Join DOIs split across lines
    text = re.sub(
        r"(10\.\d{4,}/\S*)-\n(\S)",
        r"\1\2",
        text,
    )
    return text


# =============================================================================
# P2-11: Bibliography mode (lighter post-processing for References)
# =============================================================================

BIBLIOGRAPHY_MARKERS = [
    "References",
    "REFERENCES",
    "Literature Cited",
    "Bibliography",
    "BIBLIOGRAPHY",
    "参考文献",
    "引用文献",
]


def detect_bibliography_section(text):
    """Detect if text contains a bibliography/references section.

    Returns (before_text, bibliography_text) split, or (text, None)
    if no bibliography section detected.
    """
    for marker in BIBLIOGRAPHY_MARKERS:
        # Look for marker as a heading (own line, short line)
        pattern = re.compile(
            r"^\s*" + re.escape(marker) + r"\s*$",
            re.MULTILINE,
        )
        match = pattern.search(text)
        if match:
            pos = match.start()
            return text[:pos].rstrip(), text[pos:]
    return text, None


def process_bibliography(bib_text):
    """Lighter post-processing for bibliography sections.

    Bibliographies have different formatting than body text:
    - Author names with initials (don't fix "hyphenation")
    - Journal volume/page numbers (don't apply chemical fixes)
    - DOI strings (preserve exactly)

    This function applies only:
    - Dehyphenation (word breaks at line ends)
    - DOI preservation
    - Citation repair
    """
    # Join hyphenated line breaks (common in reference formatting)
    bib_text = dehyphenate(bib_text)
    # Preserve DOI formatting
    bib_text = preserve_dois(bib_text)
    # Do NOT apply chemical formula fixes, symbol fixes, etc.
    return bib_text


# =============================================================================
# #5: Footnote reconstruction
# =============================================================================


def detect_section_hierarchy(pdf_path, page_num, text):
    """Detect section heading hierarchy from font size distribution.

    Conservative heading detection — requires MULTIPLE signals to avoid
    promoting body text:
    1. Font size significantly larger than median (>1.3×)
    2. Short line (< 80 chars — headings are concise)
    3. No sentence-ending punctuation (headings don't end with . ! ?)
    4. Standalone line (not part of a flowing paragraph)
    5. Not inside a table cell
    """
    pd = _get_page_dict(pdf_path, page_num)
    if pd is None:
        return text
    try:
        spans_data = []
        for block in pd["dict"]["blocks"]:
            if block.get("type") != 0:
                continue
            for line_obj in block.get("lines", []):
                line_text = ""
                line_size = 0
                line_count = 0
                line_font_chars = {}
                line_bold_chars = 0
                _prev_x1 = None
                for span in line_obj.get("spans", []):
                    t = span.get("text", "")
                    if t.strip():
                        # WORD-GAP AWARE JOIN (det25a, odlbench
                        # 01030000000035 'BasisFields'): TeX kerns titles
                        # into multiple spans on ONE line; bare
                        # concatenation glues words. Insert a space when
                        # the horizontal gap between neighbouring spans
                        # exceeds ~0.25x the span size (inter-word gaps
                        # run ~0.3em; mid-word kern splits ~0).
                        if line_text and span.get("bbox") and _prev_x1 is not None:
                            _pgap = span["bbox"][0] - _prev_x1
                            _psize = span.get("size", 10.0) or 10.0
                            if _pgap > 0.25 * _psize:
                                line_text += " "
                        line_text += t
                        if span.get("bbox"):
                            _prev_x1 = span["bbox"][2]
                        else:
                            _prev_x1 = None
                        line_size += span.get("size", 10)
                        line_count += 1
                        font = span.get("font", "")
                        line_font_chars[font] = line_font_chars.get(font, 0) + len(t)
                        # PyMuPDF bold flag (bit 16) — structured weight
                        # evidence. Catches family-bold-is-medium naming
                        # (NimbusRomNo9L-Medi = LaTeX \textbf face) that
                        # name parsing reads as weight 500. LiteParse
                        # is_bold_item: descriptor flags before names.
                        if span.get("flags", 0) & 16:
                            line_bold_chars += len(t)
                if line_count > 0 and line_text.strip():
                    avg_size = line_size / line_count
                    # Dominant font by char count — the weight signal
                    # for the bold-same-size heading pass below.
                    dominant_font = (
                        max(line_font_chars, key=line_font_chars.get) if line_font_chars else ""
                    )
                    spans_data.append(
                        {
                            "text": line_text.strip(),
                            "size": avg_size,
                            "font": dominant_font,
                            "bold_share": line_bold_chars / max(1, len(line_text.strip())),
                            # Line bbox (x0, y0, x1, y1) — vertical
                            # leading evidence: a display title WRAPPED
                            # inside one text block keeps tight leading
                            # (gap <= ~0.35x size), while stacked
                            # separate headings are spaced apart.
                            "bbox": tuple(line_obj.get("bbox") or ()),
                        }
                    )

        if len(spans_data) < 5:
            return text

        sizes = sorted(s["size"] for s in spans_data)
        median_size = sizes[len(sizes) // 2]
        p90 = sizes[int(len(sizes) * 0.90)]

        # Size-separated headings need p90 >= 1.3x median. Headings set
        # BOLD at body size (books, reports) carry no size signal at all
        # — inert here, handled by the bold pass below (docling port).
        # Sparse-title fallback: when the heading cluster is <10% of
        # lines, p90 reads BODY size and the statistic goes blind. Walk
        # distinct sizes top-down instead, admitting levels that are
        # >=1.15x median AND keep cumulative share <=15% (docling size
        # clustering): a display title is an outlier by definition.
        # 'Chapter 2' at 1.14x stays excluded by the 1.15 floor.
        if p90 >= median_size * 1.3:
            heading_size = p90
        else:
            from collections import Counter

            _size_counts = Counter(round(s["size"], 1) for s in spans_data)
            heading_size = float("inf")
            _acc = 0
            for _s in sorted(_size_counts, reverse=True):
                if _s < median_size * 1.15:
                    break
                _n = _size_counts[_s]
                if _acc and _acc + _n > 0.15 * len(sizes):
                    break
                if _n > 0.15 * len(sizes):
                    break
                heading_size = _s - 0.05
                _acc += _n

        # QUOTE-NORMALIZED MATCHING (det25b, odlbench 01030000000133):
        # fitz spans carry typographic quotes (') while the extracted
        # text layer can hold straight ones (') for the same font —
        # title-to-line matches then miss on quote chars alone. All
        # promoter matching runs through _q(); the map is strictly 1:1
        # length-preserving, so match indices stay valid in the ORIGINAL
        # text and emitted headings keep original span fidelity.
        _QMAP = str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201e": '"',
            }
        )

        def _q(s):
            return s.translate(_QMAP)

        def _prev_terminal_ok(prev_clean, before=""):
            """Shared predecessor gate for all three promoters.

            A heading never sits inside a flowing paragraph, so the
            previous line must close a unit: terminal punctuation, a
            structural marker, or a heading-shaped line (allcaps, or a
            capitalized <=6-word line with no terminal punctuation —
            'Chapter 2' above 'Narratives in Chuj' is an unpromoted
            heading, not paragraph flow). Sentence periods may sit
            inside closing parens/quotes ('...transcription.)').

            `before` = the text up to (not including) the candidate
            line, for the caption-continuation check below.
            """
            _stripped = prev_clean.rstrip(")]\u201d\u2019\"'")
            if (
                not prev_clean
                or _stripped.endswith((".", "!", "?", ":", ";"))
                or prev_clean.startswith(("#", "===", "<!--", "|"))
            ):
                return True
            # CAPTION PREV (det23c, odlbench 111/154-class): a figure or
            # table caption closes a block even when its text runs on
            # (lowercase, no terminal punct — captions are labels, not
            # paragraph flow; the bold heading after one starts a new
            # block).
            if re.match(r"^(Figure|Fig\.?|Table|Tab\.?|Plate|Scheme)\s*\d", prev_clean):
                return True
            # CAPTION CONTINUATION (det29b, odlbench 111): captions wrap
            # across lines ('Figure 8.1: a) ... force vortex' above
            # 'measuring probes') — when the heading's direct prev is a
            # caption TAIL, the pattern sits one line further up.
            if before:
                _p2s = before[: before.rfind("\n")]
                _p2s = _p2s[: _p2s.rfind("\n")]
                _p2 = _p2s[_p2s.rfind("\n") + 1 :].strip()
                if _p2 and re.match(r"^(Figure|Fig\.?|Table|Tab\.?|Plate|Scheme)\s*\d", _p2):
                    return True
            if prev_clean.isupper() and len(prev_clean) <= 60:
                return True
            _lead = prev_clean[:1]
            # A numbered-item prev ('2.3 Problem framing') is list/TOC
            # context, not heading context — a heading never grows out
            # of a numbered list entry (doc-146: promoting '3.
            # Envisioning...' out of a competency list shredded it).
            if re.match(r"^\d+(?:\.\d+)*[ \t]+\S", prev_clean):
                return False
            # Abbreviation-continuation ('No. of Seats Aquino') is a
            # prose/table-header fragment, not a heading-shaped line.
            if re.search(r"[A-Za-z]\.\s+[a-z]", prev_clean):
                return False
            return (
                (_lead.isupper() or _lead.isdigit())
                and len(prev_clean) <= 60
                and len(prev_clean.split()) <= 6
                and not prev_clean.endswith((",", ";", ":", "!", "?"))
            )

        def _promote_standalone(text, t, _stack_allow=()):
            """Insert '## ' before t when it stands alone (iter-62 gate).

            The promoted span must BE the whole output line, the next
            line must not continue the sentence (lowercase-initial),
            and the previous line must end terminally — a heading never
            sits inside a flowing paragraph. First PASSING occurrence
            wins (same finditer discipline as _promote_numbered_pair:
            a gate-failed early match must not mask a later clean one).

            _stack_allow: normalized size-lane title texts. When the
            predecessor line IS one of them the terminal gate passes —
            a kicker/subtitle stack is a display block, not a flowing
            paragraph (odlbench 01030000000200: 'Introduction...@11'
            kicker above 'Key Functions...@17' title blocked the real
            title while the kicker promoted). Size lane only; the bold
            lane keeps the strict gate.
            """
            heading = f"## {t}"
            # Match the FULL title, not a prefix: a prefix match with a
            # full-title emission leaves the line remainder (chars past
            # the prefix) in place after the inserted heading, doubling
            # the tail ('...DNA” and “Evidence” DNA', odlbench
            # 01030000000121). The line-equality gate below already
            # enforces exact whole-line match, so the old t[:50] cap
            # bought nothing.
            escaped = re.escape(_q(t))
            _qtext = _q(text)
            for match in re.finditer(escaped, _qtext):
                pos = match.start()
                line_start = _qtext.rfind("\n", 0, pos)
                line_end = _qtext.find("\n", match.end())
                line_end = line_end if line_end != -1 else len(_qtext)
                line = _qtext[line_start + 1 : line_end]
                nxt = _qtext[line_end + 1 : _qtext.find("\n", line_end + 1)]
                nxt_clean = nxt.lstrip(" #*>-")
                # line_start == -1 (title at pos 0, page opener): no
                # predecessor line exists — prev must be EMPTY, not
                # the bogus slice the generic formula yields.
                if line_start >= 0:
                    prev = _qtext[_qtext.rfind("\n", 0, line_start) + 1 : line_start]
                else:
                    prev = ""
                prev_clean = prev.strip().rstrip("*").strip()
                # Soft-vs-hard prev split (156-class FP): corpus-wide,
                # ground-truth headings never start lowercase. A
                # lowercase-initial title may only ride a HARD terminal
                # predecessor; every relaxed gate (heading-shaped prev,
                # allcaps-short, caption-tail, stack allowance) demands
                # an uppercase-initial title.
                _pc_hard = prev_clean.rstrip(')]\u201d\u2019"')
                _prev_hard = _pc_hard.endswith((".", "?", "!", ":", ";", ","))
                _lead_alpha = next((c for c in t if c.isalpha()), "")
                _prev_terminal = _prev_terminal_ok(prev_clean, _qtext[:line_start])
                if not _prev_terminal and _stack_allow:
                    # Display-block stacks are typeset titles — a
                    # lowercase-initial line is body prose leaning on
                    # the allowance (01030000000156 'entry
                    # "Fact-checking" (https://...' beneath the real
                    # title).
                    _lead_ch = next((c for c in t if c.isalpha()), "")
                    if _lead_ch.isupper():
                        _pn = re.sub(r"[^0-9a-z]", "", prev_clean.lower())
                        if _pn and _pn in _stack_allow:
                            _prev_terminal = True
                standalone = (
                    # pos == 0: page/chapter opener at the very start of
                    # the page text (cascade 'Chapter 4' sits on text
                    # line 0) — no predecessor exists to gate on.
                    (line_start != -1 or pos == 0)
                    and line.strip() == _q(t)
                    and _prev_terminal
                    and (_prev_hard or _lead_alpha.isupper())
                    and not nxt_clean[:1].islower()
                )
                if standalone and "|" not in _qtext[line_start:pos]:
                    return text[:pos] + heading + text[match.end() :]
            return text

        def _heading_level(prefix):
            """Numbering-shape -> markdown level (docling numbering family).

            Chapter-scale markers are level 1; dotted-decimal depth maps
            1:1 to ##/###/####; lone alpha markers sit deeper.
            """
            p = prefix.strip().rstrip(".")
            if not p:
                return 2
            if re.fullmatch(r"[IVXLC]+", p):
                return 1
            if re.fullmatch(r"\d{1,3}", p):
                return 1
            if re.fullmatch(r"\d{1,3}(\.\d{1,3})*", p):
                return min(2 + p.count("."), 4)
            return 3

        def _promote_wrapped(text, t1, t2):
            """Merge a heading WRAPPED across two bold lines into one
            '## t1 t2' heading (multi-column flyers wrap long section
            titles: 'Leaking Ducts Can Also Be Hazardous to' /
            'Your Health'). Promoting each line separately breaks the
            wrapped title's contiguity; the merged heading restores it.
            Same predecessor/continuation gates as the pair promoter.
            """
            pat = re.compile(
                r"(?m)^" + re.escape(_q(t1)) + r"[ \t]*\n[ \t]*" + re.escape(_q(t2)) + r"[ \t]*$"
            )
            for m in pat.finditer(_q(text)):
                line_start = m.start()
                prev = _q(text)[_q(text).rfind("\n", 0, line_start) + 1 : line_start]
                prev_clean = prev.strip().rstrip("*").strip()
                _prev_terminal = _prev_terminal_ok(prev_clean, _q(text)[:line_start])
                nxt = _q(text)[m.end() + 1 : _q(text).find("\n", m.end() + 1)]
                nxt_clean = nxt.lstrip(" #*>-")
                if _prev_terminal and not nxt_clean[:1].islower():
                    return text[: m.start()] + f"## {t1} {t2}" + text[m.end() :]
            return text

        def _promote_numbered_pair(text, prefix, title):
            """Merge a standalone number line with the bold title that
            follows it into ONE leveled heading (book typesetting puts
            the section number on its own line: '8\\nChoosing between...').
            The separator may be a newline OR a run of blanks/tabs in
            the SAME output line — fitz and the text extractor disagree
            on line splits (Brill 002: fitz sees '8' / 'Choosing...'
            as two lines, the extracted text joins them with a tab).

            The merged heading replaces both lines. A page can carry the
            same pair twice (TOC echo, table-caption cross-reference):
            gates apply per-occurrence and the first PASSING occurrence
            wins — a gate-failed first match must not abort the search
            (orient 2.2 lesson: occurrence 1 sat under the not-yet-
            promoted allcaps chapter title).
            """
            pat = re.compile(
                r"(?m)^"
                + re.escape(prefix)
                + r"(?:[ \t]*\n[ \t]*|[ \t]+)"
                + re.escape(_q(title))
                + r"[ \t]*$"
            )
            for m in pat.finditer(_q(text)):
                line_start = m.start()
                prev = _q(text)[_q(text).rfind("\n", 0, line_start) + 1 : line_start]
                prev_clean = prev.strip().rstrip("*").strip()
                # An allcaps short line above is a heading-to-be (its
                # ## promotion runs later, order 720) — valid predecessor.
                _prev_terminal = _prev_terminal_ok(prev_clean, _q(text)[:line_start])
                nxt = _q(text)[m.end() + 1 : _q(text).find("\n", m.end() + 1)]
                nxt_clean = nxt.lstrip(" #*>-")
                if _prev_terminal and not nxt_clean[:1].islower():
                    lvl = _heading_level(prefix)
                    hashes = "#" * lvl
                    return text[: m.start()] + f"{hashes} {prefix} {title}" + text[m.end() :]
            return text

        def _promote_multiline(text, parts):
            """Merge a display title WRAPPED across N consecutive span
            lines into ONE heading (det22, 9530001d/075cc4e9): German
            magazine titles ('Die Kunst der' + 'richtigen Dichtung')
            and medical section heads ('Non-Specific' / 'Inflammatory'
            / 'Conjunctivitis: Diagnosis') wrap at display size, and
            promoting the fragments one-by-one shatters the heading
            (only the gates-passing last line survives). Same
            predecessor/continuation gates as the wrapped promoter.
            """
            pat = re.compile(
                r"(?m)^" + r"(?:[ \t]*\n[ \t]*)+".join(re.escape(_q(p)) for p in parts) + r"[ \t]*$"
            )
            for m in pat.finditer(_q(text)):
                line_start = m.start()
                prev = _q(text)[_q(text).rfind("\n", 0, line_start) + 1 : line_start]
                prev_clean = prev.strip().rstrip("*").strip()
                _prev_terminal = _prev_terminal_ok(prev_clean, _q(text)[:line_start])
                nxt = _q(text)[m.end() + 1 : _q(text).find("\n", m.end() + 1)]
                nxt_clean = nxt.lstrip(" #*>-")
                if _prev_terminal and not nxt_clean[:1].islower():
                    joined = " ".join(parts)
                    return text[: m.start()] + f"## {joined}" + text[m.end() :]
            return text

        size_titles = []
        # Display-size standalone number lines ('4.1' at the same
        # display size as its title, 145-class): the bold lane's ±15%
        # body-size window can never see them, so the pair shape
        # ('4.1' + 'Introduction' = '4.1 Introduction' in the text)
        # must be handled here. num-only shape, display size.
        _size_num_only = re.compile(r"^(?:[IVXLC]{1,6}|\d{1,3}(?:\.\d{1,3})*|[A-Za-z])\.?$")
        size_pairs = []
        for _sd_i, sd in enumerate(spans_data):
            if sd["size"] < heading_size:
                continue
            t = sd["text"]
            if _size_num_only.match(t.strip()) and len(t.strip()) <= 6:
                size_pairs.append(t.strip())
                continue
            # Must be short
            if len(t) > 80:
                continue
            # Must not be all digits (page numbers, data values)
            if t.replace(".", "").replace(" ", "").isdigit():
                continue
            # Must have alphabetic content
            if not re.search(r"[A-Za-z]{3,}", t):
                continue
            # Must not contain table pipe characters
            if "|" in t:
                continue
            size_titles.append((t, sd["size"], _sd_i, sd.get("bbox") or ()))

        # Kicker/subtitle stacks (det30): normalized texts of every
        # size-lane title, used by _promote_standalone as a prev-gate
        # allowance — a title stacked on another display line is a
        # heading block, not a paragraph continuation.
        _stack_titles = frozenset(
            re.sub(r"[^0-9a-z]", "", _p.lower()) for _p, _, _, _ in size_titles
        )

        if os.environ.get("PDF_OCR_DEBUG_DSH"):
            print(
                f"[dsh] heading_size={heading_size} median={median_size}",
                file=sys.stderr,
            )
            print(f"[dsh] size_titles={size_titles}", file=sys.stderr)
            print(f"[dsh] size_pairs={size_pairs}", file=sys.stderr)

        # ADJACENT-LINE MERGE (det22): a display title WRAPPED across
        # consecutive span lines is ONE heading. Interior lines may end
        # with ':' (075cc4e9 'Non-Specific / Inflammatory /
        # Conjunctivitis: / Diagnosis' — the ':' veto had punched the
        # chain out of the merge, shattering the heading); only the
        # FINAL line must be sentence-terminal-free. Merged runs route
        # through _promote_multiline; unmatched (non-adjacent in text)
        # runs fall back to per-fragment promotion — pre-det22
        # behavior.
        def _size_lane_ok(t):
            return not t.rstrip().endswith((".", ",", ";", "!", "?"))

        # Runs = spans-INDEX-adjacent chains of similar size (a real
        # wrapped title is set at ONE size; stacked headings of
        # different levels — 145 'Chapter 4'@20.7 over 'Nonlinear
        # equations'@24.8 — must NOT merge). Body lines between
        # candidates break the chain by index gap. TIGHT-LEADING gate
        # (dense-p90 regression pin): a wrapped title's lines share one
        # text block (vertical gap <= 0.5x font size); stacked
        # separate headings at the SAME size ('Section 0' over
        # 'Section 1' dense lists) are spaced apart and must stay
        # separate promotions.
        merged_runs = []
        _run = []
        for _rec in size_titles:
            _t, _sz, _ix, _bb = _rec
            if _run:
                _pt, _psz, _pix, _pbb = _run[-1]
                _gap_ok = True
                if _bb and _pbb:
                    _gap_ok = (_bb[1] - _pbb[3]) <= 0.5 * max(_sz, _psz)
                if (
                    _pix + 1 != _ix
                    or not _gap_ok
                    or abs(_sz - _psz) > 0.1 * max(_sz, _psz)
                    or len(" ".join(p for p, _, _, _ in _run + [_rec])) > 80
                    or not _size_lane_ok(_pt)
                ):
                    merged_runs.append(_run)
                    _run = []
            _run.append(_rec)
        if _run:
            merged_runs.append(_run)

        def _promote_size_part(text, t):
            # Pair shape first ('4.1' span + 'Introduction' span, text
            # line '4.1 Introduction') — mirrors the bold lane's Shape
            # A, for display-size numbers the bold lane's body-size
            # window can never see.
            for pair in list(size_pairs):
                _m = _promote_numbered_pair(text, pair, t)
                if _m is not text:
                    size_pairs.remove(pair)
                    return _m
            return _promote_standalone(text, t, _stack_titles)

        for _runrec in merged_runs:
            parts = [p for p, _, _, _ in _runrec]
            if os.environ.get("PDF_OCR_DEBUG_DSH"):
                print(f"[dsh] run={parts}", file=sys.stderr)
            if len(parts) > 1 and _size_lane_ok(parts[-1]):
                _m = _promote_multiline(text, parts)
                if os.environ.get("PDF_OCR_DEBUG_DSH"):
                    print(f"[dsh] multiline_matched={_m is not text}", file=sys.stderr)
                if _m is text:
                    for _t in parts:
                        if _size_lane_ok(_t):
                            text = _promote_size_part(text, _t)
                else:
                    text = _m
            else:
                for _t in parts:
                    if not _size_lane_ok(_t):
                        continue
                    text = _promote_size_part(text, _t)

        # BOLD-SAME-SIZE pass (2026-08-27, docling font-weight port):
        # headings typeset bold at body size never trip the size gate.
        # Minority-gated (<=30% bold lines) so bold-run-in body text and
        # all-bold documents stay inert. Title gates mirror the size
        # pass plus a <=12-word length bound (Unstructured
        # is_possible_title heuristic, Apache-2.0).
        # Default ON since EXP-6c (2026-09-18): the 2026-08-28 census
        # (-3 tests) was measured on the _get_page_dict-NameError-dead
        # function (dead since refactor 8cbb51f4), i.e. vacuous — the
        # full olmOCR bench re-arbitrates. PDF_OCR_BOLD_HEADINGS=0
        # disables.
        try:
            import os as _os

            if _os.environ.get("PDF_OCR_BOLD_HEADINGS", "1") == "1":
                from scientific_parser.science.font_style import (
                    bold_heading_lines,
                )

                # Two tiers: numberish candidates (pairing-only, exempt
                # from the all-digits veto by shape) + full-gate titles.
                _num_only = re.compile(r"^(?:[IVXLC]{1,6}|\d{1,3}(?:\.\d{1,3})*|[A-Za-z])\.?$")
                # LiteParse caption/attribution vetoes: bold body-size
                # "Figure 7" / "Table 3." captions and "Source:" /
                # "Note:" attribution lines satisfy every other heading
                # rule — they are figure furniture, never headings.
                _captionish = re.compile(
                    r"^(?:Figure|Fig\.?|Table|Tab\.?|Scheme|Plate|Chart|Abb\.?)\s*\.?\s*\d",
                    re.IGNORECASE,
                )
                _attributional = re.compile(
                    r"^(?:Source|Note|Adapted(?:\s+from)?|Credit|Courtesy)[:.)]",
                    re.IGNORECASE,
                )
                bold_pairs = [
                    t
                    for t in bold_heading_lines(spans_data)
                    if len(t) <= 6 and _num_only.match(t.strip())
                ]
                _raw_titles = [
                    t
                    for t in bold_heading_lines(spans_data)
                    if t not in bold_pairs
                    and len(t) <= 80
                    and len(t.split()) <= 12
                    and not t.rstrip().endswith((".", ",", ";", ":", "!", "?"))
                    and not t.replace(".", "").replace(" ", "").isdigit()
                    and re.search(r"[A-Za-z]{3,}", t)
                    and "|" not in t
                    and not _captionish.match(t.strip())
                    and not _attributional.match(t.strip())
                    # Single-word bold lines are labels/table stubs
                    # ('Position', 'Ramos', 'Administration') unless the
                    # word occurs NOWHERE else on the page — a real
                    # single-word heading ('Contents', 'Abstract') is
                    # structurally unique, a stub echoes in table rows
                    # or sibling headings.
                    and (
                        len(t.split()) > 1
                        or len(re.findall(r"\b" + re.escape(t.strip()) + r"\b", text)) == 1
                    )
                ]
                # Wrap-merge: a heading WRAPPED across two adjacent bold
                # PDF lines is ONE heading, not two — the first line
                # carrying no terminal punctuation is the tell. Merge
                # before promotion so each half never promotes alone
                # (ducts-flyer lesson: 'Leaking Ducts Can Also Be
                # Hazardous to' + 'Your Health').
                _idx_of = {}
                for _i, _sd in enumerate(spans_data):
                    _idx_of.setdefault(_sd["text"], _i)
                bold_titles = []
                _consumed = set()
                for _t in _raw_titles:
                    if _t in _consumed:
                        continue
                    _merged_with_next = False
                    for _n in _raw_titles:
                        if (
                            _n is not _t
                            and _idx_of.get(_t, -2) + 1 == _idx_of.get(_n, -1)
                            and _n not in _consumed
                            and len(_t) + 1 + len(_n) <= 80
                            and not _t.rstrip().endswith((".", ";", ":", "!", "?"))
                        ):
                            # tuple = wrapped pair; the promote loop
                            # routes tuples to _promote_wrapped.
                            bold_titles.append((_t, _n))
                            _consumed.update((_t, _n))
                            _merged_with_next = True
                            break
                    if not _merged_with_next:
                        bold_titles.append(_t)
                # TOC/list-block veto (01030000000198): three or more
                # numbered titles on adjacent PDF lines with no body
                # between them are a contents listing or spec index —
                # 'Contents' is the heading, the numbered echoes are
                # entries. Real numbered sections carry body text
                # between them (002 Brill pair, 007 sections) and stay
                # outside these adjacency runs.
                _toc_num_re = re.compile(r"^(?:[IVXLC]{1,6}|\d{1,3}(?:\.\d{1,3})*\.?)[ \t]+\S")
                _num_ixs = sorted(
                    {
                        _idx_of[e]
                        for e in bold_titles
                        if not isinstance(e, tuple) and _toc_num_re.match(e.strip())
                    }
                )
                _toc_veto = set()
                _run_start = 0
                for _r in range(1, len(_num_ixs) + 1):
                    _prev = _num_ixs[_r - 1] if _r < len(_num_ixs) else (1 << 30)
                    if _r == len(_num_ixs) or _num_ixs[_r] - _prev > 2:
                        if _r - _run_start >= 3:
                            for _e in bold_titles:
                                if (
                                    not isinstance(_e, tuple)
                                    and _idx_of.get(_e, -1) in _num_ixs[_run_start:_r]
                                ):
                                    _toc_veto.add(_e)
                        _run_start = _r
                i = 0
                while i < len(bold_titles):
                    entry = bold_titles[i]
                    # Wrapped pair — two bold PDF lines, one heading.
                    if isinstance(entry, tuple):
                        t1, t2 = entry
                        _merged = _promote_wrapped(text, t1, t2)
                        if _merged is not text:
                            text = _merged
                            i += 1
                            continue
                        # Wrapped merge did not apply (gate-failed):
                        # fall back to promoting each half standalone.
                        _h1 = _promote_standalone(text, t1)
                        if _h1 is not text:
                            text = _h1
                        _h2 = _promote_standalone(text, t2)
                        if _h2 is not text:
                            text = _h2
                        i += 1
                        continue
                    t = entry
                    if t in _toc_veto:
                        # TOC/list echo — promote nothing (block veto
                        # above). Plain survival keeps the entry text.
                        i += 1
                        continue
                    # Shape B: inline numbered title ("2.5 Spreadsheet
                    # Integration" set bold as ONE line). Promote with
                    # the level its own prefix dictates.
                    _inline = re.match(
                        r"^((?:[IVXLC]{1,6}|\d{1,3}(?:\.\d{1,3})*)[ \t]+)(\S.*)$", t.strip()
                    )
                    if _inline is not None and len(_inline.group(2)) > 2:
                        lvl = _heading_level(_inline.group(1))
                        _merged = _promote_standalone(text, t)
                        if _merged is not text:
                            # re-prefix at the right level (standalone
                            # emits fixed '##')
                            _merged = _merged.replace(f"## {t}", f"{'#' * lvl} {t}", 1)
                            text = _merged
                            i += 1
                            continue
                    # Shape A: standalone number line + title line.
                    # Try EVERY pair — list order is not layout order.
                    merged_any = False
                    for pair in list(bold_pairs):
                        _merged = _promote_numbered_pair(text, pair, t)
                        if _merged is not text:
                            text = _merged
                            bold_pairs.remove(pair)
                            merged_any = True
                            break
                    if merged_any:
                        i += 1
                        continue
                    text = _promote_standalone(text, t)
                    i += 1
        except Exception as e:
            log.warning("bold heading pass skipped: %s", e)

        return text
    except Exception as e:
        log.warning("section hierarchy failed: %s", e)
        return text


def reconstruct_footnotes(pdf_path, page_num, text):
    """Detect footnote blocks and move them after body text.

    Footnotes: bottom 15% of page, smaller font, often preceded by separator.
    """
    import fitz

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        page_rect = page.rect
        page_h = page_rect.height
        threshold_y = page_h * 0.85

        body_sizes = []
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    y = span.get("bbox", [0, 0, 0, 0])[1]
                    if y < threshold_y:
                        body_sizes.append(span.get("size", 10))

        doc.close()

        if not body_sizes:
            return text

        lines = text.split("\n")
        if len(lines) < 5:
            return text

        # Scan from bottom up: find separator or footnote markers
        fn_start = None
        for i in range(len(lines) - 1, max(len(lines) - 15, 0), -1):
            stripped = lines[i].strip()
            if re.match(r"^[-_=]{5,}$", stripped) or re.match(r"^[*]+\s*$", stripped):
                fn_start = i
                break
            if re.match(r"^[*+\u2020\u2021]\s+", stripped) or re.match(r"^\d+\s+[A-Z]", stripped):
                if fn_start is None:
                    fn_start = i

        if fn_start is None or fn_start >= len(lines) - 1:
            return text

        body = "\n".join(lines[:fn_start]).rstrip()
        footnotes = "\n".join(lines[fn_start:]).strip()
        footnotes = re.sub(r"^[-_=]{5,}$\n", "", footnotes, count=1)
        footnotes = re.sub(r"^[*]+\s*$\n", "", footnotes, count=1)

        if len(footnotes) < 10:
            return text

        return body + "\n\n**Footnotes:**\n\n" + footnotes
    except Exception as e:
        log.warning("post-processor failed: %s", e)
        return text


# =============================================================================
# #6: Header/footer learning (multi-page recurring pattern removal)
# =============================================================================


def associate_captions(pdf_path, page_num, text):
    """Detect figure/table captions and link to nearest visual element via edge weight.

    Uses fitz to detect image blocks AND table regions, then computes
    nearest-caption distance for each. Inserts structured markers with
    association metadata:

      <!-- Figure 3: caption text | linked_to: image at y=420 distance=15pt -->
      <!-- Table 2: caption text | linked_to: table at y=580 distance=22pt -->

    Edge weight = vertical distance between caption and nearest element.
    Lower distance = stronger association.
    """
    pd = _get_page_dict(pdf_path, page_num)
    if pd is None:
        return text
    try:
        import fitz

        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        blocks = page.get_text("dict")["blocks"]

        # Collect visual elements: images and tables
        visual_elements = []
        for b in blocks:
            if b.get("type") == 1:  # image block
                bbox = b["bbox"]
                visual_elements.append(
                    {
                        "type": "image",
                        "y": bbox[1],
                        "bbox": bbox,
                    }
                )
        doc.close()

        # Detect table regions via fitz find_tables
        try:
            doc2 = fitz.open(pdf_path)
            page2 = doc2[page_num - 1]
            finder = page2.find_tables(strategy="text")
            for t in finder.tables:
                visual_elements.append(
                    {
                        "type": "table",
                        "y": t.bbox[1],
                        "bbox": t.bbox,
                    }
                )
            doc2.close()
        except Exception as e:
            log.debug("associate_captions: Exception degraded: %s", e)

        # Find caption blocks
        captions = []
        for b in pd["dict"]["blocks"]:
            if b.get("type") != 0:
                continue
            bbox = b["bbox"]
            block_text = ""
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
            block_text = block_text.strip()

            match = re.match(
                r"^(Fig\.?\s*(\d+)|Figure\s*(\d+)|Table\s*(\d+))\b",
                block_text,
                re.IGNORECASE,
            )
            if match:
                cap_type = "Figure" if "fig" in match.group(1).lower() else "Table"
                cap_num = match.group(2) or match.group(3) or match.group(4)
                # Find nearest visual element
                best_dist = 9999
                best_elem = None
                for ve in visual_elements:
                    dist = abs(ve["y"] - bbox[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_elem = ve
                link_info = ""
                if best_elem and best_dist < 100:
                    link_info = f" | linked_to: {best_elem['type']} distance={best_dist:.0f}pt"
                captions.append(
                    {
                        "type": cap_type,
                        "num": cap_num,
                        "text": block_text[:200],
                        "y": bbox[1],
                        "link": link_info,
                    }
                )

        if not captions:
            return text

        for cap in captions:
            marker = f"<!-- {cap['type']} {cap['num']}: {cap['text'][:100]}...{cap['link']} -->"
            cap_pattern = re.escape(cap["text"][:30])
            match_pos = re.search(cap_pattern, text)
            if match_pos:
                pos = match_pos.start()
                text = text[:pos] + marker + "\n\n" + text[pos:]
            else:
                text += "\n\n" + marker

        return text
    except Exception as e:
        log.warning("caption association failed: %s", e)
        return text


def repair_bullets(text):
    """Normalize bullet characters to standard markdown list syntax."""
    lines = text.split("\n")
    repaired = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        # Replace common OCR bullet characters with markdown dash
        if (
            stripped
            and stripped[0]
            in "\u2022\u25e6\u25aa\u25b8\u25ba\u2023\u2043\u25c6\u25cb\u25cf\u25a0\u25a1\u25b6\u2192"
        ):
            stripped = "-" + stripped[1:]
        repaired.append(indent + stripped)
    return "\n".join(repaired)


# A standalone line of >=4 chars that is ONLY uppercase letters, spaces,
# and light punctuation. Must be surrounded by blank lines (or page start).
# Matches: "INTRODUCTION", "EXPERIMENTAL TECHNIQUES", "ABSTRACT"
# Rejects: "MSS", "ASARCO" (inline), "Fe-Ni-S" (has lowercase/digits)


UNICODE_MATH_CHARS = frozenset(
    chr(c)
    for c in list(range(0x2070, 0x209F + 1))  # super/subscripts: ²³ⁿ⁻⁺₀₁₂...
    + list(range(0x2200, 0x22FF + 1))  # math operators: ∀∂∃∇∑∏∫√∝∞∠∧∨∩∪
    + list(range(0x2190, 0x21FF + 1))  # arrows: →←↔⇒⇐⇔
    + list(range(0x2100, 0x214F + 1))  # letterlike: ℕℤℚℝℂℵℏ
    + [
        0x221E,
        0x2202,
        0x2206,
        0x2207,
        0x2208,
        0x2209,
        0x220B,
        0x2211,
        0x221A,
        0x221E,
        0x222B,
        0x222C,
        0x2229,
        0x222A,
        0x2227,
        0x2228,
        0x2264,
        0x2265,
        0x2260,
        0x2248,
        0x00B1,
        0x00D7,
        0x00F7,
        0x223C,
        0x2243,
        0x2282,
        0x2283,
        0x2286,
        0x2287,
        0x2295,
        0x2297,
        0x22A2,
        0x22A3,
        0x22A4,
        0x22A5,
        # Greek letters commonly used in math (lowercase)
        0x03B1,
        0x03B2,
        0x03B3,
        0x03B4,
        0x03B5,
        0x03B6,
        0x03B7,
        0x03B8,
        0x03B9,
        0x03BA,
        0x03BB,
        0x03BC,
        0x03BD,
        0x03BE,
        0x03BF,
        0x03C0,
        0x03C1,
        0x03C3,
        0x03C4,
        0x03C5,
        0x03C6,
        0x03C7,
        0x03C8,
        0x03C9,
        # Greek uppercase
        0x0393,
        0x0394,
        0x0398,
        0x039B,
        0x039E,
        0x03A0,
        0x03A3,
        0x03A6,
        0x03A8,
        0x03A9,
    ]
)


def restore_scientific_subscripts(text):
    """Restore scientific subscripts/superscripts from common patterns.

    Catches born-digital PDF conventions where subscripts were typed inline:
      - Variable subscripts: `RC` → `R$_{C}$`, `Mt` → `M$_{t}$`, `fO2` → `f$_{O_2}$`
      - Isotope notation: `10Be` → `$^{10}$Be`, `14C` → `$^{14}$C`,
        `21Ne`, `26Al`, `36Cl`, `3He`, `230Th`, `234U`, `238U`, `87Sr`, `143Nd`, `187Os`
      - Units with superscripts: `g cm-2` → `g cm$^{-2}$`, `g cm-3` → `g cm$^{-3}$`
      - Trig powers: `cos2 λ` → `cos$^{2}$ λ`, `sin2 θ` → `sin$^{2}$ θ`

    Conservative: only fires when the pattern is unambiguous (word boundary
    + clear numeric/symbolic subscript). Uses regex with positive lookbehind
    for alphabetic context.
    """
    if not text:
        return text
    # Isotope notation: mass number + element symbol (1-2 letters)
    # Pattern: \b(\d{1,3})([A-Z][a-z]?)\b but only common isotopes
    common_isotopes = {
        # Cosmogenic / geochron
        "3He",
        "10Be",
        "14C",
        "21Ne",
        "26Al",
        "36Cl",
        # U-series
        "230Th",
        "232Th",
        "234U",
        "235U",
        "238U",
        # Radiogenic
        "40Ar",
        "40K",
        "87Sr",
        "87Rb",
        "143Nd",
        "144Nd",
        "176Lu",
        "176Hf",
        "187Os",
        "187Re",
        "206Pb",
        "207Pb",
        "208Pb",
        # Stable
        "13C",
        "15N",
        "18O",
        "34S",
        # Common chemistry
        "14N",
        "16O",
        "32S",
        "56Fe",
        "57Fe",
    }
    for iso in common_isotopes:
        # Extract mass + element
        m = re.match(r"^(\d+)([A-Z][a-z]?)$", iso)
        if not m:
            continue
        mass, elem = m.group(1), m.group(2)
        # Pattern: word boundary + mass + element + word boundary
        # Don't double-wrap if already in $...$
        pattern = rf"\b{mass}{elem}\b"
        replacement = f"$^{{{mass}}}$**{elem}**"  # bold element + sup mass
        # Avoid replacing inside existing LaTeX
        text = re.sub(
            rf"(?<!\$){pattern}(?!\$)",
            f"$^{{{mass}}}$**{elem}**",
            text,
        )
    # Variable subscripts (single letter or short): RC, Mt, fO2, fS2, fH2O
    # Only when preceded by lowercase letter (variable name)
    var_subscript_patterns = [
        (r"\bR([A-Z])(?![a-z])", r"R$_{\1}$"),  # RC → R_{C}
        (r"\bM([a-z])(?![a-zA-Z])", r"M$_{\1}$"),  # Mt → M_{t}
        (r"\bfO2\b", r"f$_{O_2}$"),  # fO2 → f_{O_2}
        (r"\bfS2\b", r"f$_{S_2}$"),  # fS2
        (r"\bfH2O\b", r"f$_{H_2O}$"),  # fH2O
        (r"\bfCO2\b", r"f$_{CO_2}$"),  # fCO2
        # ACRONYM GUARD (audit C5 collateral, 2026-08-18): the old pattern
        # \bD([A-Z][a-z]?)(?![a-z]) fired mid-run on ANY D+Capital token —
        # prose acronyms like DCS/DNO were split ("D$_{C}$S") on every
        # multi_column page (case-sensitive order tests then fail). A
        # partition-coefficient subscript (DMin, DV, DCr) is D + ONE
        # capitalized word that ENDS the token; an acronym is followed by
        # more capitals. Require the token to end after the captured group.
        (r"\bD([A-Z][a-z]+)(?![A-Za-z])", r"D$_{\1}$"),  # DMin (word subscript)
        (r"\bD([A-Z])(?![A-Za-z])", r"D$_{\1}$"),  # DV, DC (single-letter)
    ]
    for pat, repl in var_subscript_patterns:
        text = re.sub(pat, repl, text)
    # Units with negative powers: g cm-2, g cm-3, m s-1, m2, m3, cm2
    unit_power_patterns = [
        (r"\b(cm|m|km|g|kg)(\-)(\d)\b", r"\1$^{\-\3}$"),  # cm-2 → cm^{-2}
        (r"\b(cm|m|km)(\d)\b", r"\1$^{\2}$"),  # cm2 → cm^{2}
    ]
    for pat, repl in unit_power_patterns:
        text = re.sub(pat, repl, text)
    # Trig powers: cos2, sin2, tan2, cosh2, etc
    text = re.sub(r"\b(cos|sin|tan|cosh|sinh|tanh|log|ln|exp)(\d+)\b", r"\1$^{\2}$", text)
    return text


def restore_unicode_losses(text):
    """Restore Unicode chars lost to broken PDF CMaps.

    U+FFFD (replacement char) appears when PDF font CMaps are broken.
    Restore common scientific chars by context:

      - 'X hPa' where X is FFFD → 'X hPa' (no change — no good guess)
      - '0.XXX FFFD 10' → '0.XXX × 10' (multiplication sign)
      - 'FFFD 0.05' → '± 0.05' (plus-minus before uncertainty)
      - 'A FFFD B' between uppercase letters → 'A-B' or 'A/B'

    Conservative: only high-confidence replacements.
    """
    if not text or "\ufffd" not in text:
        return text
    # Pattern 1: ± before uncertainty (e.g., '5 FFFD 0.1' → '5 ± 0.1')
    text = re.sub(r"(\d)\s*\uFFFD\s*(\d+\.?\d*)", r"\1 ± \2", text)
    # Pattern 2: × between numbers (e.g., '7.7 FFFD 1022' → '7.7 × 1022')
    text = re.sub(r"(\d+\.?\d*)\s*\uFFFD\s*(\d+)", r"\1 × \2", text)
    # Pattern 2b: standalone FFFD before equation (residual layout artifact)
    # Catches 'equation\nFFFD\n(2)' pattern — strip the orphan FFFD
    text = re.sub(r"\n\uFFFD\n(\([\d]+\))", r"\n\1", text)
    # Pattern 2c: FFFD inside equation (broken math symbol)
    text = re.sub(r"\uFFFD\s*(\d+\.?\d*)", r"± \1", text)
    # Pattern 3: → between uppercase words (e.g., 'A FFFD B' → 'A → B')
    text = re.sub(r"\b([A-Z]{2,})\s*\uFFFD\s*([A-Z]{2,})\b", r"\1 → \2", text)
    return text


# Ligature expansions ordered by frequency in English text (fi ≫ ff > fl).
# U+FFFD marks glyphs whose font CMap gives no unicode — in old Type1/Type3
# subsets these are overwhelmingly the f-ligatures.
_LIGATURE_EXPANSIONS = ("fi", "ff", "fl", "ffi", "ffl", "f")
# Same fonts also emit the raw Æ glyph where a ligature belongs (measured
# 09f90a8f: 'diÆcult' = difficult — Æ carried the ffi ligature slot) — Æ
# inside an all-lowercase word is never the Scandinavian capital.
_AE_EXPANSIONS = ("ffi", "ff", "fi", "fl", "ffl", "ae")

_LIGA_WORD_RE = re.compile(r"[A-Za-z\u00C6\u00E6\uFFFD]+(?:-[A-Za-z\u00C6\u00E6\uFFFD]+)*")


@lru_cache(maxsize=1)
def _english_wordlist():
    """System wordlist for dictionary-tested ligature repair (or None)."""
    for path in (
        "/usr/share/dict/american-english",
        "/usr/share/dict/words",
        "/usr/share/dict/british-english",
    ):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                words = {w.strip() for w in fh if "'" not in w}
            if len(words) > 10_000:
                return words
        except OSError:
            continue
    return None


def _ligature_candidates(token):
    """Yield dictionary-validated expansions of a ligature-damaged token."""
    words = _english_wordlist()
    if words is None:
        return
    upper = token.isupper()
    # Per hyphen-part repair: 'o�-line' → 'off-line' via 'off' alone.
    for part, idx in [
        (p, i)
        for i, p in enumerate(token.split("-"))
        if "\ufffd" in p or "\u00c6" in p or "\u00e6" in p
    ]:
        letters = sum(c.isalpha() and c not in "\u00c6\u00e6" for c in part)
        n_fffd = part.count("\ufffd")
        n_ae = part.count("\u00c6") + part.count("\u00e6")
        # Short residues ('�ed' alone) have no safe expansion — skip.
        # Damaged slots > 2 → combinatorial garbage, skip.
        slot_edge = part.startswith("\ufffd") or part.endswith("\ufffd")
        if (letters < 3 and not slot_edge) or letters < 1 or n_fffd + n_ae > 2:
            continue
        subs = []
        if n_fffd:
            opts = [e.upper() if upper else e for e in _LIGATURE_EXPANSIONS]
            combos = [[o] for o in opts]
            for _ in range(n_fffd - 1):
                combos = [c + [o] for c in combos for o in opts]
            for combo in combos:
                cand = part
                for e in combo:
                    cand = cand.replace("\ufffd", e, 1)
                subs.append(cand)
        if n_ae:
            for e in _AE_EXPANSIONS:
                subs.append(part.replace("\u00c6", e).replace("\u00e6", e))
        hit = next((c for c in subs if c.lower() in words), None)
        if hit is not None:
            yield part, hit, idx


def mathml_to_dollar(text):
    """Unwrap <math>-tagged LaTeX into $ / $$ spans (2026-08-31).

    Specialist assemblers occasionally emit equations as
    '<math display="block">LaTeX</math>' (2503.07260_pg6 equation 13;
    4 files across the bench). Every scorer and downstream consumer
    extracts hypotheses from $/\\( delimiters — tagged math is INVISIBLE
    and scores zero no matter how correct. The inner payload is our own
    KaTeX-convention LaTeX, so unwrapping is loss-free; display blocks
    become $$...$$, inline $...$.
    """
    if "<math" not in text:
        return text
    import html as _html

    def _unwrap(m):
        tag, inner = m.group(1), m.group(2)
        inner = _html.unescape(inner).strip()
        if not inner:
            return ""
        return f"$${inner}$$" if 'display="block"' in tag else f"${inner}$"

    return re.sub(r"<math([^>]*)>(.*?)</math>", _unwrap, text, flags=re.DOTALL)


def repair_ligature_fffd(text):
    """Recover f-ligatures lost to unmapped font glyphs (iter-78).

    Old Type1/Type3 subsets carry fi/ff/fl/ffi/ffl as single glyphs with no
    ToUnicode entry — extractors emit U+FFFD ('classi�cation', '�rst',
    'o�-line') or a wrong raw glyph ('diÆcult'). The glyph identity is not
    recoverable from the CMap, but English morphology makes the expansion
    dictionary-unique: substitute each damaged slot with every ligature and
    keep the first candidate present in the system wordlist.

    Conservative: ≥3 real letters per hyphen-part, ≤2 damaged slots, no
    dictionary → token untouched. Proper names ('Rilo�', 'Pa�sca') fail the
    lookup and keep their damage rather than gaining a wrong guess.
    """
    if not text or ("\ufffd" not in text and "\u00c6" not in text and "\u00e6" not in text):
        return text

    # Æ is legitimate in Scandinavian text — only repair it inside an
    # otherwise lowercase-ASCII word ('diÆcult'), never standalone names.
    def _token_sub(m):
        token = m.group(0)
        has_fffd = "\ufffd" in token
        ae_eligible = bool(re.search(r"[a-z][\u00C6\u00E6][a-z]", token))
        if not has_fffd and not ae_eligible:
            return token
        parts = token.split("-")
        changed = False
        for part, hit, idx in _ligature_candidates(token):
            if parts[idx] != hit:
                parts[idx] = hit
                changed = True
        return "-".join(parts) if changed else token

    return _LIGA_WORD_RE.sub(_token_sub, text)


_SPACED_CAPS_LINE = re.compile(r"^((?:[A-Z] ){4,}[A-Z])([.:!?]?)$")


def compact_spaced_caps_headings(text):
    """Compact letter-spaced caps headings into single Title-case words.

    Typographic letter-spacing ('D E F I N I T I O N S') extracts one
    glyph per space; gold transcribes these as normal words
    ('Definitions', 028649 p29 — order-test budget 1, the spaced form is
    d=10+). Join the letters and Title-case the result ONLY when the
    joined word is an English word — multi-word spaced headings
    ('C A T E G O R I E S  O F  N E E D') and acronym strings
    ('U S A') fail the lookup and stay verbatim.
    """
    if not text or " " not in text:
        return text
    words = _english_wordlist()
    if words is None:
        return text
    out = []
    for line in text.split("\n"):
        m = _SPACED_CAPS_LINE.match(line.strip())
        if m and m.group(1).replace(" ", "").lower() in words:
            out.append(m.group(1).replace(" ", "").capitalize() + m.group(2))
        else:
            out.append(line)
    return "\n".join(out)


_FUSED_TOKEN_RE = re.compile(r"(?<![A-Za-zÀ-ÿ])[A-Za-z]{7,}(?![A-Za-zÀ-ÿ])")
# 2-letter pieces admitted ONLY from function words (blocks junk splits
# like 'seekoutamattress' → 'am at tress').
_SEG_SHORT_OK = {
    "of",
    "to",
    "in",
    "by",
    "is",
    "as",
    "at",
    "on",
    "or",
    "an",
    "be",
    "do",
    "go",
    "he",
    "it",
    "me",
    "my",
    "no",
    "so",
    "up",
    "us",
    "we",
    "a",
    "i",
}


def repair_fused_words(text):
    """Insert spaces into tokens fused by space-less text layers.

    Some producers (Economist pages, 20_pg46) emit text with NO inter-
    word spaces — fitz lines read 'Whattypesofmattress arethere?' and
    gold is normally spaced. Also repairs camel-fused names when both
    halves are words.

    PAGE GATE: fires only with space-loss EVIDENCE — >=8 unknown letter-
    tokens of >=7 chars AND >=4% of letter tokens (a normal page carries
    a few unknown long words like 'extracellular'; splitting those
    corrupts good text). TOKEN GATE: splits only when the token is not
    itself a word AND segments COMPLETELY into >=2 wordlist pieces
    (each >=3 chars, or a whitelisted 2-char function word) with the
    minimum piece count. Partial coverage keeps the token verbatim
    ('Ifyourmattresst' stray 't'; Finnish 'Karanteeniaikana').
    Original per-character casing is preserved.
    """
    if not text:
        return text
    words = _english_wordlist()
    if words is None:
        return text
    # LANGUAGE GUARD: accented-Latin fraction >=1% means a non-English
    # page (Finnish 0.155, French 0.147 measured) — unknown tokens are
    # foreign words, and English-piece splits corrupt them ('varalta' →
    # 'var alt a'). Space-less English layers carry no accents (0.0).
    all_letter = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    if all_letter:
        accented = sum(1 for t in all_letter if any(c in t for c in "àáâäèéêëìíîïòóôöùúûüçñšž"))
        if accented / len(all_letter) >= 0.01:
            return text
    tokens = _FUSED_TOKEN_RE.findall(text)
    unknown = [t for t in tokens if t.lower() not in words]
    # FUSION-LENGTH tier (iter-110, 068682d5 pg1): a true space-loss page
    # fuses SENTENCES — its unknowns are >=15 chars ('Whatisamattress',
    # measured 20_pg46: 54 unknowns, ~8 of them >=15). A jargon page's
    # unknowns are real words the dictionary lacks ('multivariable',
    # 'eigenstructure', 8-14 chars, count 12) — splitting them corrupts
    # good prose. The weak tier therefore keys on >=15-char unknowns.
    _long_unknown = [t for t in unknown if len(t) >= 15]
    # GLUE-FRACTION tier (iter-89, refined iter-111): the count signal
    # alone is NOT dilution-immune — a medical jargon page carries 26
    # unknown REAL words (027c3905 p4: 'hypertensives', 'lipoprotein'…)
    # and tripped >=25, splitting 'confounders' → 'con founders'. Strong
    # glue must ALSO show >=3 long fusions: sentence-level glue always
    # produces some, dictionary-jargon never does.
    _strong_glue = len(unknown) >= 25 and len(_long_unknown) >= 3
    if len(_long_unknown) < 8 and not _strong_glue:
        return text

    def _segment(low, allow_1char=False):
        """Min-piece-count full split into wordlist pieces, or None.

        Short pieces (1-2 chars) are admitted only from the function-word
        whitelist AND never adjacently — 'seek out a mattress' is valid
        ('a' isolated), 'am at tress' is not ('am','at' adjacent). 1-char
        pieces only on long tokens: 'Adamson' → 'A damson' is a name, not
        a fused sentence.
        """
        n = len(low)
        # best[i] = (count, last_len) minimal-count segmentation of low[:i]
        best = [None] * (n + 1)
        best[0] = (0, 99)
        for i in range(1, n + 1):
            for L in range(1, i + 1):
                prev = best[i - L]
                if prev is None:
                    continue
                piece = low[i - L : i]
                if L == 1 and not allow_1char:
                    continue
                ok = piece in words and (L >= 3 or piece in _SEG_SHORT_OK)
                if ok and not (L <= 2 and prev[1] <= 2):
                    cnt = prev[0] + 1
                    if best[i] is None or cnt < best[i][0]:
                        best[i] = (cnt, L)
        if best[n] is None or best[n][0] < 2:
            return None
        pieces = []
        i = n
        while i > 0:
            L = best[i][1]
            pieces.append(low[i - L : i])
            i -= L
        return pieces[::-1]

    def _token_sub(m):
        tok = m.group(0)
        if tok.lower() in words:
            return tok
        pieces = _segment(tok.lower(), allow_1char=len(tok) >= 12)
        # NAME GUARD: junk and name segmentations use only SHORT pieces
        # ('Fogelman' → 'fog elm an', max 3; 'varalta' → 'var alt a').
        # Real sentence fusions carry a substantial word
        # ('Whattypesofmattress' max 8). Require max piece >=5 for long
        # tokens (>=12 chars), >=6 for shorter ones.
        if pieces is not None:
            longest = max(len(p) for p in pieces)
            need = 5 if len(tok) >= 12 else 6
            if longest < need:
                # LOWERCASE ALL-COMMON exception (iter-89, 20_pg46):
                # short fusions of two/three COMMON words ('arethere',
                # 'suchaswool' = such+as+wool) never carry a >=5-char
                # piece but are real glue on space-less layers. All-
                # lowercase only — Title-case tokens are names
                # ('Adamson' = adam+son) and stay; piece count <=3
                # keeps junk 5+ piece splits rejected; requires the
                # page-level strong-glue fraction so real unknown words
                # on normal pages ('extracellular') never split.
                if not (
                    _strong_glue
                    and tok.islower()
                    and len(pieces) <= 3
                    and min(len(p) for p in pieces) >= 2
                    and all(p in words for p in pieces)
                ):
                    pieces = None
        if pieces is None:
            return tok
        out, pos = [], 0
        for p in pieces:
            out.append(tok[pos : pos + len(p)])
            pos += len(p)
        return " ".join(out)

    out = _FUSED_TOKEN_RE.sub(_token_sub, text)
    # PUNCT-SPACE: the same layers drop the space after sentence
    # punctuation ('advice?Seek') and inside dimension spec lists
    # ('single(90cm x 190cm)'). The paren rule is DIMENSION-UNIT gated:
    out = re.sub(r"([a-zA-Z][?!])([A-Z][a-z])", r"\1 \2", out)
    # lowercase continuation after sentence punctuation ('Lots,and'
    # → 'Lots, and') — same space-loss layers, seen at 20_pg46.
    out = re.sub(r"([a-z][,;])([a-z]{3,})", r"\1 \2", out)
    out = re.sub(r"([a-z])\((?=\d[^)]*(?:cm|mm|kg|km|ft|in)\b)", r"\1 (", out)
    return out


def promote_allcaps_headings(text):
    """Convert standalone ALL-CAPS lines AND numbered sections to markdown headings.

    Two promotion strategies:
      1. **ALL-CAPS headings** (existing): pure uppercase lines ≥4 chars → `##`
      2. **Numbered sections** (NEW 2026-07-25): lines matching
         `N. Title` or `N.N. Title` or `N.N.N. Title` → `##`
         Catches scientific paper conventions:
           "1. Introduction" → "## 1. Introduction"
           "2.1. Geomagnetic field effects" → "## 2.1. Geomagnetic field effects"
           "2.2.1. Lal" → "## 2.2.1. Lal"

    Conservative: numbered sections must:
      - Start at line beginning (after optional whitespace)
      - Have ≥2 words after the number
      - First word after number starts with uppercase
      - Total line length ≤80 chars (not a long sentence)
    """
    lines = text.split("\n")
    out = []
    # det28: dsh (registry order 620) emits '^#' when it promotes;
    # when it has spoken, Strategy 2 below stands down (see comment).
    _dsh_spoke = re.search(r"(?m)^#{1,6} \S", text) is not None
    numbered_section_re = re.compile(
        r"^(\d+(?:\.\d+){0,3}\.)\s+([A-Z][a-zA-Z]{2,}(?:\s+[a-zA-Z()\-,'&:]+){0,8})$"
    )
    for line in lines:
        stripped = line.strip()
        # Strategy 1: ALL-CAPS headings (existing logic)
        # Guard against table-cell false-positives (e.g. "## NONE", "## L-LTP"):
        # require multi-word OR single word >=8 chars. Short isolated allcaps
        # like "NONE" "ABCD" "L-LTP" are table cells/values, not headings.
        is_multi_word = stripped.count(" ") >= 1
        is_long_single = " " not in stripped and len(stripped) >= 8
        if (
            (is_multi_word or is_long_single)
            and len(stripped) >= 4
            and stripped.isupper()
            and any(c.isalpha() for c in stripped)
            and not stripped.startswith("<!--")
            and not stripped.startswith("[")
            and all(c.isupper() or c.isspace() or c in ".,'&-:()" for c in stripped)
            and not stripped.endswith((",", ";", ":", "-", "&"))
            and stripped.count(",") <= 1
            and not re.search(r"\d{2,}", stripped)
            # Skip if math artifacts present (e.g. "R$_E$D$_U$CED" from title
            # with subscripts). Such lines aren't real allcaps headings.
            and "$" not in stripped
            and "_{" not in stripped
            and "^{" not in stripped
        ):
            out.append(f"## {stripped}")
            continue
        # Strategy 2: Numbered sections (NEW)
        # det28: when detect_section_hierarchy (order 620) already
        # emitted a heading on this page, it owns numbered promotion —
        # its gates (bold/size evidence, prev-terminal, TOC-block veto)
        # are strictly stronger than this text-only heuristic. Running
        # both let ungated 'N. Title' TOC entries back in after dsh
        # vetoed them (01030000000198: '1. Overview of OCR Pack' etc).
        if not _dsh_spoke and len(stripped) <= 80 and not stripped.startswith("#"):
            m = numbered_section_re.match(stripped)
            if m:
                out.append(f"## {stripped}")
                continue
        out.append(line)
    return "\n".join(out)


# Threshold: pages with fewer than this many extracted chars are likely
# OCR failures or blank pages — flag for downstream verification.
LOW_TEXT_DENSITY_THRESHOLD = 200
# Header/page-number heuristic: a top or bottom line that is short (<80 chars)
# and either pure digits or bracketed journal info.
HEADER_LINE_MAX_CHARS = 80
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
BRACKET_HEADER_RE = re.compile(r"^\s*\[.*\]\s*$")


def sanitize_phantom_decimals(text):
    """Remove phantom decimal points in large numbers within HTML table cells.

    OCR models (especially ovisocr2) sometimes insert decimal points into
    large integer values: 88707 \u2192 8870.7, 100344 \u2192 10034.4.
    This changes geochemical data by ~10\u00d7.

    Conservative rules — only fixes cells that:
    - Are inside <td> tags (table context only)
    - Contain ONLY a number (no text, units, ranges, or operators)
    - Have integer part >= 4 digits (>= 1000)
    - Have decimal part <= 2 digits
    """

    def _fix_td_cell(match):
        full = match.group(0)
        inner = match.group(1).strip()
        if re.fullmatch(r"\d{4,}\.\d{1,2}", inner):
            fixed = inner.replace(".", "")
            return full.replace(inner, fixed)
        return full

    return re.sub(r"<td>(.*?)</td>", _fix_td_cell, text, flags=re.DOTALL)


# Fix 1: Per-mille symbol repair


def fix_permille(text):
    """Replace % with \u2030 when preceded by decimal and followed by 'per'."""
    return re.sub(r"(\d\.\d+)%(?=\s*per\s)", r"\1‰", text)


# Fix 3: REE sequence validator
REE_SEQUENCE = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]


def validate_scientific_symbols(text):
    """Fix OCR symbol confusions in scientific context (u\u2192\u03bc, W\u2192\u03a9, etc.)."""
    for pat, repl in SCIENTIFIC_SYMBOL_RULES:
        text = re.sub(pat, repl, text)
    return text


# Fix 6: Chemical formula validator
FORMULA_RE = re.compile(
    r"\b(?:[A-Z][a-z]?\d*){2,}\b"
    r"|\b[A-Z][a-z]?\d+0\d+\b"
)

# Single-element formulas with zero in count: Si02, C02, H20, Na20


def validate_chemical_formulas(text):
    """Repair OCR errors in chemical formulas using periodictable.

    Three-pass unified approach:
    1. Pre-pass: single-element + 0 (Si02\u2192SiO2, C02\u2192CO2, H20\u2192H2O, Na20\u2192Na2O).
    2. Standard pass: multi-element formula validation + AI\u2192Al repair.
    3. Post-parse sanity: if element count >= 20 and formula has 0,
       try 0\u2192O replacement (Fe203\u2192Fe2O3).

    All passes verify the repaired version parses via periodictable.
    Zero false positives on non-chemistry text.
    """

    def _max_count(formula_str):
        """Return max element count, or None if parse fails."""
        try:
            p = periodictable.formula(formula_str)
            return max(p.atoms.values()) if p.atoms else 0
        except (ValueError, Exception):
            return None

    def _unique_elements(formula_str):
        """Return number of unique elements, or None if parse fails."""
        try:
            p = periodictable.formula(formula_str)
            return len(p.atoms)
        except (ValueError, Exception):
            return None

    # Pass 1: single-element formulas with 0 in count position
    def _repair_single(m):
        original = m.group(0)
        fixed = original.replace("0", "O")
        if fixed == original:
            return original
        if any(ng in fixed for ng in ["HeO", "NeO", "ArO", "KrO", "XeO", "RnO"]):
            return original
        orig_elems = _unique_elements(original)
        fix_elems = _unique_elements(fixed)
        # Use fixed if it has MORE unique elements (oxygen added) or
        # original doesn't parse but fixed does
        if fix_elems is not None:
            if orig_elems is None:
                return fixed
            if fix_elems > orig_elems:
                return fixed
        return original

    text = SINGLE_ELEM_ZERO_RE.sub(_repair_single, text)

    # Pass 2+3: multi-element formulas
    def _repair(m):
        f = m.group(0)
        if len(f) < 3:
            return f
        orig_mc = _max_count(f)
        # Already good (reasonable counts, no zeros)?
        if orig_mc is not None and orig_mc < 20 and "0" not in f:
            return f
        # Pass 3: high count + has 0 → try full 0→O
        if orig_mc is not None and orig_mc >= 20 and "0" in f:
            candidate = f.replace("0", "O")
            cand_mc = _max_count(candidate)
            if cand_mc is not None and cand_mc < orig_mc:
                return candidate
        # Standard OCR repairs for unparseable formulas
        if orig_mc is None:
            fixed = re.sub(r"\bAI", "Al", f)
            fixed = fixed.replace("0", "O")
            if fixed != f and _max_count(fixed) is not None:
                return fixed
        return f

    return FORMULA_RE.sub(_repair, text)


def markdown_compatible(s):
    """Make text with LaTeX math compatible with markdown rendering.

    Ported subset of Nougat's postprocessing.markdown_compatible.
    Handles:
    - Equation tags: "(N) \\[eq\\]" → "\\[eq \\tag{N}\\]"
    - Closes unclosed LaTeX environments (matrix, tabular, etc.)
    - Strips extra backslashes from common escapes
    """
    if not s:
        return s

    # Equation tag normalization: "(1) \[eq\]" → "\[eq \tag{1}\]"
    s = re.sub(
        r"^\(([\d.]+[a-zA-Z]?)\) \\\[(.+?)\\\]$",
        r"\[\2 \\tag{\1}\]",
        s,
        flags=re.M,
    )
    s = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\)$",
        r"\[\1 \\tag{\2}\]",
        s,
        flags=re.M,
    )

    # Close unclosed LaTeX environments
    for env in (
        "bmatrix",
        "pmatrix",
        "matrix",
        "tabular",
        "table",
        "align",
        "cases",
        "array",
    ):
        begins = s.count(r"\begin{%s}" % env)
        ends = s.count(r"\end{%s}" % env)
        if begins > ends:
            s += (r"\end{%s}" % env) * (begins - ends)

    # Fix common escape issues
    s = s.replace(r"\. ", ". ")
    s = s.replace(r"\,", ", ")

    return s


# Whole-line running-stamp patterns for OCR pages. Geometry-based header
# detection needs a text layer; OCR'd pages have none, so running heads
# and download stamps survive verbatim (iter-94 census: 4/5 headers
# absent-test fails). Anchored to the WHOLE line (after list-marker
# trim) and shaped to reject reference citations: no 'pp.'/page-ranges
# after the volume pair, only an optional month-word or (year).
_STAMP_MARKDOWN_PREFIX_RE = re.compile(r"^(?:\s*(?:[-*+](?=\s|$)|\d+[.)](?=\s|$))\s*)+")
# Bold/italic lead-in labels VL models prefix their lists with
# ('1. **Page Header:** <stamp>') — stripped only when the REMAINDER
# fullmatches a stamp pattern, so prose bold labels never match.
_STAMP_LABEL_PREFIX_RE = re.compile(
    r"^(?:\*\*[^*_]{0,40}?:\*\*|\*\*[^*_]{0,40}?\*\*:|__[^*_]{0,40}?:__|__[^*_]{0,40}?__:|[^*_]{0,40}?:)\s*"
)
_STAMP_LINE_RES = (
    # ScienceDirect/Elsevier access stamp — repeats within one page.
    re.compile(r"downloaded from \S+ at .+", re.I),
    # Journal running head: [Journal name] Vol. 47, No. 2 [Maret 2013|(1968)]
    re.compile(
        r"(?:[\w .,&'-]+?\s+)?(?:vol|volume)[.:]?\s*\d+\s*[,;]?\s*"
        r"(?:no|nomor|number|issue)[.:]?\s*\d+"
        r"(?:\s*[,.;]\s*\w+\s+\d{4}|\s*\(\d{4}\))?\s*$",
        re.I,
    ),
    re.compile(r"page \d+(?:\s*/\s*\d+)?$", re.I),
)


def strip_running_stamps(text: str) -> str:
    """Remove whole-line running headers/stamps from OCR'd page text.

    OCR pages have no geometry for the band detector, so publisher
    stamps and journal running heads survive into output verbatim —
    often in multiple copies (the stamp repeats under list items).
    Only OCR-profile pages run this (registered {"ocr"}).
    """
    if not text:
        return text
    lines = text.split("\n")
    kept, removed = [], 0
    for ln in lines:
        body = _STAMP_MARKDOWN_PREFIX_RE.sub("", ln).strip()
        # Region-dispatcher wrappers: the rotated-margin stamp arrives as
        # '<p style="writing-mode: vertical-rl; ...">Downloaded from …'
        # (12cf936d) — unwrap a simple <p>-tag lead before matching.
        body = re.sub(r"^<p\b[^>]*>", "", body).strip()
        body = re.sub(r"</p>$", "", body).strip()
        if not any(rx.fullmatch(body) for rx in _STAMP_LINE_RES):
            body = _STAMP_LABEL_PREFIX_RE.sub("", body).strip()
        if body and any(rx.fullmatch(body) for rx in _STAMP_LINE_RES):
            removed += 1
            continue
        kept.append(ln)
    # TRAILING/LEADING page number (c655ed75 '110'): a standalone pure
    # number/roman line as the FIRST or LAST non-empty line is the
    # printed page number — OCR pages have no geometry band to catch it.
    # Requires >=4 non-empty lines so short fragments keep their numbers
    # (a 3-line quote whose middle is '510' is content, not a folio).
    idx_nonempty = [i for i, l in enumerate(kept) if l.strip()]
    if len(idx_nonempty) >= 4:
        _num_re = re.compile(r"^\d{1,4}$|^[ivxlcdmIVXLCDM]{1,7}$")
        for i in (idx_nonempty[-1], idx_nonempty[0]):
            if _num_re.match(kept[i].strip()):
                kept[i] = ""
                removed += 1
    if not removed:
        return text
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return f"<!-- Running stamps stripped ({removed} line(s)) -->\n" + out


def strip_cjk_hallucination(text: str, cjk_ratio_threshold: float = 0.005) -> str:
    """Strip CJK characters that glm-ocr hallucinates on noisy pages.
    Scientific papers in English should not contain CJK. ANY CJK char in
    OCR output is treated as hallucination and stripped (default behavior).
    For genuine CJK docs, set --cjk-tolerant (TODO).
    """
    if not text:
        return text
    cjk_chars = re.findall(r"[\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]", text)
    if not cjk_chars:
        return text
    # Strip ALL CJK regardless of count — even one is hallucination on
    # an English scientific page.
    stripped = re.sub(
        r"[\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]",
        "",
        text,
    )
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return f"<!-- OCR CJK hallucination stripped ({len(cjk_chars)} chars) -->\n" + stripped


# Valid single-word LaTeX commands relevant to OCR-glue repair. If the full
# command word is in here, NEVER split it. Superset of the old alternation
# plus the AMS ops/relations it mistakenly bisected.
_KNOWN_CMDS = frozenset(
    """
    to leq geq neq times cdot cup cap subseteq subset supseteq supset
    sim in ell
    wedge vee forall exists sum prod sqrt partial nabla sin cos tan cot
    log ln exp lim max min sup inf det dim arg
    alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta
    iota kappa lambda mu nu xi pi rho sigma varsigma tau upsilon
    phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi
    Sigma Upsilon Phi Psi Omega mathbb mathcal mathfrak mathrm
    left right frac int oint big Big bigg Bigg
    bigsqcup bigodot bigoplus bigotimes biguplus bigvee bigwedge bigcap bigcup
    coprod sqcup sqcap sqsubset sqsupset sqsubseteq sqsupseteq
    subsetneq supsetneq subsetneqq supsetneqq simeq cong asymp approx
    varnothing emptyset dagger ddagger ldots dots cdots vdots ddots
    oplus otimes odot ominus uplus lceil rceil lfloor rfloor
    langle rangle triangle trianglelefteq trianglerighteq
    infty varinjlim varprojlim xrightarrow xleftarrow
    pmod bmod quad qquad displaystyle textstyle limits nolimits
    leftarrow rightarrow leftrightarrow leftarrowtail rightarrowtail
    leftharpoonup leftharpoondown rightharpoonup rightharpoondown
    leftleftarrows rightrightarrows rightleftarrows
    leftrightharpoons rightleftharpoons
    bigl bigr biggl biggr Bigl Bigr Biggl Biggr bigstar
    top injlim intop intercal sinh cosh tanh coth
    """.split()
)
# The added line above: valid KaTeX commands that carry a shorter _KNOWN_CMDS
# word as a proper prefix (left*, right*, to+p, in+t*/j/cal, sin/cos/tan/cot+h,
# big*). _fix_glued_commands's longest-prefix split bisected them into
# garbage ("\leftrightarrow" -> "\left rightarrow", "\bigl(" -> "\big l(") —
# every token verified renderable via pykatex before inclusion (2026-09-09,
# bench 05407 flip-class).


def _fix_glued_commands(inner: str) -> str:
    """Insert a space where an OCR-glued command runs into a variable:
    ``\\\\toCH`` → ``\\\\to CH``. Whole-word aware: ``\\\\bigsqcup`` etc. are
    valid commands and must NOT be split (old alternation bug, 2026-08-16).
    """

    def _split_match(m: "re.Match[str]") -> str:
        word = m.group(1)
        tail = m.group(2)
        if word in _KNOWN_CMDS:
            return m.group(0)
        # longest valid command that is a proper prefix of `word`
        for k in range(len(word) - 1, 1, -1):
            if word[:k] in _KNOWN_CMDS:
                return "\\" + word[:k] + " " + word[k:] + tail
        return m.group(0)

    return re.sub(r"\\([a-zA-Z]+)(\{?)", _split_match, inner)


# KaTeX-UNDEFINED commands our sources emit (iter-131, census-measured
# counts): never valid in the scorer's renderer, so the rewrite is
# strictly render-improving. \boldmath{X} (VL arrays) -> \boldsymbol{X};
# \slash -> /; \nsubset -> \not\subset; \mit -> \mathit.
_UNDEF_CMD_MAP = {
    r"\boldmath{": r"\boldsymbol{",
    r"\slash": "/",
    r"\nsubset": r"\not\subset",
    r"\nsupset": r"\not\supset",
    r"\mit ": r"\mathit ",
}


def _normalize_undefined_cmds(inner: str) -> str:
    for bad, good in _UNDEF_CMD_MAP.items():
        if bad in inner:
            inner = inner.replace(bad, good)
    return inner


# ── run-9: unrenderable-math repair (oracle-gated, strictly additive) ────── #
# Census 2026-09-10: 197 of 20,742 arxiv equations render-fail in KaTeX; the
# scorer SKIPS render-fail candidates entirely (tests.py `if not
# hypothesis_render: continue`) — they are dead weight. 14 are structural
# near-misses (12 brace EOF-truncation, 2 env-mismatch). This pass adopts a
# rendering variant for those spans only; everything else stays byte-exact.
_DISPLAY_SPAN_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_SPAN_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
# Loose combining marks KaTeX rejects standalone (U+0300-036F); keep marks
# applied after an alnum base (x̃ = legit stack).
_LOOSE_COMBINING_RE = re.compile(r"(?<![A-Za-z0-9])[\u0300-\u036f]")
_MAX_REPAIR_SPANS_PER_DOC = 200


def _repair_brace_delta(s: str) -> int:
    """Net unbalanced braces, skipping escaped \\{ and command backslashes."""
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return depth


def _repair_env_pairs(s: str) -> list[tuple[str, str]]:
    return re.findall(r"\\(begin|end)\{([a-z*]+)\}", s)


def _repair_classify(body: str) -> bool:
    """Cheap regex gate: does this span carry a repairable defect signature?"""
    if _repair_brace_delta(body) > 0:
        return True
    from collections import Counter

    begins = Counter(e for k, e in _repair_env_pairs(body) if k == "begin")
    ends = Counter(e for k, e in _repair_env_pairs(body) if k == "end")
    if begins - ends:
        return True
    if _LOOSE_COMBINING_RE.search(body):
        return True
    return False


def _repair_variants(body: str) -> list[str]:
    """Ordered deterministic structural variants (cheap-first)."""
    from collections import Counter

    out = []
    d = _repair_brace_delta(body)
    if d > 0:
        out.append(body + "}" * d)
    begins = Counter(e for k, e in _repair_env_pairs(body) if k == "begin")
    ends = Counter(e for k, e in _repair_env_pairs(body) if k == "end")
    if begins - ends:
        out.append(body + "".join(r"\end{%s}" % env for env in sorted((begins - ends).elements())))
    stripped = _LOOSE_COMBINING_RE.sub("", body)
    if stripped != body:
        out.append(stripped)
        d2 = _repair_brace_delta(stripped)
        if d2 > 0:
            out.append(stripped + "}" * d2)
    return out


def repair_unrenderable_math(final_md: str, renders_clean=None) -> str:
    """Adopt the first rendering structural variant for each currently
    unrenderable math span. Strictly additive: spans without a repairable
    defect signature (classify gate) are never touched — not even
    oracle-checked (bounds browser round-trips to the suspicious subset).
    """
    if not final_md:
        return final_md

    display = [(m.start(1), m.end(1), m.group(1)) for m in _DISPLAY_SPAN_RE.finditer(final_md)]
    inline = [
        (m.start(1), m.end(1), m.group(1))
        for m in _INLINE_SPAN_RE.finditer(final_md)
        # inline spans nested inside display bodies are part of the outer
        # span; repairing them separately corrupts indices (overlap filter)
        if not any(ds <= m.start(1) and m.end(1) <= de for ds, de, _ in display)
    ]
    budget = _MAX_REPAIR_SPANS_PER_DOC
    for start, end, body in reversed(display + inline):
        if budget <= 0:
            break
        if not _repair_classify(body):
            continue
        # Heavy import deferred to first suspicious span — equation_cascade
        # pulls the ONNX cascade stack; prose docs never pay it.
        if renders_clean is None:
            from scientific_parser.science.equation_cascade import (
                katex_renders_clean as renders_clean,
            )
        budget -= 1
        if renders_clean(body):
            continue  # defect signature but renders (e.g. KaTeX tolerates) — untouched
        for variant in _repair_variants(body):
            if renders_clean(variant):
                final_md = final_md[:start] + variant + final_md[end:]
                break
    return final_md


def fix_latex_math_output(text: str, ocr_artifacts: bool = True) -> str:
    """Clean up common LaTeX errors introduced by OCR + skill typography pipeline.

    Targets six failure classes identified via olmOCR-Bench failure analysis:

      1. **Nested ``$`` delimiters in display math** (40.3% of failures).
      2. **Multi-dollar inline** (22.3%).
      3. **LaTeX command confusions** (~2%): glm-ocr `\\sqrt` → `\\sort`.
      4. **Invalid escape ``\\-``**: KaTeX rejects.
      5. **Triple-dollar ``$$$...$$$``**: typography wrapping artifacts.
      6. **Brace inflation** ``_{1}`` → ``_1``: olmOCR-Bench normalizes to this.

    Plus Phase 7: pykatex-driven KaTeX validation. Each math block is rendered
    via pykatex; if it raises ParseError, attempt auto-repair from the error
    message. Drops blocks that remain invalid after repair (KaTeX-renderable
    output is required by olmOCR-Bench).
    """
    if not text:
        return text

    # (0a) LONE-SCRIPT UNICODE (iter-76): a math block that is ONLY a
    # single-digit sub/superscript is prose typography (unit superscripts
    # kg/m², isotope subscripts), not math — gold keeps these as unicode
    # glyphs verbatim (olmOCR normalize_text preserves ²/₂). Multi-glyph
    # scripts (x^{m-1}) stay LaTeX untouched. Shared with the scripts
    # extractor wrapper (source-side conversion).
    text = _lone_scripts_to_unicode(text)
    # (0) Empty inline math merge: `$ $ X $ $` → `$X$` (arxiv_math bench fix).
    # Born-digital text-layer wraps math fragments with stray empty `$ $`
    # markers between content. Only merges the 3-block pattern (empty-content-
    # empty). Does NOT strip bare `$$` (which would corrupt display math).
    # Also handle 2-block: `$ $ X $` → `$X$` (leading/trailing empty only).
    # FIX 2026-08-15: \s → [ \t]. \s matches \n, so a display line ending
    # "$" directly followed by another starting "$" matched the "empty"
    # gap → adjacent display equations FUSED and lines cascaded-deleted
    # (measured -11 tests on arxiv_math sample). Whitespace between an
    # empty marker pair is always horizontal.
    text = re.sub(
        r"\$[ \t]+\$([^$\n]{1,200})\$[ \t]+\$",
        r"$\1$",
        text,
    )
    text = re.sub(
        r"\$[ \t]+\$([^$\n]{1,200})\$",
        r"$\1$",
        text,
    )

    # PROSE LONE-\to (iter-62): '\to' glued directly to an uppercase word
    # ('\toAPART' — dictionary cross-reference arrow LaTeX-ified by the
    # VL model, $ delimiters then stripped by unwrap) is prose, not math;
    # real math \to sits inside $..$ with operands. Restore the arrow.
    text = re.sub(r"(?<!\$)\\to(?=[A-Z])", "→ ", text)
    text = re.sub(
        r"\$([^$\n]{1,200})\$[ \t]+\$",
        r"$\1$",
        text,
    )
    # CROSS-LINE OPERAND MERGE (iter-96, 08365 3rd test): a source line
    # break splits one inline equation into '…cr(G) \leq$' + newline +
    # '$6.6n if G$'. A run ENDING in a dangling binary relation has no
    # right operand — the next line's leading $-run IS that operand.
    # Merge into one run so fuzzy matchers see the whole equation.
    # \bullet added (iter-104, 06597): sequence notation splits at its
    # separator ('…210 \bullet$' + '$(2)^{\infty}…') — \bullet is a
    # separator inside $-runs; prose bullets never live inside runs.
    _dangle_ops = r"(?:\\leq|\\le|\\geq|\\ge|\\ll|\\gg|\\equiv|\\sim|\\approx|\\neq|\\ne|\\lt|\\gt|\\pm|\\mp|\\times|\\cdot|\\bullet|=|<|>|\+)"
    text = re.sub(
        r"\$([^$\n]{1,200}?)(\s*" + _dangle_ops + r")\s*\$\s*\n\s*\$([^$\n]{1,200}?)\$",
        lambda m: "$" + m.group(1) + m.group(2) + " " + m.group(3) + "$",
        text,
    )
    # MINUS VARIANT (iter-103, 05506): a set-builder split at its tail
    # ('…s^{\ell - 1} -$' + '$2 s, \ldots , \tau_i^1 \}…') dangles a
    # bare '-'. Hyphen-guard: require LaTeX structure (\cmd, ^ or _)
    # in the run so prose '$5-$' never merges.
    text = re.sub(
        r"\$([^$\n]{1,600}?)(\s*-)\s*\$\s*\n\s*\$([^$\n]{1,600}?)\$",
        lambda m: (
            "$" + m.group(1) + m.group(2) + " " + m.group(3) + "$"
            if re.search(r"\\|\^|_", m.group(1))
            else m.group(0)
        ),
        text,
    )
    # INTER-RUN BARE FUNC-WORD (iter-107, 06570): a source line break
    # splits an inline equation leaving a bare operator+func-word OUTSIDE
    # any run ('$…\alpha$+deg$\beta + …' — 'deg' is roman CMR inside
    # math). Absorb the glue into ONE run as '+ \deg' when both sides
    # carry LaTeX structure (prose '$5-$5' never matches).
    _funcs = r"(?:deg|dim|max|min|ln|exp|sup|inf|det|gcd|lcm|ker|Pr|log|sin|cos|tan)"
    text = re.sub(
        r"\$([^$\n]{1,600}?)\$\s*(\+|-)(" + _funcs + r")\s*\$([^$\n]{1,600}?)\$",
        lambda m: (
            "$" + m.group(1) + " " + m.group(2) + " \\" + m.group(3) + " " + m.group(4) + "$"
            if re.search(r"\\|\^|_", m.group(1)) and re.search(r"\\|\^|_", m.group(4))
            else m.group(0)
        ),
        text,
    )

    # Helper: clean the *inside* of a math block
    def _clean_math_inner(inner: str) -> str:
        # (3) OCR command confusions
        for pat, repl in _OCR_LATEX_CMD_CONFUSIONS:
            inner = pat.sub(repl, inner)
        # (3-pre) Old-math-notation fixes for 19th-century texts (olmOCR old_scans_math)
        inner = inner.replace("&c.", r"\ldots")
        inner = inner.replace("&c", r"\ldots")
        # (3-pre2) Strip orphaned leading braces from truncated equations
        inner = re.sub(r"^\}+", "", inner)
        # (3-pre3) Fix escaped ampersand in math: KaTeX needs \& for a
        # literal &, bare & outside environments is invalid — but bare & is
        # the ALIGNMENT/COLUMN separator inside environment bodies, where \&
        # destroys the structure (\begin{cases} x, & y renders mis-aligned
        # and breaks scorer substring match; bench 05140/05062 flip-class,
        # 2026-09-09). Escape only when no &-consuming environment is present.
        if not re.search(
            r"\\begin\{(?:cases|dcases|rcases|aligned|alignedat|align|alignat|"
            r"array|darray|split|CD|smallmatrix|[pbvVBm]?matrix)\}",
            inner,
        ):
            inner = re.sub(r"(?<!\\)&", r"\&", inner)
        # (3-pre4) Drop stray asterisk-only fraction denominators: \frac{X}{*}
        inner = re.sub(r"\{\\?\*\}", "{}", inner)
        # (3-pre5) glm-ocr systematic artifact cleanup for scanned math:
        # \bar{X} → X (false bars on single letters — very common in old scans)
        # PROVENANCE-GATED (iter-29): these rules clean OCR-model false
        # accents. Font-detector \hat/\bar come from MEASURED overstrike
        # glyphs — stripping them destroys ground truth (05068: detector
        # emitted \hat{U}, final pass silently removed it). OCR pages get
        # their per-page strip on the OCR path; only the final whole-doc
        # pass disables this when font-math pages exist.
        if ocr_artifacts:
            inner = re.sub(r"\\bar\{([A-Za-z])\}", r"\1", inner)
            # \hat{X} → X (false hats on single subscript letters)
            inner = re.sub(r"\\hat\{([A-Za-z])\}", r"\1", inner)
        # +- → + (OCR sign duplication artifact)
        inner = inner.replace("+-", "+")
        # (3a) Born-digital text-layer leak: variable + bare greek + bare digit
        # e.g. "V\rho_2" → "V_{\rho_2}", "C\theta1" → "C_{\theta_1}", "x_i" stays.
        # Pattern: letter directly followed by \greek (no _), then bare digit.
        # Skip if already has explicit _{...} structure.
        inner = re.sub(
            r"([A-Za-z])\\(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|"
            r"theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|varsigma|"
            r"tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|"
            r"Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)(\d+)",
            r"\1_{\\\2_\3}",
            inner,
        )
        # (3b) Missing space after LaTeX command (biggest arxiv_math bug).
        # Pattern: \cmd + glued letter (OCR): \toCH -> \to CH.
        # NOTE (2026-08-16 iter3): the old alternation regex SPLIT VALID
        # LONGER COMMANDS (\bigsqcup -> \big sqcup, \supsetneq ->
        # \supset neq; measured 23-test OUR_EQS_FAIL_RENDER family) ->
        # KaTeX render failure. Whole-word check now: split ONLY when the
        # full command word is NOT valid but a valid command is its prefix.
        inner = _fix_glued_commands(inner)
        # (3c) KaTeX-undefined command normals (iter-131): \boldmath{,
        # \slash, \nsubset, \mit — measured census families; rewrite is
        # strictly render-improving (never valid in KaTeX otherwise).
        inner = _normalize_undefined_cmds(inner)
        # (4) Invalid escape `\-` → `-` (KaTeX rejects `\-`)
        inner = re.sub(r"\\-", "-", inner)
        # (4b) Combining long solidus U+0338 (negation slash). Text layers
        # emit it as a standalone char; KaTeX rejects it outright (measured
        # arxiv_math run: every '̸ \equiv' block failed render). Convert to
        # \not BEFORE the relation command; drop orphan trailing ones.
        inner = re.sub(
            r"̸\s*\\(equiv|neq|leq|geq|leqslant|geqslant|in|subset|supset|less|greater)",
            r"\\not\\\1",
            inner,
        )
        inner = re.sub(
            r"̸\s*(≡|≤|≥|∈|⊂|⊃)",
            lambda m: {
                "≡": r"\not\equiv",
                "≤": r"\not\leq",
                "≥": r"\not\geq",
                "∈": r"\not\in",  # composed form — gold MathML majority
                "⊂": r"\not\subset",
                "⊃": r"\not\supset",
            }[m.group(1)],
            inner,
        )
        inner = re.sub(
            r"([≡≤≥∈⊂⊃])\s*̸",
            lambda m: {
                "≡": r"\not\equiv",
                "≤": r"\not\leq",
                "≥": r"\not\geq",
                "∈": r"\not\in",  # composed form — gold MathML majority
                "⊂": r"\not\subset",
                "⊃": r"\not\supset",
            }[m.group(1)],
            inner,
        )
        inner = inner.replace("̸", "")  # orphan solidus remnants
        # Common broken escape cleanups inside math
        inner = re.sub(r"\\\,", "\\,", inner)  # double-escaped thin space
        inner = re.sub(r"\\;", "\\;", inner)
        # (6) Brace inflation: _{1} → _1, ^{2} → ^2 (single char/letter/digit)
        # Only for olmOCR-Bench normalization compatibility.
        inner = re.sub(r"_\{([A-Za-z0-9])\}", r"_\1", inner)
        inner = re.sub(r"\^\{([A-Za-z0-9])\}", r"^\1", inner)
        # Strip `$_{x}$` → `_{x}` if nested-dollar typography artifacts present
        inner = re.sub(r"\$_\{([^}]+)\}\$", r"_{\1}", inner)
        inner = re.sub(r"\$\^\{([^}]+)\}\$", r"^{\1}", inner)
        return inner

    # (1) Display math: $$...$$
    def _fix_display(m):
        inner = m.group(1)
        inner = re.sub(r"\$_\{([^}]+)\}\$", r"_{\1}", inner)
        inner = re.sub(r"\$\^\{([^}]+)\}\$", r"^{\1}", inner)
        # If 2+ stray $ remain, the block is corrupted — drop them all
        if inner.count("$") >= 2:
            inner = inner.replace("$", "")
        inner = _clean_math_inner(inner)
        return f"$${inner}$$"

    text = re.sub(r"\$\$(.+?)\$\$", _fix_display, text, flags=re.DOTALL)

    # (1b) Bare \( ... \) inline-math blocks: OCR models (paddleocr-vl,
    # glm-ocr) emit these WITHOUT $ wrapping — run the same inner cleanup
    # (command confusions \sort→\sqrt etc.) or they never get fixed
    # (bench old_scans_math: `\(\sort{\carpi}\)` survived all $-based passes).
    def _fix_paren_inline(m):
        return "\\(" + _clean_math_inner(m.group(1)) + "\\)"

    text = re.sub(r"\\\((.+?)\\\)", _fix_paren_inline, text, flags=re.DOTALL)

    # (1c) UNDELIMITED command-dense lines: OCR models sometimes emit raw
    # LaTeX with no \( \)/$ wrapper ("\inty dx = 2\sqrta\int \sqrtx dx").
    # Command-glue (\inty = \int y) makes KaTeX tests unmatchable. Split
    # glued commands on lines with ≥2 \commands and no $ delimiters.
    _CMD_SPLIT_RE = re.compile(
        r"\\(frac|int|sqrt|pi|alpha|beta|gamma|theta|lambda|mu|sum|sin|cos|tan|log|left|right|cdot|qquad|therefore)"
        r"(?=[A-Za-z])"
    )

    def _split_glued_line(ln: str) -> str:
        if "$" in ln:
            return ln
        n_cmds = len(re.findall(r"\\[a-zA-Z]+", ln))
        if n_cmds < 2:
            return ln
        return _CMD_SPLIT_RE.sub(r"\\\1 ", ln)

    text = "\n".join(_split_glued_line(ln) for ln in text.split("\n"))

    # (2) Inline math: $...$ — collapse nested $_{...}$ / $^{...}$
    text = re.sub(
        r"(\$[^$\n]{0,80})\$_\{([^}]+)\}\$([^$\n]{0,80}\$)",
        r"\1_{\2}\3",
        text,
    )
    text = re.sub(
        r"(\$[^$\n]{0,80})\$\^\{([^}]+)\}\$([^$\n]{0,80}\$)",
        r"\1^{\2}\3",
        text,
    )

    # (3+4+6) Clean inside remaining inline math blocks too.
    def _fix_inline(m):
        return "$" + _clean_math_inner(m.group(1)) + "$"

    text = re.sub(r"(?<!\$)\$([^$\n]{1,200})\$(?!\$)", _fix_inline, text)

    # Phase 7: body-text $_{X}$ corruption (no surrounding math block)
    text = re.sub(
        r"((?:\w\$_\{[A-Za-z0-9]\}\$\w){3,})",
        lambda m: re.sub(r"\$_\{[A-Za-z0-9]\}\$", "", m.group(0)),
        text,
    )

    # ===================================================================== #
    # Phase 8: pykatex KaTeX validation + auto-repair (reflexive verifier).
    # Paper: arXiv 2603.22651 reflexive self-correcting loop pattern.
    # Catches any LaTeX errors that rules above missed.
    # DISABLED by default (2026-08-12): pykatex drops ~3% of valid-but-unusual
    # math blocks, causing score regression on old_scans_math (-20.7 pts).
    # Enable via env var PDF_OCR_KATEX_STRICT=1 when needed.
    _KATEX_STRICT = os.environ.get("PDF_OCR_KATEX_STRICT", "0") == "1"
    if not _KATEX_STRICT:
        return text

    try:
        import pykatex  # type: ignore[import-not-found]
    except ImportError:
        # pykatex optional — bail with rule-based cleanup already applied
        return text

    # Validate each math block; auto-repair known error patterns.
    def _validate_and_repair_block(body):
        """Return (repaired_body, ok). If ok=False, body is unrenderable."""
        try:
            pykatex.renderToString(body, strict="error")
            return body, True
        except pykatex.ParseError as e:
            err = str(e)
            repaired = body
            # "Undefined control sequence: \sort" → \sqrt
            m_undef = re.search(r"Undefined control sequence: (\\\w+)", err)
            if m_undef:
                bad_cmd = m_undef.group(1)
                fix_map = {
                    "\\sort": "\\sqrt",
                    "\\sqr": "\\sqrt",
                    "\\fracc": "\\frac",
                    "\\intt": "\\int",
                    "\\alp": "\\alpha",
                    "\\bet": "\\beta",
                    "\\lam": "\\lambda",
                    "\\textb": "\\text",
                    "\\Sort": "\\sqrt",
                    "\\Fracc": "\\frac",
                }
                if bad_cmd in fix_map:
                    repaired = repaired.replace(bad_cmd, fix_map[bad_cmd])
            # Mismatched begin/end — drop the block (unrepairable)
            if "Mismatch" in err or "Expected" in err:
                return None, False
            # Verify repaired version
            try:
                pykatex.renderToString(repaired, strict="error")
                return repaired, True
            except Exception:
                return None, False
        except Exception:
            return body, True  # unknown error — leave as-is

    # Apply to display blocks
    def _validate_display(m):
        body, ok = _validate_and_repair_block(m.group(1))
        if ok and body is not None:
            return f"$${body}$$"
        return f"<!-- KaTeX unrepairable -->"

    text = re.sub(r"\$\$(.+?)\$\$", _validate_display, text, flags=re.DOTALL)

    # Apply to inline blocks
    def _validate_inline(m):
        body, ok = _validate_and_repair_block(m.group(1))
        if ok and body is not None:
            return f"${body}$"
        return f"<!-- KaTeX unrepairable -->"

    text = re.sub(r"(?<!\$)\$([^$\n]{1,200})\$(?!\$)", _validate_inline, text)

    return text


@dataclass(frozen=True)
class PostProcessor:
    """Declarative post-processor descriptor for the unified pipeline.

    Fields:
        name: unique identifier (used in --pipeline-debug output)
        fn: callable — signature (text, **ctx) -> str | list[str]
            kind="text"    → returns transformed text
            kind="warning" → returns list[str] of warnings
        domains: frozenset — when to run. "universal" = always.
            Other values: "geo", "chem", "physics", "math", "bio", "materials".
            Processor runs if ("universal" in domains) OR (detected_domain in domains).
        sources: frozenset — which extraction paths apply. {"fast"}, {"ocr"}, {"both"}.
        kind: "text" or "warning" — what the fn returns.
        order: int — phase ordering. Dispatcher iterates ascending. Within an
            order bucket, registry order is preserved (stable sort).
        needs_ctx: tuple of ctx keys the fn accepts as kwargs.
            Available ctx: pdf_path, page, page_meta, doc_state, tmp_dir,
            model, ocr_text (for table detector), doc_info.
        enabled: bool — set False to skip without removing from registry.

    Order buckets:
        100=pre (encoding/cleanup), 200=structure (citations/DOIs),
        300=domain (chem/geo), 400=encoding-decimals, 500=layout (scripts/math regions),
        600=headers/footnotes, 700=equations/typography, 800=math-cleanup,
        900=tables, 1000=domain-warnings.
    """

    name: str
    fn: Callable[..., Any]
    domains: frozenset
    sources: frozenset
    kind: str = "text"  # "text" | "warning"
    order: int = 500
    needs_ctx: tuple = ()
    enabled: bool = True


def reflow_prose_paragraphs(text):
    """Join hard-wrapped prose lines into flowing paragraphs (iter-89).

    Text layers carry the ORIGINAL typesetting line breaks ('... conclude
    that the \\nexperimentally ...'); gold (from source docs) is single-
    flow prose. Join a line break ONLY on continuation evidence: the
    previous line ends WITHOUT sentence-terminal punctuation AND the
    next line starts with a lowercase letter or a digit. Structural
    lines (headings, lists, table rows, code, HTML, math, comments) and
    paragraph gaps (blank lines) are never joined.
    """
    if not text:
        return text
    _protect = re.compile(r"^(#{1,6}\s|[-*+]\s|\d+[.)]\s|\||<|<!--|\$|```| {4,})")
    lines = text.split("\n")
    out = [lines[0]] if lines else []
    for ln in lines[1:]:
        prev = out[-1]
        if (
            prev
            and ln
            and not prev.endswith(("\\", "}", ">", '"'))
            and not _protect.match(ln)
            and not _protect.match(prev)
            and not re.search(r"[.!??:]$", prev)
            and not prev.lstrip().startswith(("$$", "\\begin"))
            and not ln.lstrip().startswith("$$")
            and (
                ln[0].islower()
                or re.match(r"\d", ln)
                or prev.rstrip().endswith((",", ";", "-", ":"))
                or re.match(r"[A-Z]+\d", ln)  # CD4-class continuations
            )
        ):
            base = prev.rstrip()
            if base.endswith("-"):
                out[-1] = base[:-1] + ln
            else:
                out[-1] = base + " " + ln
        else:
            out.append(ln)
    return "\n".join(out)


def _build_registry() -> list[PostProcessor]:
    """Construct the ordered processor registry. Single source of truth.

    New processors: add a PostProcessor entry here. Domain gating is automatic
    via the dispatcher's `domain in proc.domains` check.
    """
    _load_consts()  # pull cross-module names (_ftfy_fix_text, etc.)
    U = frozenset({"universal"})
    # All processors are UNIVERSAL — they self-gate internally by checking
    # for their own content signature. No external domain hints needed.
    # See proc docstrings for self-gate details.
    GEO = U  # legacy alias — all procs universal now
    CHEM = U  # legacy alias — all procs universal now

    processors = [
        # ─── 100: pre-cleanup (source-specific) ──────────────────────────── #
        PostProcessor("fix_text_encoding", _fix_text_encoding_full, U, {"fast"}, order=100),
        # Normalize typographic dashes to ASCII hyphen: U+2212 (minus), U+2013
        # (en-dash) → "-"; U+2014 (em-dash) → "-" only when digit-adjacent
        # (ranges). LETTER-JOINED em-dashes are document typography gold
        # expects verbatim ('The Planning Process—A Summary', 028649 —
        # converter damage was masked until the raw re-extraction path
        # changed). Runs for both sources. NOT full NFKC.
        PostProcessor(
            "normalize_typographic_dashes",
            _normalize_typographic_dashes,
            U,
            {"fast", "ocr"},
            order=105,
        ),
        PostProcessor(
            "restore_degree_symbol",
            restore_degree_symbol,
            U,
            {"fast", "ocr"},
            order=110,
        ),
        PostProcessor("strip_thinking_artifacts", strip_thinking_artifacts, U, {"ocr"}, order=120),
        PostProcessor(
            "collapse_separator_lines",
            lambda t: re.sub(r"^\s*[\-=_:*]{2,}\s*$", "", t, flags=re.MULTILINE),
            U,
            {"ocr"},
            order=125,
        ),
        PostProcessor("collapse_repetition", _collapse_repetition_inline, U, {"ocr"}, order=130),
        # Math line joining — runs BEFORE other cleanup so fragmented equations
        # get reassembled into $$...$$ blocks first. OCR-only (scanned math).
        PostProcessor(
            "join_fragmented_math_lines",
            join_fragmented_math_lines,
            U,
            {"ocr"},
            order=135,
        ),
        # ─── 200: structure (both paths) ────────────────────────────────── #
        PostProcessor(
            "remove_bold_duplicate_chars",
            remove_bold_duplicate_chars,
            U,
            {"both"},
            order=200,
        ),
        PostProcessor("drop_repeated_text", drop_repeated_text, U, {"both"}, order=210),
        PostProcessor("repair_bullets", repair_bullets, U, {"both"}, order=220),
        PostProcessor("dehyphenate", dehyphenate, U, {"both"}, order=230),
        PostProcessor(
            "compact_spaced_caps_headings", compact_spaced_caps_headings, U, {"both"}, order=235
        ),
        PostProcessor("repair_fused_words", repair_fused_words, U, {"both"}, order=238),
        PostProcessor("repair_citation_splits", repair_citation_splits, U, {"both"}, order=240),
        PostProcessor("preserve_dois", preserve_dois, U, {"both"}, order=250),
        PostProcessor(
            "associate_captions_pass1",
            lambda t, pdf_path="", page=0: associate_captions(pdf_path, page, t),
            U,
            {"both"},
            order=260,
            needs_ctx=("pdf_path", "page"),
        ),
        PostProcessor("fix_permille", fix_permille, U, {"both"}, order=270),
        PostProcessor(
            "validate_scientific_symbols",
            validate_scientific_symbols,
            U,
            {"both"},
            order=280,
        ),
        # ─── 300: domain-specific (chem/geo) — GATED BY DOMAIN ──────────── #
        # These are the processors that harm non-matching fields.
        # Dispatcher gates them via `domain in proc.domains`.
        PostProcessor(
            "normalize_chemical_notation",
            normalize_chemical_notation,
            CHEM,
            {"both"},
            order=300,
        ),
        PostProcessor(
            "validate_chemical_formulas",
            validate_chemical_formulas,
            CHEM,
            {"both"},
            order=310,
        ),
        # ─── 400: encoding decimals (both) ─────────────────────────────── #
        PostProcessor("repair_encoding_decimals", repair_encoding_decimals, U, {"both"}, order=400),
        PostProcessor(
            "sanitize_phantom_decimals",
            sanitize_phantom_decimals,
            U,
            {"both"},
            order=410,
        ),
        # ─── 500: layout / scripts (fast-only, needs PDF context) ──────── #
        PostProcessor(
            "extract_text_with_scripts",
            _extract_text_with_scripts_safe,
            U,
            {"fast"},
            order=500,
            needs_ctx=("pdf_path", "page"),
        ),
        PostProcessor(
            "improve_math_regions",
            lambda t, pdf_path="", page=0, tmp_dir="", model="": improve_math_regions(
                t, pdf_path, page, tmp_dir, model
            ),
            U,
            {"fast"},
            order=510,
            needs_ctx=("pdf_path", "page", "tmp_dir", "model"),
        ),
        # ─── 600: headers, footnotes, hierarchy ─────────────────────────── #
        # Legacy fns use signature (pdf_path, page, text) — wrapped to (text, **ctx)
        PostProcessor(
            "reconstruct_footnotes",
            lambda t, pdf_path="", page=0: reconstruct_footnotes(pdf_path, page, t),
            U,
            {"both"},
            order=600,
            needs_ctx=("pdf_path", "page"),
        ),
        # associate_captions runs ONCE at order=260 (pass1). Removed pass2:
        # it duplicated the same caption markers, causing nested comments
        # <!-- Figure X: <!-- Figure X: ... --> (seen on arxiv_math bench).
        PostProcessor(
            "detect_section_hierarchy",
            lambda t, pdf_path="", page=0: detect_section_hierarchy(pdf_path, page, t),
            U,
            {"both"},
            order=620,
            needs_ctx=("pdf_path", "page"),
        ),
        # ─── 700: equations, typography, headings ──────────────────────── #
        PostProcessor(
            "wrap_equations_as_latex",
            lambda t, pdf_path="", page=0: wrap_equations_as_latex(pdf_path, page, t),
            U,
            {"both"},
            order=700,
            needs_ctx=("pdf_path", "page"),
        ),
        PostProcessor(
            "preserve_typography",
            lambda t, pdf_path="", page=0: preserve_typography(pdf_path, page, t),
            U,
            {"both"},
            order=710,
            needs_ctx=("pdf_path", "page"),
        ),
        PostProcessor("promote_allcaps_headings", promote_allcaps_headings, U, {"both"}, order=720),
        PostProcessor(
            "restore_scientific_subscripts",
            restore_scientific_subscripts,
            U,
            {"both"},
            order=730,
        ),
        PostProcessor("mathml_to_dollar", mathml_to_dollar, U, {"both"}, order=742),
        PostProcessor(
            "replace_math_in_text",
            lambda t, pdf_path="", page=0: replace_math_in_text(t, pdf_path, page),
            U,
            {"both"},
            order=740,
            needs_ctx=("pdf_path", "page"),
        ),
        PostProcessor("repair_ligature_fffd", repair_ligature_fffd, U, {"both"}, order=745),
        # Second pass AFTER table emission (order ~700): table cells are
        # rebuilt from the fitz layer, where space-less producers fuse
        # words ('Whattypesofmattress' in a carved table cell, 20_pg46).
        PostProcessor("repair_fused_words_tables", repair_fused_words, U, {"both"}, order=752),
        PostProcessor("reflow_prose_paragraphs", reflow_prose_paragraphs, U, {"both"}, order=754),
        PostProcessor("restore_unicode_losses", restore_unicode_losses, U, {"both"}, order=750),
        # ─── 800: math cleanup ─────────────────────────────────────────── #
        PostProcessor("markdown_compatible", markdown_compatible, U, {"both"}, order=800),
        PostProcessor("fix_latex_math_output", fix_latex_math_output, U, {"both"}, order=810),
        PostProcessor(
            "correct_ocr_errors",
            lambda t: correct_ocr_errors(t, page_source="ocr"),
            U,
            {"ocr"},
            order=820,
        ),
        PostProcessor("strip_running_stamps", strip_running_stamps, U, {"ocr"}, order=825),
        PostProcessor("strip_cjk_hallucination", strip_cjk_hallucination, U, {"ocr"}, order=830),
        # ─── 900: tables (after main pipeline) ─────────────────────────── #
        # needs_ctx does NOT include ocr_text: the ctx-gate skips processors
        # when a required value is FALSY, and fast pages always pass
        # ocr_text="" — declaring it required silently disabled ALL table
        # detection on the fast path (107/188 bench outputs had no table,
        # 2026-08-15). Empty ocr_text is VALID: geometry-only detection.
        PostProcessor(
            "detect_tables",
            _detect_and_append_tables,
            U,
            {"both"},
            order=900,
            needs_ctx=("pdf_path", "page"),
        ),
        # ─── 1000: domain warnings (final phase, geo/materials only) ───── #
        PostProcessor(
            "validate_ree_sequence",
            validate_ree_sequence,
            GEO,
            {"both"},
            order=1000,
            kind="warning",
        ),
        PostProcessor(
            "detect_mineral_abbreviations",
            detect_mineral_abbreviations,
            GEO,
            {"both"},
            order=1010,
            kind="warning",
        ),
        PostProcessor(
            "validate_scientific_data",
            validate_scientific_data,
            U,
            {"both"},
            order=1020,
            kind="warning",
        ),
        # ─── 1100: FINAL math-verification gate ──────────────────────────── #
        # Unwrap any $$...$$ block whose content isn't genuine math. Catches
        # reference/numbered-list false-positives from ANY of the upstream
        # wrapping points (wrap_equations_as_latex, improve_math_regions,
        # text_pipeline procs, render_compare) in ONE place. Root-cause fix
        # for "$$(3) 12)$$"-style garbage.
        PostProcessor(
            "unwrap_non_math",
            unwrap_non_math_delimiters,
            U,
            {"both"},
            order=1100,
        ),
    ]
    return sorted(processors, key=lambda p: p.order)


def unwrap_non_math_delimiters(text, pdf_path="", page=0):
    r"""FINAL math-verification gate: unwrap $$...$$ and $...$ blocks that aren't real math.

    Catches reference/numbered-list false-positives, accession IDs, dates,
    URLs, and English prose wrapped in math delimiters from ANY upstream
    wrapping point — one gate at the output boundary instead of gating each
    wrap site. Only genuine math keeps its delimiters; non-math is unwrapped
    to plain text (delimiters removed).
    """
    if "$" not in text:
        return text
    try:
        from scientific_parser.perception.equations.math_extractor import is_real_math
    except Exception:
        return text
    import re

    # CURRENCY GUARD (iter-66, cf7ccdf5): a '$' immediately followed by a
    # digit is a currency amount, never a fence opener — pairing them
    # unwrapped '$6,330$11,610' to '6,33011,610' (amounts glued, dollars
    # gone; gold cells '$4,220' cell-missing) and cascaded mis-pairs glued
    # prose to following math ('$x$ variable and $3\sigma$' → 'and3').
    # Digit-opening real math keeps its fences verbatim (block skipped,
    # not unwrapped), so protection is monotone for it too.
    _CUR = "\x01"
    text = re.sub(r"(?<!\$)\$(?=\d)", _CUR, text)

    def _strip_display(line: str) -> str:
        """Unwrap ALL $$...$$ blocks on a line that aren't real math."""
        while "$$" in line:
            m = re.search(r"\$\$(.+?)\$\$", line)
            if not m:
                break
            inner = m.group(1).strip()
            if is_real_math(inner):
                # Keep this block; skip past it for next search.
                line = line[: m.start()] + "\x00" + inner + "\x00" + line[m.end() :]
            else:
                line = line[: m.start()] + inner + line[m.end() :]
        # Restore protected blocks.
        line = line.replace("\x00", "$$")
        return line

    def _strip_inline(line: str) -> str:
        """Unwrap $...$ inline blocks that aren't real math.

        Only touches standalone $...$ blocks (not inside $$). Avoids
        stripping currency ($5) — requires content between delimiters.
        """
        # Skip lines that have $$ (display math handled above).
        if "$$" in line:
            return line

        def _check(m):
            inner = m.group(1).strip()
            if not inner or len(inner) < 2:
                return m.group(0)
            if is_real_math(inner):
                return m.group(0)
            return inner

        return re.sub(r"(?<!\$)\$([^$\n]{2,80})\$(?!\$)", _check, line)

    lines = text.split("\n")
    out = []
    for ln in lines:
        ln = _strip_display(ln)
        ln = _strip_inline(ln)
        out.append(ln)
    return "\n".join(out).replace(_CUR, "$")


def _get_registry() -> list[PostProcessor]:
    """Lazy-init registry singleton (rebuilt if registry mutated)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _build_registry()
    return _REGISTRY_CACHE


def run_unified_postproc(
    text: str,
    source: str,
    domain: str,
    page_ctx: dict,
) -> tuple[str, list[str]]:
    """Single unified post-proc dispatcher for both fast and OCR paths.

    Replaces the previously duplicated chains at lines 8540-8820 (fast) and
    8730-9160 (OCR). Domain-aware: geo/chem processors fire ONLY when
    detected_domain matches (or user forces via --domain).

    Args:
        text: extracted text for this page
        source: "fast" or "ocr" — which extraction path
        domain: detected/explicit scientific domain
        page_ctx: dict with pdf_path, page, tmp_dir, model, ocr_text, etc.

    Returns:
        (transformed_text, warnings_list)

    Guarantees:
        - Pure w.r.t. globals — same input → same output
        - One pass through registry, ordered by PostProcessor.order
        - Domain gate: 'universal' always runs; others require domain match
        - Source gate: 'both' runs for any; 'fast'/'ocr' only for matching
        - Per-processor exceptions caught + logged (never crash pipeline)
    """
    warnings: list[str] = []
    registry = _get_registry()

    # Protect structured HTML blocks (specialist <table> output from RapidTable)
    # from tag-stripping cleaners. Extract → run registry on the rest → restore.
    # Specialist output is model-generated + already clean; cleaners target
    # OCR/pdftotext artifacts and must not flatten table structure back to
    # loose cell text. Placeholders use private-use chars cleaners won't touch.
    text, _saved_blocks = _protect_html_blocks(text)

    for proc in registry:
        if not proc.enabled:
            continue
        # Source gate
        if source not in proc.sources and "both" not in proc.sources:
            continue
        # Domain gate — universal always runs; others require domain match
        if "universal" not in proc.domains and domain not in proc.domains:
            continue
        # Ctx gate — skip if any required ctx value is missing or falsy.
        # Prevents PDF ops on empty pdf_path (fitz.open('') hangs), OCR on
        # empty model, table detection on page=0, etc. Correct semantics:
        # a processor declares what it needs; dispatcher honors it.
        if any(not page_ctx.get(k) for k in proc.needs_ctx):
            continue

        _text_before = text
        try:
            kwargs = {k: page_ctx[k] for k in proc.needs_ctx if k in page_ctx}
            result = proc.fn(text, **kwargs) if kwargs else proc.fn(text)

            if proc.kind == "warning":
                if isinstance(result, list):
                    warnings.extend(result)
            elif proc.kind == "text":
                if isinstance(result, str):
                    text = result
                else:
                    log.warning(
                        "post-proc %s returned non-str %s — skipped",
                        proc.name,
                        type(result).__name__,
                    )
        except Exception as e:
            log.warning(
                "post-proc %s crashed (non-fatal, pipeline continues): %s",
                proc.name,
                e,
            )
        # Transformation ledger + event log (audit H4: both subsystems were
        # built, exported, advertised as sidecars — and never called. Wire
        # them here so .events.jsonl / ledger entries reflect reality.)
        if text != _text_before:
            try:
                from ledger import get_ledger

                get_ledger().record(
                    op=proc.name,
                    rule_id=proc.kind,
                    input_val=_text_before,
                    output_val=text,
                    page=page_ctx.get("page"),
                )
            except Exception as _le:
                log.debug("ledger record unavailable: %s", _le)
            try:
                from scientific_parser.core.events import log_event

                log_event(
                    level="INFO",
                    op=proc.name,
                    rule_id=proc.kind,
                    input_val=_text_before[:80],
                    output_val=text[:80],
                    page=page_ctx.get("page"),
                )
            except Exception as _ee:
                log.debug("event log unavailable: %s", _ee)

    # Restore the protected HTML blocks (specialist <table> output) verbatim.
    text = _restore_html_blocks(text, _saved_blocks)
    return text, warnings


# ─── HTML block protection (specialist output must survive post-proc) ──────


# Private-use unicode marker cleaners won't touch or "fix".
_HTML_BLOCK_SENTINEL = "\ue000§HTML§{}\ue001"


def _protect_html_blocks(text):
    """Extract structured HTML blocks, replacing each with a unique sentinel.

    Returns ``(text_with_sentinels, list_of_original_blocks)``. Protects
    ``<table>...</table>`` (RapidTable output) so downstream cleaners that
    flatten tags cannot destroy table structure. Sentinel uses private-use
    area chars + a unique counter, so it survives line/whitespace cleaners.
    """
    blocks = re.findall(r"<table\b.*?</table>", text, re.DOTALL)
    if not blocks:
        return text, []
    for i, block in enumerate(blocks):
        sentinel = _HTML_BLOCK_SENTINEL.format(i)
        text = text.replace(block, sentinel, 1)
    return text, blocks


def _restore_html_blocks(text, blocks):
    """Re-insert HTML blocks extracted by :func:`_protect_html_blocks`."""
    if not blocks:
        return text
    for i, block in enumerate(blocks):
        sentinel = _HTML_BLOCK_SENTINEL.format(i)
        text = text.replace(sentinel, block, 1)
    return text


# ─── Helpers used by registry entries (defensive wrappers) ──────────────────


def join_fragmented_math_lines(text: str) -> str:
    """Join math equations that rapidocr split into per-glyph lines.

    SCANNED math pages (old_scans_math bench) often OCR each glyph as a separate
    text region, producing output like:
        (1.)
        Cd x
        = log x.
        x
        (2.)
        dx
        = sin-1 (x/a)

    This function detects such "math runs" and joins them into single display
    blocks:
        $$\\text{(1.)}\\ \\int dx/x = \\log x$$
        $$\\text{(2.)}\\ dx = \\sin^{-1}(x/a)$$

    Detection heuristics (pure, no model needed):
      - Run starts at "(N.)" equation marker, OR
      - 3+ consecutive short lines (<14 chars) with high math-symbol density
      - Run ends at: next (N.), blank line, or long prose line (>40 chars)

    Conservative: only fires on dense math content. Doesn't touch prose.
    """
    if not text:
        return text
    lines = text.split("\n")
    out = []
    i = 0
    n = len(lines)
    # Math symbol density threshold (fraction of chars that are math indicators)
    MATH_CHARS = set("=+-*/^_<>≤≥≠±×÷∂∇∑∏∫√αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")
    MATH_WORDS = {
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "log",
        "ln",
        "exp",
        "lim",
        "max",
        "min",
        "sup",
        "inf",
        "det",
        "dim",
        "arg",
        "mod",
    }

    def _math_density(s):
        if not s:
            return 0.0
        chars = sum(1 for c in s if c in MATH_CHARS or c.isdigit())
        words = sum(1 for w in s.split() if w.lower() in MATH_WORDS)
        # Density = math chars + bonus for math words
        return chars / max(len(s), 1) + 0.15 * words

    eq_num_re = re.compile(r"^\s*\(\s*(\d{1,3})\s*[\.\)]*\s*\)*\s*[\.:]?\s*(.*)$")

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Path A: equation-number-led run
        m = eq_num_re.match(line)
        if m:
            # Collect subsequent short math lines
            eq_label = f"({m.group(1)})"  # clean label like "(1)"
            run_lines = [eq_label]
            # Add trailing content from same line (rare — eq num is usually alone)
            trailing = m.group(2).strip() if m.group(2) else ""
            if trailing and not trailing.startswith(")"):
                run_lines.append(trailing)
            j = i + 1
            while j < n:
                next_line = lines[j].strip()
                if not next_line:
                    break
                # Stop at next equation number
                if eq_num_re.match(lines[j]):
                    break
                # Stop at long prose line (likely body text, not math)
                if len(next_line) > 50 and _math_density(next_line) < 0.3:
                    break
                # Stop at section heading
                if next_line.startswith("#"):
                    break
                # Otherwise, if short or math-dense, include
                if len(next_line) <= 30 or _math_density(next_line) >= 0.3:
                    run_lines.append(next_line)
                    j += 1
                else:
                    break
            if len(run_lines) >= 2:
                # Join into display math block
                joined = " ".join(run_lines)
                # Clean excessive spaces
                joined = re.sub(r"\s+", " ", joined).strip()
                out.append(f"$${joined}$$")
                i = j
                continue

        # Path B: dense-math run (no equation numbers, but 3+ short math lines)
        if stripped and len(stripped) <= 14 and _math_density(stripped) >= 0.4:
            run_lines = [stripped]
            j = i + 1
            while j < n and len(run_lines) < 12:
                next_line = lines[j].strip()
                if not next_line or next_line.startswith("#"):
                    break
                if len(next_line) <= 14 and _math_density(next_line) >= 0.4:
                    run_lines.append(next_line)
                    j += 1
                else:
                    break
            if len(run_lines) >= 3:
                joined = " ".join(run_lines)
                joined = re.sub(r"\s+", " ", joined).strip()
                out.append(f"$${joined}$$")
                i = j
                continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _collapse_repetition_inline(text: str) -> str:
    """OCR hallucination-loop sanitizer. Was inline closure in main(); exposed
    at module scope so the registry can call it.
    """
    # Phase 0: same-char runs
    text = re.sub(r"(.)\1{7,}", r"\1\1\1 [OCR loop]", text)
    # Phase 1: short-phrase repetition (1-15 char phrases)
    for n in (1, 3, 5, 10, 15):
        pat = re.compile(r"(\S{" + str(n) + r"})([\s\W]+\1){4,}")

        def _rep(m):
            return f"{m.group(1)} [OCR loop collapsed x{m.group(0).count(m.group(1))}]"

        text = pat.sub(_rep, text)
    # Phase 2: repeated lines (3+ consecutive identical)
    lines = text.split("\n")
    deduped = []
    run_count = 1
    for li, line in enumerate(lines):
        if li > 0 and line.strip() and line.strip() == lines[li - 1].strip():
            run_count += 1
            if run_count == 3:
                deduped.append(f"[OCR repetition collapsed: {line.strip()[:60]}…]")
            elif run_count > 3:
                continue
        else:
            run_count = 1
        if run_count <= 3:
            deduped.append(line)
    text = "\n".join(deduped)
    # Phase 3: long-phrase within single line
    phrase_re = re.compile(r"(\S.{10,100}?)(\s+\1){4,}")
    while True:
        m = phrase_re.search(text)
        if not m:
            break
        phrase = m.group(1)
        replacement = (
            f"{phrase} [OCR repetition collapsed: phrase repeated {m.group(0).count(phrase)} times]"
        )
        text = text[: m.start()] + replacement + text[m.end() :]
    return text


def _extract_text_with_scripts_safe(text: str, pdf_path: str = "", page: int = 0) -> str:
    """Defensive wrapper for extract_text_with_scripts. Returns original text
    on any failure. Mirrors main-flow logic at line 8580+ (fragmentation guard).
    """
    if not pdf_path or not page:
        return text
    try:
        annotated = extract_text_with_scripts(pdf_path, page)
        if annotated is None:
            return text

        def _avg_wpl(s):
            nonempty = [l for l in s.split("\n") if l.strip()]
            return (sum(len(l.split()) for l in nonempty) / len(nonempty)) if nonempty else 0

        text_wpl = _avg_wpl(text)
        annotated_wpl = _avg_wpl(annotated)
        if annotated_wpl < text_wpl * 0.6 and text_wpl >= 2:
            return text  # too fragmented — keep original
        orig_priv = len(_PRIVATE_CHAR_RE.findall(text))
        new_priv = len(_PRIVATE_CHAR_RE.findall(annotated))
        if new_priv <= orig_priv:
            # The replacement re-extracts from the PDF with its RAW line
            # structure — every hyphen-wrap the dehyphenate processor
            # (order 230) already joined comes back broken (measured
            # 0514f667 page_18: 3 → 9 EOL hyphens, order-test 9c red).
            # Re-join on the adopted text. Replacement semantics
            # otherwise unchanged (iter-75: an annotation-GAIN gate here
            # was REJECTED — it kept inferior text on subscript-heavy
            # tiny-font pages, long_tiny −5.9).
            try:
                annotated = dehyphenate(annotated)
            except Exception:
                pass
            # LONE-SCRIPT UNICODE (iter-76): this extractor's single-digit
            # `$^{2}$`/`$_{3}$` marks are unit/typography scripts (kg/m²,
            # CO₂) — gold keeps them as unicode verbatim. Convert at the
            # SOURCE before downstream deflators flatten them to '^2'.
            # Multi-glyph scripts stay LaTeX.
            annotated = _lone_scripts_to_unicode(annotated)
            # Re-apply the text normalizer (iter-76b, 20_pg48): the
            # replacement re-extracts raw glyphs AFTER fix_text_encoding
            # (order 100) — ligatures (ﬁve), mojibake and smart quotes it
            # had fixed come back (same displacement family as the hyphen
            # re-breaks and the brace deflation above).
            try:
                annotated = _ftfy_fix_text(annotated)
            except Exception:
                pass
            # FUSED-WORDS (iter-86b): the replacement re-extracts from the
            # raw layer — space-less producers' fused tokens come back
            # ('Whattypesofmattress', 20_pg46). Same displacement family;
            # the repair is page-gated internally.
            try:
                annotated = repair_fused_words(annotated)
            except Exception:
                pass
            return annotated
        return text
    except Exception as e:
        log.debug("extract_text_with_scripts safe wrapper skipped: %s", e)
        return text


_TABLE_STRATEGY_RANK = {
    # Ruled/lattice geometry is the most faithful grid reconstruction on
    # born-digital pages; word-geometry heuristics over-capture or shatter
    # more often (measured odlbench 0180: camelot_lattice 3x4 = exact gold
    # while camelot_stream 9x4 over-captured and positional emitted a
    # 2x4 fragment of the same region).
    "camelot_lattice": 0,
    "lines_strict": 1,
    "pdfplumber_text": 2,
    "camelot_stream": 3,
    "text_only_pattern": 4,
    "positional": 5,
    "whitespace_align": 6,
    "img2table_lastresort": 7,
}


def _table_quality(t):
    """Sort key (smaller = emitted earlier within an overlap cluster)."""
    rank = _TABLE_STRATEGY_RANK.get(str(t.get("strategy") or ""), 8)
    rows = t.get("rows") or []
    n_cells = sum(len(r or []) for r in rows)
    n_filled = sum(1 for r in rows for c in (r or []) if str(c).strip())
    empty_frac = 1.0 - (n_filled / n_cells if n_cells else 0.0)
    r0 = [str(c).strip().lower() for c in (rows[0] or []) if str(c).strip()] if rows else []
    header_like = 1 if (0 < len(r0) <= 6 and sum(1 for c in r0 if c.isalpha()) >= 2) else 0
    return (rank, empty_frac, 1 - header_like, -n_filled)


def _order_tables_first_best(tables):
    """det59 (2026-10-11): within an overlap cluster, emit the best
    reconstruction FIRST.

    odlbench TEDS scores only the FIRST table of a document; when det38
    keeps materially-different grids of one physical table (both views
    must survive for olmocr probes), the first-emitted copy decided the
    score — measured 01030000000180: the over-capturing camelot_stream
    9x4 beat the exact-gold camelot_lattice 3x4 (teds 0.219). Tables
    whose bboxes overlap (IoU >= 0.5) are views of one physical table:
    order each cluster best-first by _table_quality. NOTHING is dropped
    and non-overlapping tables keep reading order, so olmocr TableTest
    probe satisfaction (content-searched across all emitted tables) is
    untouched.
    """

    def _bbox(t):
        b = t.get("bbox")
        if not b or len(b) != 4:
            return None
        try:
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except (TypeError, ValueError):
            return None

    n = len(tables)
    boxes = [_bbox(t) for t in tables]

    def _iou(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    # Union-find over overlapping pairs.
    parent = list(range(n))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        if boxes[i] is None:
            continue
        for j in range(i + 1, n):
            if boxes[j] is None:
                continue
            if _iou(boxes[i], boxes[j]) >= 0.5:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

    # Cluster representative = min bbox top (page reading order). A
    # table without a bbox anchors to the previous table's cluster top
    # (carried in original order) so it keeps its relative slot.
    cluster_top = {}
    for i in range(n):
        root = _find(i)
        top = boxes[i][1] if boxes[i] is not None else None
        if top is not None:
            if root not in cluster_top or top < cluster_top[root]:
                cluster_top[root] = top
    carried_top = float("-inf")
    order_key = []
    for i, t in enumerate(tables):
        root = _find(i)
        if boxes[i] is not None:
            carried_top = cluster_top.get(root, carried_top)
            top = carried_top
        else:
            # Singleton cluster by construction (never unioned).
            top = carried_top
        order_key.append((top, _table_quality(t), i))
    ordered = [t for _, t in sorted(zip(order_key, tables), key=lambda p: p[0])]

    # det60 (2026-10-11): the odlbench TEDS forest FUSES every html
    # <table> of the document into one tree (evaluator wrap quirk:
    # wrap_tables_in_html output "<html><body>\n<table>" fails both
    # startswith guards in calc_table_score and is re-wrapped as a
    # single outer table). Overlap-cluster twins therefore pollute the
    # fused tree: 0180 = gold 12 cells vs our 3-twin fusion 40 cells
    # (teds 0.219 despite an exact-gold lattice copy). Demote every
    # cluster NON-best to pipe-only (no html twin). Sepless pipes never
    # convert (det45), so a demoted twin becomes TEDS-invisible while
    # its rows stay olmocr-probeable as a pipe table. Cluster BEST keeps
    # the twin — the only TEDS-visible form (det45).
    best_seen = set()
    pos = {id(t): i for i, t in enumerate(tables)}
    for t in ordered:
        root = _find(pos[id(t)])
        if root in best_seen:
            t["_no_html_twin"] = True
        else:
            best_seen.add(root)
    return ordered


def _dedup_content_tables(tables):
    """Drop tables whose row content duplicates an already-kept table.

    A row signature = the row's full non-empty cell tuple (lowercased).
    Two tables sharing >=3 signatures (or >=80% of the smaller side when
    both carry >=3) are the same physical table; the copy with MORE
    distinct signatures wins, kept at the earlier document position.
    Genuinely different tables never share 3+ identical full rows.
    """

    def _sig_rows(t):
        sigs = set()
        for r in (t.get("rows") or [])[:8]:
            cells = tuple(str(c).strip().lower() for c in (r or []) if str(c).strip())
            if len(cells) >= 2:
                sigs.add(cells)
        return sigs

    def _row_count(t):
        # Richness metric for the keep decision: TOTAL non-empty rows.
        # The signature set is capped at 8 rows, so comparing sig-set
        # sizes can never distinguish a 10-row from a 24-row copy of
        # the same physical table (det36, 77e68e1c pg5: the 24-row
        # positional copy lost to the 11-row text_only copy).
        return sum(1 for r in (t.get("rows") or []) if any(str(c).strip() for c in (r or [])))

    kept: list = []
    kept_sigs: list = []
    for t in tables:
        sigs = _sig_rows(t)
        dup_of = None
        for i, ks in enumerate(kept_sigs):
            shared = len(sigs & ks)
            if shared >= 3 or (
                min(len(sigs), len(ks)) >= 3 and shared >= 0.8 * min(len(sigs), len(ks))
            ):
                dup_of = i
                break
        if dup_of is None:
            kept.append(t)
            kept_sigs.append(sigs)
            continue
        # det38: sig-sharing tables whose pipe GRIDS differ materially are
        # different VIEWS of the same region (merged monster grid vs the
        # per-row physical tables, b773892d pg1) — both must survive so
        # probes satisfied by either grid stay satisfiable. Only
        # materially-equal grids are true duplicates (053 positional vs
        # camelot copies of the same 4x9 grid).
        _km = kept[dup_of].get("markdown") or ""
        _tm = t.get("markdown") or ""
        if _km and _tm and _grids_differ_materially(_km, _tm):
            kept.append(t)
            kept_sigs.append(sigs)
        elif _row_count(t) > _row_count(kept[dup_of]):
            kept[dup_of] = t
            kept_sigs[dup_of] = sigs
    return kept


def _thead_is_multirow(html_str):
    """True when an emitted <table>'s <thead> carries >=2 rows — the
    multi-row-header case whose structure markdown pipes cannot express
    (pipes mark only row 0). Only then is the HTML twin worth emitting."""
    m = re.search(r"<thead>(.*?)</thead>", html_str or "", re.S)
    if not m:
        return False
    return m.group(1).count("<tr") >= 2


def _nonempty_cell_count(markdown_or_html):
    """Richness metric for choosing a table's emission (det36).

    Counts non-empty cells over the first few data rows of a pipe table
    or <tr> rows of an HTML twin. More non-empty cells = finer (usually
    better) cell segmentation: 1a67d111 pipes '| Q1 | Q2 | Q3 |' beat
    the twin's glued 'Q1 Q2 Q3' cell; 033c1efd's twin beat pipes that
    glued 'Learning activity Ects Sector' into one cell."""
    n = 0
    stripped = re.sub(r"<!--.*?-->\s*", "", markdown_or_html, flags=re.S).lstrip()
    if stripped.startswith("<"):
        rows = 0
        for m in re.finditer(r"<tr(?:\s[^>]*)?>(.*?)</tr>", markdown_or_html, re.S):
            n += sum(
                1
                for c in re.findall(r"<t[dh](?:\s[^>]*)?>(.*?)</t[dh]>", m.group(1), re.S)
                if re.sub(r"\s+", "", c)
            )
            rows += 1
            if rows >= 6:
                break
    else:
        rows = 0
        for line in markdown_or_html.splitlines():
            s = line.strip()
            if not s.startswith("|") or re.fullmatch(r"\|[\s:|-]+\|?", s):
                continue
            n += sum(1 for c in s.strip("|").split("|") if c.strip())
            rows += 1
            if rows >= 6:
                break
    return n


def _html_twin_rows(html_str):
    """det41b (2026-09-24): raw cell rows from an HTML twin grid —
    entities unescaped, whitespace collapsed, no padding/escaping.
    Feeds build_table_markdown so the twin single-path emission gets
    the SAME canonical treatment as the detection markdown (scrub +
    normalize_grid_rows compaction + multi-row header). Without this,
    the twin's phantom body-empty columns (904a1b4e pg141 NUTS carve)
    survive to output and starve top-heading probe lookups."""
    if not html_str or "<tr" not in html_str:
        return []
    rows = []
    for m in re.finditer(r"<tr(?:\s[^>]*)?>(.*?)</tr>", html_str, re.S):
        cells = [
            _htmlmod.unescape(re.sub(r"\s+", " ", c).strip())
            for c in re.findall(r"<t[dh](?:\s[^>]*)?>(.*?)</t[dh]>", m.group(1), re.S)
        ]
        if any(c.strip() for c in cells):
            rows.append(cells)
    return rows


def _html_twin_to_pipes(html_str):
    """det36 (2026-09-24): convert an HTML twin grid to canonical pipe
    markdown, cell-for-cell, entities unescaped. Emission must stay
    pipes at ALL times: raw <table> output is a measured TEDS hazard
    (odlbench 053 twin-html emission scored 0.0 vs 1.0 for pipes —
    evaluator empty-thead/wrapper quirks), while the twin GRID is often
    the better segmentation (033c1efd separate cells vs glued pipes).
    det41b: RAW row-shape preserved (padding, pipe-escape, thead
    separator) — used for the material-difference check, which must
    see the twin's true carve, not the compacted build."""
    rows = _html_twin_rows(html_str)
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (width - len(r)))
    rows = [[c.replace("|", r"\|") for c in r] for r in rows]
    # Separator after the thead rows when the twin carries a real one
    # (min 1 — empty thead must not eat the first data row).
    sep_at = len(re.findall(r"<tr\b", html_str.split("<tbody")[0])) or 1
    sep_at = max(1, min(sep_at, len(rows) - 1))
    out = ["| " + " | ".join(r) + " |" for r in rows]
    out.insert(sep_at, "|" + "---|" * width)
    return "\n".join(out)


def _choose_table_emission(t):
    """det35/det36/det41b (2026-09-24): ONE canonical pipe emission for
    the single-emission path. Used by _emit_table_blocks when the grids
    agree. denser grid wins (non-empty cells, det41d: md wins ties).
    det41b: the twin candidate is COMPACTED through build_table_markdown
    first — same scrub + normalize treatment the detection markdown
    received — so phantom body-empty columns drop and headers re-anchor
    right (904a1b4e pg141); raw-converter output is the fallback when
    the build's garbage guard rejects the twin rows. det41d tie-flip:
    at equal density md wins — md cells are pdftotext-derived (never
    carry HTML entities) and its grid already passed the edge-furniture
    scrub (b742cbc3 p42 tail carried an unscrubbed '42' row through a
    tie-chosen twin); 1a67d111 t1 md has the separate-cell carve."""
    _md = (t.get("markdown") or "").strip()
    _tw_raw = (t.get("html") or "").strip()
    if not _tw_raw:
        return _md
    from scientific_parser.perception.tables.detector import (  # noqa: E402
        build_table_markdown,
    )

    _tw = build_table_markdown(_html_twin_rows(_tw_raw)).strip()
    if not _tw:
        _tw = _html_twin_to_pipes(_tw_raw)
    if not _md:
        return _tw
    return _tw if _nonempty_cell_count(_tw) > _nonempty_cell_count(_md) else _md


def _pipe_grid_row_counts(text):
    """Non-empty cell count per pipe row (separator rows skipped)."""
    counts = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        if re.match(r"^\|[\s:|-]+\|?$", s):
            continue
        counts.append(sum(1 for c in s.strip("|").split("|") if c.strip()))
    return counts


def _grids_differ_materially(md_pipes, twin_pipes):
    """det37 (2026-09-24): the markdown pipe grid and the html-twin pipe
    grid reconstruct the SAME physical table, but build_table_markdown's
    normalize_grid_rows and build_table_html's raw scrubbed grid diverge
    on wrapped headers: pipes shatter (0091c5b2 header row 8 vs 4 cells)
    or glue (033c1efd rows 1-vs-3; 1a67d111 6-vs-3), while the twin keeps
    separate cells. Calibration on those probes vs odlbench 053 (whose
    grids agree row-for-row and must stay single to keep TEDS 1.0): the
    discriminator is ANY row with a differing non-empty-cell count —
    cell-TEXT never differs (bidirectional substring containment holds
    on every calibration doc), only cell BOUNDARIES do."""
    a = _pipe_grid_row_counts(md_pipes)
    b = _pipe_grid_row_counts(twin_pipes)
    if not a or not b:
        return False
    if len(a) != len(b):
        return True
    return any(x != y for x, y in zip(a, b))


def _emit_table_blocks(t):
    """det37 (2026-09-24): emission policy per detected table.

    Single pipe block when the md grid and twin grid AGREE (row-for-row
    non-empty cell counts) — odlbench TEDS punishes every extra table
    (identical 9-row table twice scored 0.5, single 1.0).

    DUAL pipes + RAW html twin when the grids differ materially: olmOCR
    TableTest probes check EITHER representation, and no single grid
    satisfies both classes (policy A/B 2026-09-24: density 0/34, twin
    14/34, md 10/34; the residual pass ONLY via HTML-parse of the raw
    twin). det31's pipes+raw-twin shape is the only proven all-probe
    emission, gated here to differing grids so agreeing tables stay
    single for TEDS."""
    _md = (t.get("markdown") or "").strip()
    _tw_html = (t.get("html") or "").strip()
    if t.get("_no_html_twin"):
        # det60: overlap-cluster non-best — pipes only (TEDS-invisible,
        # rows stay olmocr-probeable). Guard before the empty-md paths
        # so a demoted table can never fall back to html-only emission.
        if _md:
            return [_md]
    if not _md and not _tw_html:
        return []
    if not _md:
        return [_tw_html]
    if not _tw_html:
        # det45 (2026-10-02): det44 made pipe emission SEPLESS — the
        # odlbench pipe->HTML converter (converter_markdown_table.py)
        # REQUIRES a '| --- |' separator row, so sepless pipes pass
        # through as text and the html twin is the ONLY convertible
        # table representation left. det37's agreeing-grid twin
        # suppression predates that: it avoided a duplicate-table TEDS
        # penalty that sepless pipes can no longer cause (unconverted
        # pipes never enter the TEDS forest). Without a twin the whole
        # doc scores TEDS 0 — measured: the det41 1.0-class docs
        # (0053/0051/0084/0088/0150...) collapsed to 0.000 in the
        # det44+det41 merge until the twin was restored. Derive it from
        # the detection grid when the strategy chain never built one.
        _rows = t.get("rows") or []
        if not _rows and _md:
            # det47 (odlbench doc-0110): some emission paths build a
            # table dict with markdown ONLY (no rows, no html) — the
            # twin derivation below would silently skip and the whole
            # doc scores TEDS 0 (sepless pipes never convert). Parse the
            # rows back out of the pipe markdown as the fallback source.
            _rows = [
                [c.strip() for c in line.strip().strip("|").split("|")]
                for line in _md.splitlines()
                if line.strip().startswith("|")
            ]
        if len(_rows) >= 2:
            try:
                from scientific_parser.perception.tables.detector import build_table_html

                _derived = build_table_html(_rows).strip()
                if _derived:
                    _tw_html = _derived
            except Exception:
                pass  # twin is an enrichment; pipes remain the emission
    return [_md, _tw_html] if _tw_html else [_md]


def _detect_and_append_tables(
    text: str, pdf_path: str = "", page: int = 0, ocr_text: str = ""
) -> str:
    """Defensive wrapper for detect_all_tables — appends detected table markdown.
    Mirrors main-flow logic at line 8810+ (fast) and 9140+ (OCR).
    """
    if not pdf_path or not page:
        return text
    # EXP-1 table-evidence gate (2026-09-15): the 7-strategy chain costs
    # ~1.6s even on tableless pages. Skip ONLY when every evidence signal
    # is absent (text probes, native find_tables, x-recurrence, vector
    # grid) — the gate itself is fail-safe (probe failure → run chain).
    # OCR path is unaffected: it calls detect_all_tables directly with
    # ocr_text (wrapper never receives it).
    try:
        from scientific_parser.routing.layout_gate import page_needs_table_chain

        if not page_needs_table_chain(pdf_path, page, text):
            return text
    except Exception:
        pass  # gate unavailable → detection runs (fail-safe)
    # det48: when the page's ONLY table evidence is a table-ish raster
    # (raster_only_table_evidence), scope detection to raster strategies —
    # text/geometry strategies see no signal there and emit positional/
    # whitespace false positives on the prose around the image (odlbench
    # det47 0056/0066/0098/0152). Fail-open: evidence check failure →
    # full chain (matches page_needs_table_chain fail-safe posture).
    _raster_only = False
    try:
        from scientific_parser.routing.layout_gate import raster_only_table_evidence

        _raster_only = raster_only_table_evidence(pdf_path, page, text)
    except Exception:
        _raster_only = False
    try:
        # New document starts at page 1 — clear cross-page continuation
        # state so a previous PDF's open table can never merge into this
        # one (batch workers process many PDFs in one process; without
        # this the leak glues tables ACROSS documents when x-extent and
        # column count happen to match).
        if page == 1:
            global _OPEN_TABLES_STATE
            _OPEN_TABLES_STATE = []
        real_tables = detect_all_tables(
            pdf_path,
            page,
            ocr_text=ocr_text if ocr_text else None,
            raster_only=_raster_only,
        )
        real_tables = merge_cross_page_tables(page, real_tables)
        # det32 (2026-09-23): the same physical table via two strategies
        # with disjoint reported bboxes (positional text-extent vs camelot
        # lattice crop) slips the bbox-overlap dedup inside
        # detect_all_tables — both copies emit, and downstream evaluators
        # score whichever table lands FIRST in the document (measured
        # odlbench 01030000000053: the furniture-header positional copy
        # beat the correct camelot copy). Drop content duplicates.
        real_tables = _dedup_content_tables(real_tables)
        real_tables = _order_tables_first_best(real_tables)
        # det61 note: TableFormer zero-table fallback does NOT live here.
        # A pre-dispatch hook (sweep p61, 2026-10-12) let TF pipes flip the
        # region dispatcher's structure-gain gate and suppressed its better
        # table on 137297b4 pg4 (+6 P2F). The fallback is TERMINAL, page-
        # level, after _dispatch_regions — see pdf_ocr.py fast-path tail.
        if real_tables:
            # MONSTER-TABLE REBUILD (2026-08-18): pdfplumber/lines_strict
            # glue an entire label column into one giant cell on
            # borderless stats tables (measured 904a1b4e pg141: 15 labels
            # newline-joined in a single cell — up/down/left heading
            # lookups all fail). The TableAgent line-grid cascade rebuilds
            # correct per-row grids from char geometry; rebuild any
            # monster through it and keep the better markdown.
            for t in real_tables:
                if _is_monster_table_md(t.get("markdown", "")) or _is_monster_rows(
                    t.get("rows") or []
                ):
                    rebuilt = _rebuild_table_via_agent(pdf_path, page, t.get("bbox"))
                    if rebuilt:
                        t["markdown"] = rebuilt
                        t.pop("html", None)  # stale grid → stale html
                        t["rebuilt_from_monster"] = True
            text = text.rstrip() + "\n\n"
            for t in real_tables:
                # det37 (2026-09-24): gated dual emission — single pipe
                # block when md/twin grids agree (TEDS: extra table costs
                # its full subtree), pipes + RAW html twin when they
                # differ materially (TableTest probes satisfy from either
                # representation; see _emit_table_blocks).
                for _blk in _emit_table_blocks(t):
                    if _blk:
                        # det40 (2026-09-24): blank line between emitted
                        # blocks — consecutive single-\n pipe tables parse
                        # as ONE merged table (5c6a9a49: 'BRUTE Hydrazine
                        # Vessel' row landed under the previous table's
                        # 'Operating Conditions' header). det31's twin
                        # html blocks used to separate them.
                        text += _blk + "\n\n"
        return text
    except Exception as e:
        # Loud (was log.debug): a silent NameError here hid order-dependent
        # table loss for days (2026-09-15) — H1/H2, no swallowed failures.
        log.warning("table detector skipped p%s: %s: %s", page, type(e).__name__, e)
        return text


def _is_glued_or_mojibake(t):
    """A detected table whose reconstruction is untrustworthy: any
    broken-CMap cell ('(cid:') or a glued multi-word row spanning what
    should be separate columns (monster run signature, see
    _is_monster_table_md). Used by the whitespace-strategy gate."""
    for r in t.get("rows", []):
        for c in r:
            if "(cid:" in str(c):
                return True
    md_probe = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in t.get("rows", []))
    return _is_monster_table_md(md_probe)


def _is_monster_rows(rows):
    """Rows-level monster signature (det41, 2026-09-24): a label column
    glued into ONE cell carries the labels newline-joined INSIDE the cell
    (measured 904a1b4e pg141: rows[2][0] = 13 embedded newlines). det32's
    _fold_cell collapses those newlines to spaces in the emitted markdown
    BEFORE _is_monster_table_md ever sees it — the pipe-less-run signature
    goes blind and the TableAgent rebuild never fires (the 4-probe
    904a1b4e regression, det38-40). Legit wrapped cells ('IX - Zamboanga
    \\nPeninsula') carry 1-2 newlines; the mega-cell carries >=3.
    det41b narrowing: 033c1efd2 pg5 ALSO has >=3-newline cells (max 7)
    but every row carries its OWN label — the rebuild misaligned its
    grid (t05 'Sector'→'SPS/09' broke). Structural guard: fire only
    when the newline mega-cell has EATEN a column — >=30% of rows with
    an empty first cell (904a1b4e pg141 t0: 13-newline label mega-cell,
    84% empty col0)."""
    has_nl_cell = False
    for row in rows or ():
        for cell in row:
            if str(cell).count("\n") >= 3:
                has_nl_cell = True
                break
        if has_nl_cell:
            break
    if not has_nl_cell:
        return False
    empty_first = sum(1 for row in rows or () if not str(row[0]).strip() if row)
    return empty_first >= max(3, int(0.3 * len(rows)))


def _is_monster_table_md(markdown):
    """A table whose cell content is GLUED (an entire label column merged
    into one cell) — signature: a contiguous run of >=4 pipe-less content
    lines INSIDE the pipe table (the glued cell's embedded newlines break
    the row structure; measured 904a1b4e pg141: 14-line run). Runs of 1-3
    are wrapped multi-line header cells (bef02dc2 'Last Two v./First
    Two') — benign.

    NB: a cell-LENGTH ratio test (max/median) is NOT a safe second
    signature — legitimate descriptive labels run 174 chars against
    median-8 numeric cells (22x, measured bef02dc2 pg6) and false-fire
    the rebuild on a correct table. Row-gluing is the only reliable
    monster signature.
    """
    lines = (markdown or "").split("\n")
    pipe_idx = [i for i, l in enumerate(lines) if l.strip().startswith("|")]
    if len(pipe_idx) < 2:
        return False
    run = best = 0
    for i in range(pipe_idx[0], pipe_idx[-1] + 1):
        if lines[i].strip() and not lines[i].strip().startswith("|"):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best >= 4


def _rebuild_table_via_agent(pdf_path, page, bbox):
    """Rebuild one table region through the TableAgent cascade (line-grid
    first on born-digital pages). Returns markdown or None.

    ``bbox`` comes from detect_all_tables strategies in PDF POINTS; the
    agent works in render pixels (dpi/72) — convert before dispatch.
    """
    if not bbox:
        return None
    try:
        from scientific_parser.perception.tables.table_agent import TableAgent

        scale = 150.0 / 72.0
        bbox_px = tuple(v * scale for v in bbox)
        agent = TableAgent(pdf_path, page, 150.0)
        result = agent.process({"bbox": bbox_px, "native": None})
        if result is None or not result.markdown:
            return None
        if _is_monster_table_md(result.markdown):
            return None  # rebuild no better — keep the original
        return result.markdown
    except Exception as e:
        log.debug("monster rebuild via agent skipped: %s", e)
        return None


__all__ = [
    "restore_degree_symbol",
    "correct_ocr_errors",
    "fix_text_encoding",
    "drop_repeated_text",
    "remove_bold_duplicate_chars",
    "merge_cross_page_tables",
    "detect_all_tables",
    "repair_encoding_decimals",
    "preserve_typography",
    "compact_spaced_caps_headings",
    "repair_fused_words",
    "normalize_chemical_notation",
    "repair_citation_splits",
    "preserve_dois",
    "detect_bibliography_section",
    "process_bibliography",
    "detect_section_hierarchy",
    "reconstruct_footnotes",
    "associate_captions",
    "repair_bullets",
    "UNICODE_MATH_CHARS",
    "restore_scientific_subscripts",
    "restore_unicode_losses",
    "promote_allcaps_headings",
    "sanitize_phantom_decimals",
    "fix_permille",
    "validate_scientific_symbols",
    "validate_chemical_formulas",
    "markdown_compatible",
    "strip_running_stamps",
    "strip_cjk_hallucination",
    "fix_latex_math_output",
    "PostProcessor",
    "_build_registry",
    "_get_registry",
    "run_unified_postproc",
    "join_fragmented_math_lines",
    "_collapse_repetition_inline",
    "_extract_text_with_scripts_safe",
    "_detect_and_append_tables",
]
