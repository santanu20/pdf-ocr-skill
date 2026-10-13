"""Scientific validators — REE sequence, mineral abbreviations, data sanity.

Extracted from scripts/pdf_ocr.py during Phase R8 refactoring (2026-08-12).

All validators are SELF-GATING: they check for their own content signature
before firing. No domain hints needed — universal parser, no field bias.
"""

from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state


def _pdf_ocr():
    import sys

    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


_LATE = frozenset(
    {
        "REE_SEQUENCE",
        "REE_OCR_CONFUSIONS",
        "_has_monotonic_re_run",
        "OXIDE_MW",
    }
)


_CONSTS_LOADED = False


def _load_consts():
    """Lazy-load cross-module constants on first call."""
    global _CONSTS_LOADED
    if _CONSTS_LOADED:
        return
    p = _pdf_ocr()
    g = globals()
    for name in (
        "REE_SEQUENCE",
        "REE_OCR_CONFUSIONS",
        "_has_monotonic_re_run",
        "OXIDE_MW",
    ):
        if hasattr(p, name):
            g[name] = getattr(p, name)
    _CONSTS_LOADED = True


def __getattr__(name):
    if name in _LATE:
        return getattr(_pdf_ocr(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


MINERAL_ABBREVS = {
    "Ab": "albite",
    "Act": "actinolite",
    "Bt": "biotite",
    "Cal": "calcite",
    "Chl": "chlorite",
    "Cpx": "clinopyroxene",
    "Grt": "garnet",
    "Hbl": "hornblende",
    "Ilm": "ilmenite",
    "Kfs": "K-feldspar",
    "Ky": "kyanite",
    "Mag": "magnetite",
    "Ms": "muscovite",
    "Ol": "olivine",
    "Opx": "orthopyroxene",
    "Pl": "plagioclase",
    "Qtz": "quartz",
    "Rt": "rutile",
    "Sil": "sillimanite",
    "Spl": "spinel",
    "St": "staurolite",
    "Ttn": "titanite",
    "Tur": "tourmaline",
    "Zrn": "zircon",
    "Amp": "amphibole",
    "Px": "pyroxene",
    "Ep": "epidote",
    "Ser": "serpentine",
    "Tlc": "talc",
    "Tr": "tremolite",
    "Ap": "apatite",
    "Sp": "spinel",
}


def detect_mineral_abbreviations(text):
    """Detect Kretz/IUGS mineral abbreviations in text. Returns warning list.

    SELF-GATING (no domain hint needed): fires ONLY when geological context is
    present (rock/mineral/petrology terms nearby). Prevents false positives on
    math/physics/bio papers where "Bt." could be a section label or "Pl." a
    variable. Universal processor — runs for any field safely.
    """
    _load_consts()  # lazy-load cross-module constants
    # Self-gate: require geological context anywhere in the document text.
    # Without these terms, mineral abbreviations are not petrographic.
    if not re.search(
        r"\b(?:rock|granite|basalt|gabbro|peridotite|rhyolite|andesite|diorite|"
        r"sediment|metamorph|igneous|volcanic|plutonic|craton|thin.?section|"
        r"petrograph|litholog|mineral(?:ogic|ogy)?|ore|crystall|epidosite|"
        r"facies|fabric|tectonite|mylonite|gneiss|schist|amphibolite)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return []
    warnings = []
    # Check for corrupted mineral abbreviations
    for abbrev, full_name in MINERAL_ABBREVS.items():
        # Trailing dot: Bt. → Bt
        dotted = abbrev + "."
        if dotted in text:
            warnings.append(f"mineral_abbrev_dot: '{dotted}' should be '{abbrev}' ({full_name})")
    return warnings


# =============================================================================
# P2-8: Reference-aware parsing
# =============================================================================

# DOI pattern: 10.NNNN/identifier


def validate_scientific_data(text):
    """Domain-specific geochemistry validation. Returns warning list.

    Checks: oxide totals (~100 +/- 5), U-Pb age range.
    Oxide total: does NOT sum Fe2O3 + FeO together (separate iron species).
    """
    _load_consts()  # lazy-load cross-module constants
    warnings = []

    # 1. Oxide total check — per first occurrence only (first sample/row)
    # Do NOT sum Fe2O3 and FeO together — they're separate iron species
    oxide_values = {}
    for oxide in OXIDE_MW:
        pattern = re.escape(oxide).replace("2", "[\u20822]").replace("3", "[\u20833]")
        match = re.search(pattern + r"\s+(\d+\.?\d*)", text)
        if match:
            val = float(match.group(1))
            if 0.01 < val < 120:
                oxide_values[oxide] = val

    if len(oxide_values) >= 6:
        # If both Fe2O3 and FeO present, use only FeO*1.1113 (convert to Fe2O3 total)
        # or use only whichever was reported as "total iron"
        if "Fe2O3" in oxide_values and "FeO" in oxide_values:
            # Don't double-count: use Fe2O3 only (drop FeO from total)
            total_values = {k: v for k, v in oxide_values.items() if k != "FeO"}
        else:
            total_values = oxide_values

        total = sum(total_values.values())
        if total < 95 or total > 105:
            warnings.append(
                f"oxide_total_anomalous: sum={total:.1f}% "
                f"(expected ~100+/-5). Values: {total_values}"
            )

    # 2. U-Pb age validation
    age_matches = re.findall(
        r"(\d+\.?\d*)\s*[\u00b1]\s*(\d+\.?\d*)\s*(Ma|Ga|ka)\b",
        text,
    )
    for age_str, err_str, unit in age_matches:
        age = float(age_str)
        if unit == "Ga" and age > 4.6:
            warnings.append(f"age_implausible: {age} {unit} exceeds Earth age")
        elif unit == "Ma" and age > 4600:
            warnings.append(f"age_implausible: {age} {unit} > 4600 Ma")
        elif age < 0:
            warnings.append(f"age_negative: {age} {unit}")

    return warnings


# =============================================================================


def validate_ree_sequence(text, window_chars: int = 200, min_cluster_size: int = 3):
    """Detect REE SEQUENCE anomalies (not body-text mentions). Returns warning list.

    Fires ONLY on clusters that contain a monotonic run of ≥4 REE symbols —
    the signature of an actual REE listing (axis label, plot row, table
    header, separator-delimited enumeration). Scattered body-prose mentions
    of ratios (Eu/Eu*, La/Yb, Sm/Nd) and individual anomalies (Ce anomaly,
    Eu anomaly) appear in non-monotonic order and are intentionally ignored.

    Parameters
    ----------
    text : str
        Page text (tables stripped before REE scan).
    window_chars : int
        Maximum gap (in characters) between two consecutive REE mentions for
        them to be considered part of the same cluster. 200 ≈ 10 typical
        REE labels — accommodates standard REE plot axes without merging
        unrelated prose mentions.
    min_cluster_size : int
        Minimum REE count within a cluster to be considered for validation.
        Final gating is via ``_has_monotonic_re_run`` (≥4 monotonic).
    """
    _load_consts()  # lazy-load cross-module constants
    # Split into prose and table sections
    # Table sections are between |---|...| patterns (multi-sample data tables
    # legitimately contain La/Ce/Eu once per row).
    prose_parts = re.split(r"\n\|[-:| ]+\|\n.*?(?=\n\n|\Z)", text, flags=re.DOTALL)
    prose_text = "\n".join(prose_parts)

    found = []
    for m in re.finditer(r"\b([A-Z][a-z]{0,2})\b", prose_text):
        t = m.group(1)
        if t in REE_SEQUENCE or t in REE_OCR_CONFUSIONS:
            found.append((m.start(), t))

    if len(found) < min_cluster_size:
        return []

    # Group REE mentions into clusters where consecutive mentions are within
    # window_chars of each other. Only clusters that contain a monotonic REE
    # run (signature of a real listing) are validated.
    warnings = []
    i = 0
    n = len(found)
    while i < n:
        # Extend cluster forward while next REE is within window_chars
        j = i
        while j + 1 < n and found[j + 1][0] - found[j][0] <= window_chars:
            j += 1
        cluster = found[i : j + 1]
        if len(cluster) >= min_cluster_size and _has_monotonic_re_run(cluster):
            seen: list[str] = []
            expected = list(REE_SEQUENCE)
            for pos, token in cluster:
                resolved = REE_OCR_CONFUSIONS.get(token, token)
                if token in REE_OCR_CONFUSIONS:
                    warnings.append(
                        f"ree_ocr_confusion: '{token}' at pos {pos} likely "
                        f"'{REE_OCR_CONFUSIONS[token]}'"
                    )
                if resolved not in REE_SEQUENCE:
                    continue
                if resolved in seen:
                    warnings.append(f"ree_duplicate: '{resolved}' at pos {pos}")
                    continue
                if seen:
                    last = REE_SEQUENCE.index(seen[-1])
                    curr = REE_SEQUENCE.index(resolved)
                    if curr < last:
                        warnings.append(
                            f"ree_reversed: '{resolved}' at pos {pos} after '{seen[-1]}'"
                        )
                seen.append(resolved)
                if resolved in expected:
                    expected.remove(resolved)
            # Only flag missing-REE if cluster covers >=8 distinct REE
            # (a real near-complete sequence listing).
            if len(seen) >= 8:
                for miss in expected:
                    if miss != "Pm":
                        warnings.append(f"ree_missing: '{miss}' not in sequence")
        # Advance past this cluster
        i = j + 1 if j > i else i + 1

    return warnings


# Fix 5: Scientific symbol validator

__all__ = [
    "MINERAL_ABBREVS",
    "def detect_mineral_abbreviations",
    "def validate_scientific_data",
    "def validate_ree_sequence",
]
