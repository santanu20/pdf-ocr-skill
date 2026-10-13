"""Evidence operations — multi-level evidence chain + weighted resolver.

Companion to core/ir.py. Provides:
  - merge_evidence: combine evidence from multiple sources
  - rank_by_priority: order by E-level priority (utility, used by audit)
  - EvidenceResolver: pick winning Evidence by weighted score
  - DEFAULT_RESOLVER: module-level singleton (priority-fallback behavior)

Resolver weights (per architecture proposal item #1, 2026-08-13):
    weighted_score = SOURCE_WEIGHT * source_reliability[level]
                   + CONF_WEIGHT   * confidence
                   + VERIF_WEIGHT  * (verification.score if verification else NEUTRAL)

Without calibration data these are heuristic defaults. Do NOT tune blindly.
"""

from __future__ import annotations

from .ir import EVIDENCE_PRIORITY, Evidence, VerificationResult

# === Source reliability by E-level ================================== #
# Higher = trusted more (when verification signal is absent). Reflects the
# observation that native PDF text is more reliable than inferred semantic
# values. Values in [0, 1].
EVIDENCE_SOURCE_RELIABILITY: dict[str, float] = {
    "E0_native_pdf_text": 1.00,
    "E1_native_pdf_geometry": 0.95,
    "E2_deterministic_reconstruction": 0.90,
    "E3_classical_ocr": 0.70,
    "E4_specialist_ocr": 0.85,
    "E5_vlm": 0.60,
    "E6_semantic": 0.50,
}

NEUTRAL_VERIFICATION_SCORE = 0.5  # used when Evidence.verification is None
DEFAULT_SOURCE_WEIGHT = 0.40
DEFAULT_CONF_WEIGHT = 0.30
DEFAULT_VERIF_WEIGHT = 0.30


class EvidenceResolver:
    """Pick the winning Evidence from a chain using weighted scoring.

    The score combines three signals:
      * source_reliability: prior trust in the E-level (E0 native text > E5 VLM)
      * confidence: model/emitter-reported confidence
      * verification: typed-validator score (None = neutral)

    Tie-break: lowest E-number wins (E0 < E1 < ... < E6). This makes the
    resolver's output deterministic when scores are equal.

    Per architecture proposal item #1 (2026-08-13): replaces the fixed
    ``min(E-level)`` priority lookup with evidence-weighted selection.
    """

    def __init__(
        self,
        source_weight: float = DEFAULT_SOURCE_WEIGHT,
        conf_weight: float = DEFAULT_CONF_WEIGHT,
        verif_weight: float = DEFAULT_VERIF_WEIGHT,
        source_reliability: dict[str, float] | None = None,
        neutral_verification: float = NEUTRAL_VERIFICATION_SCORE,
    ) -> None:
        """Initialize resolver.

        Args:
            source_weight: weight for source_reliability term. Default 0.40.
            conf_weight: weight for confidence term. Default 0.30.
            verif_weight: weight for verification term. Default 0.30.
            source_reliability: optional override for the per-level reliability
                table. If None, uses :data:`EVIDENCE_SOURCE_RELIABILITY`.
            neutral_verification: score used when Evidence.verification is None.
                Default 0.5 (neither pass nor fail).

        Raises:
            ValueError: if weights are negative or don't sum > 0.
        """
        if min(source_weight, conf_weight, verif_weight) < 0:
            raise ValueError(
                f"weights must be non-negative, got "
                f"source={source_weight} conf={conf_weight} verif={verif_weight}"
            )
        total = source_weight + conf_weight + verif_weight
        if total <= 0:
            raise ValueError(f"weights must sum > 0, got {total}")
        self.source_weight = source_weight
        self.conf_weight = conf_weight
        self.verif_weight = verif_weight
        self.source_reliability = (
            source_reliability
            if source_reliability is not None
            else dict(EVIDENCE_SOURCE_RELIABILITY)
        )
        self.neutral_verification = neutral_verification
        # Normalize weights so the final score stays in [0, 1] regardless of
        # the user-supplied magnitudes.
        self._w_source = source_weight / total
        self._w_conf = conf_weight / total
        self._w_verif = verif_weight / total
        self._level_priority = {lvl: i for i, lvl in enumerate(EVIDENCE_PRIORITY)}

    def score(self, ev: Evidence) -> float:
        """Compute weighted score for one Evidence. Range [0, 1]."""
        rel = self.source_reliability.get(ev.level, 0.5)
        conf = max(0.0, min(1.0, float(ev.confidence)))
        if ev.verification is not None:
            verif = float(ev.verification.score)
        else:
            verif = self.neutral_verification
        return self._w_source * rel + self._w_conf * conf + self._w_verif * verif

    def resolve(self, evidence_list: list[Evidence]) -> Evidence | None:
        """Pick the Evidence with the highest weighted score.

        Tie-break: lowest E-level number wins (E0 native beats E5 VLM at
        equal score). Returns None for empty input.
        """
        if not evidence_list:
            return None

        # max() with tuple key (score, -priority_index) — Python's max picks
        # the FIRST max in iteration order on ties, so we explicitly break
        # ties by E-level priority (lower index = higher priority = wins).
        def sort_key(ev: Evidence) -> tuple[float, int]:
            return (self.score(ev), -self._level_priority.get(ev.level, len(EVIDENCE_PRIORITY)))

        return max(evidence_list, key=sort_key)


# Default resolver instance — used by BlockIR.resolve() when caller passes
# None. Mirrors the legacy priority-fallback behavior via the weighting:
# source_reliability dominates (E0=1.0 > E1=0.95 > ... > E6=0.50), and at
# equal reliability (same E-level) confidence + verification break the tie.
DEFAULT_RESOLVER = EvidenceResolver()


# === Utility helpers (kept) ========================================= #


def rank_by_priority(evidence_list: list[Evidence]) -> list[Evidence]:
    """Sort evidence by E-level priority (lowest E-number first).

    Utility used by audits/debug tooling. NOT used for resolution —
    resolution goes through :class:`EvidenceResolver`.
    """
    priority = {lvl: i for i, lvl in enumerate(EVIDENCE_PRIORITY)}
    return sorted(evidence_list, key=lambda e: priority.get(e.level, len(EVIDENCE_PRIORITY)))


def merge_evidence(existing: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    """Merge two evidence lists. Dedupes by (level, source)."""
    seen = {(e.level, e.source) for e in existing}
    out = list(existing)
    for e in new:
        key = (e.level, e.source)
        if key not in seen:
            out.append(e)
            seen.add(key)
    return out


__all__ = [
    "EvidenceResolver",
    "DEFAULT_RESOLVER",
    "EVIDENCE_SOURCE_RELIABILITY",
    "NEUTRAL_VERIFICATION_SCORE",
    "DEFAULT_SOURCE_WEIGHT",
    "DEFAULT_CONF_WEIGHT",
    "DEFAULT_VERIF_WEIGHT",
    "rank_by_priority",
    "merge_evidence",
    "Evidence",
    "EVIDENCE_PRIORITY",
    "VerificationResult",
]
