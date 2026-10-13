"""Adaptive compute budget allocator (proposal item #23).

Per architecture proposal:
  "Build an adaptive compute budget. For each page:
      compute_budget = f(uncertainty, complexity, region_types, prev_conf)

   Easy page → native PDF text → no OCR
   Medium → native + cheap OCR
   Hard equation → crop → high-res math OCR → verifier
   Difficult table → table detector → high-res crop → table parser → schema validator
   Ambiguous result → second model"

Implementation:
  - Budget class: time-budget + model-tier per page
  - Allocator: takes PageComplexity (from coarse_to_fine) + uncertainty
  - Tier-aware model selection (cheap → expensive)
  - Bounded retry budget (e.g. max 3 retries per region)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import sys

if "scripts" not in sys.path:
    sys.path.insert(0, "scripts")


@dataclass
class ComputeBudget:
    """Compute budget allocation for one page."""

    time_budget_s: float = 30.0  # max wall-clock for this page
    model_tier: str = "cheap"  # cheap|medium|expensive
    ocr_max_retries: int = 1  # max retries on disagreement
    enable_verifier: bool = False  # run pykatex/validators
    enable_region_crops: bool = False  # high-res only on specific regions
    enable_second_model: bool = False  # run backup model on disagreement
    enable_table_schema: bool = False
    enable_equation_ast: bool = False
    enable_vector_extraction: bool = False
    notes: list = field(default_factory=list)

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)


def allocate_budget(
    tier: str,
    uncertainty: float = 0.0,
    is_math_heavy: bool = False,
    is_table_heavy: bool = False,
    is_multicol: bool = False,
    prior_confidence: float = 1.0,
) -> ComputeBudget:
    """Allocate compute budget based on tier + signals.

    Args:
        tier: from coarse_to_fine.assess_page_complexity().tier
        uncertainty: 0.0 (certain) to 1.0 (very uncertain)
        is_math_heavy: page has math content
        is_table_heavy: page has tables
        is_multicol: multi-column layout
        prior_confidence: confidence from earlier pass (if re-evaluation)

    Returns:
        ComputeBudget instance.
    """
    budget = ComputeBudget()

    if tier == "trivial":
        budget.time_budget_s = 1.0
        budget.model_tier = "none"  # use PDF text directly
        budget.add_note("trivial: PDF text layer only, no OCR")

    elif tier == "simple":
        budget.time_budget_s = 5.0
        budget.model_tier = "cheap"
        budget.enable_verifier = True
        budget.add_note("simple: cheap OCR + basic validators")

    elif tier == "medium":
        budget.time_budget_s = 30.0
        budget.model_tier = "medium"
        budget.enable_verifier = True
        budget.enable_region_crops = True
        if is_multicol:
            budget.enable_vector_extraction = True
            budget.add_note(
                "medium + multicol: enable vector extraction for column detection"
            )
        if is_table_heavy:
            budget.enable_table_schema = True
            budget.add_note("medium + tables: enable schema inference")
        if is_math_heavy:
            budget.enable_equation_ast = True
            budget.add_note("medium + math: enable equation AST parsing")

    elif tier == "hard":
        budget.time_budget_s = 120.0
        budget.model_tier = "expensive"
        budget.enable_verifier = True
        budget.enable_region_crops = True
        budget.enable_second_model = uncertainty > 0.5
        budget.ocr_max_retries = 3 if uncertainty > 0.5 else 1
        if is_math_heavy:
            budget.enable_equation_ast = True
        if is_table_heavy:
            budget.enable_table_schema = True
            budget.enable_vector_extraction = True
        budget.add_note(
            f"hard: expensive tier, retries={budget.ocr_max_retries}, "
            f"second_model={budget.enable_second_model}"
        )

    # Adjust for uncertainty (proposal item #24)
    if uncertainty > 0.5 and not budget.enable_second_model:
        budget.enable_second_model = True
        budget.ocr_max_retries = max(budget.ocr_max_retries, 2)
        budget.add_note(f"high uncertainty ({uncertainty:.2f}): enabling 2nd model")

    # Adjust for prior confidence (re-evaluation case)
    if prior_confidence < 0.5:
        budget.time_budget_s *= 1.5  # give more time
        budget.add_note(
            f"low prior confidence ({prior_confidence:.2f}): +50% time budget"
        )

    return budget


def summarize_budget(budget: ComputeBudget) -> str:
    """One-line human-readable summary."""
    flags = []
    if budget.enable_verifier:
        flags.append("verify")
    if budget.enable_region_crops:
        flags.append("crops")
    if budget.enable_second_model:
        flags.append("2nd-model")
    if budget.enable_table_schema:
        flags.append("table-schema")
    if budget.enable_equation_ast:
        flags.append("eq-AST")
    if budget.enable_vector_extraction:
        flags.append("vectors")
    return (
        f"tier={budget.model_tier} time={budget.time_budget_s:.0f}s "
        f"retries={budget.ocr_max_retries} [{','.join(flags) if flags else 'none'}]"
    )


__all__ = ["ComputeBudget", "allocate_budget", "summarize_budget"]
