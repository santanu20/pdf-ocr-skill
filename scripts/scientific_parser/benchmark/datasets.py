"""Benchmark datasets — load olmOCR-Bench and other standard datasets.

Per proposal item #38: SciDocBench is aspirational. We integrate with
olmOCR-Bench (Allen AI) for now.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BenchDataset:
    """One benchmark dataset."""

    name: str
    root: Path
    categories: list
    format: str  # "olmocr" | "omnidocbench" | "pubtabnet" | ...


def load_olmocr_bench(root: Path) -> BenchDataset:
    """Load an olmOCR-Bench dataset directory.

    Expected structure:
      root/
        pdfs/<category>/*.pdf
        <category>.jsonl
        baseline.jsonl
    """
    root = Path(root)
    cats = []
    if (root / "pdfs").exists():
        cats = sorted(d.name for d in (root / "pdfs").iterdir() if d.is_dir())
    return BenchDataset(
        name="olmocr-bench",
        root=root,
        categories=cats,
        format="olmocr",
    )


def list_tests(dataset: BenchDataset, category: str) -> list:
    """List all test cases for a category."""
    import json

    jsonl = dataset.root / f"{category}.jsonl"
    if not jsonl.exists():
        return []
    return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]


def list_pdfs(dataset: BenchDataset, category: str) -> list:
    """List all PDF files for a category."""
    pdf_dir = dataset.root / "pdfs" / category
    if not pdf_dir.exists():
        return []
    return sorted(pdf_dir.glob("*.pdf"))


__all__ = ["BenchDataset", "load_olmocr_bench", "list_tests", "list_pdfs"]
