"""Document hierarchy — section / subsection detection.

Per proposal L2 (Structure): paragraph, table, equation, figure, caption,
reading order. Hierarchy detects section levels (1, 1.1, 1.1.1).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SectionNode:
    """One section in the document hierarchy."""

    level: int  # 1 = top-level, 2 = subsection, ...
    title: str
    number: Optional[str] = None  # "1", "1.1", "2.3.1"
    page: Optional[int] = None
    bbox: Optional[tuple] = None
    children: list = field(default_factory=list)  # nested SectionNode


# Heading patterns (numbered)
_HEADING_PATTERNS = [
    # "1. Introduction" or "1 Introduction"
    re.compile(r"^(?P<num>\d+)\.?\s+(?P<title>[A-Z][^\n]{2,80})$", re.MULTILINE),
    # "1.1 Subsection" or "1.1. Subsection"
    re.compile(
        r"^(?P<num>\d+\.\d+(?:\.\d+)*)\.?\s+(?P<title>[A-Z][^\n]{2,80})$", re.MULTILINE
    ),
    # "Section 1: Title"
    re.compile(
        r"^Section\s+(?P<num>\d+)\s*:\s*(?P<title>[^\n]+)$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # ALL CAPS heading (no number)
    re.compile(r"^(?P<title>[A-Z][A-Z\s]{4,60})$", re.MULTILINE),
]


def detect_sections(text: str, page_num: Optional[int] = None) -> list[SectionNode]:
    """Detect section headings in text. Returns flat list (caller builds tree)."""
    sections = []
    for pat in _HEADING_PATTERNS:
        for m in pat.finditer(text):
            num = m.groupdict().get("num")
            title = m.groupdict().get("title", "").strip()
            level = (num.count(".") + 1) if num else 1
            sections.append(
                SectionNode(
                    level=level,
                    title=title,
                    number=num,
                    page=page_num,
                )
            )
    # Dedupe by (level, title)
    seen = set()
    out = []
    for s in sections:
        key = (s.level, s.title.lower())
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def build_tree(sections: list[SectionNode]) -> list[SectionNode]:
    """Build nested tree from flat list. Top-level sections returned."""
    if not sections:
        return []
    # Sort by appearance order (assumed)
    roots = []
    stack = []  # current nesting
    for s in sections:
        # Pop stack until we find a parent with smaller level
        while stack and stack[-1].level >= s.level:
            stack.pop()
        if stack:
            stack[-1].children.append(s)
        else:
            roots.append(s)
        stack.append(s)
    return roots


__all__ = ["SectionNode", "detect_sections", "build_tree"]
