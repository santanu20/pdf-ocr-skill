"""Markdown output adapter — emit DocumentIR as readable markdown."""

from __future__ import annotations


def document_to_markdown(doc) -> str:
    """Render DocumentIR to markdown.

    Layout: page separators, block-level typing (text/equation/table/figure),
    preserves reading order.
    """
    out_lines = []
    for page in doc.pages:
        out_lines.append("\n")
        for block_id in page.reading_order:
            block = next((b for b in page.blocks if b.id == block_id), None)
            if not block:
                continue
            winner = block.resolve()
            value = winner.value if winner else None
            if value is None:
                continue
            if block.type == "title":
                out_lines.append(f"\n## {value}\n")
            elif block.type == "subtitle":
                out_lines.append(f"\n### {value}\n")
            elif block.type == "equation":
                # Already in $...$ or $$...$$ format
                if not (str(value).startswith("$")):
                    value = f"$${value}$$"
                out_lines.append(f"\n{value}\n")
            elif block.type == "table":
                out_lines.append(f"\n{value}\n")
            elif block.type == "figure":
                out_lines.append(f"\n<!-- FIGURE: {value} -->\n")
            elif block.type == "caption":
                out_lines.append(f"\n*{value}*\n")
            else:  # text
                out_lines.append(str(value))
        out_lines.append("")
    return "\n".join(out_lines)


def write_markdown(doc, path) -> None:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document_to_markdown(doc), encoding="utf-8")


__all__ = ["document_to_markdown", "write_markdown"]
