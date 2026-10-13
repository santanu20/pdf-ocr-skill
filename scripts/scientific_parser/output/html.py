"""HTML output adapter — emit DocumentIR as semantic HTML with provenance."""

from __future__ import annotations

import html


def document_to_html(doc, title: str = "Document") -> str:
    """Render DocumentIR to standalone HTML page with semantic markup."""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">',
        "<style>",
        "body{font-family:Georgia,serif;max-width:780px;margin:2em auto;padding:0 1em;color:#222}",
        ".page-break{page-break-after:always;margin:2em 0;border-top:1px solid #ccc}",
        ".block{margin:0 0 1em 0}",
        ".title{font-size:1.6em;font-weight:bold}",
        ".subtitle{font-size:1.2em;font-weight:bold}",
        ".equation{margin:1em 0;text-align:center}",
        ".caption{font-style:italic;color:#555;font-size:0.9em}",
        ".provenance{font-size:0.7em;color:#888;background:#f8f8f8;padding:0.2em 0.4em;border-radius:2px}",
        "</style>",
        "</head>",
        "<body>",
    ]

    for page in doc.pages:
        parts.append(f'<div class="page-break" data-page="{page.page_num}">')
        for block_id in page.reading_order:
            block = next((b for b in page.blocks if b.id == block_id), None)
            if not block:
                continue
            winner = block.resolve()
            value = winner.value if winner else None
            if value is None:
                continue
            text = html.escape(str(value))
            btype = block.type
            conf = winner.confidence if winner else 0.0
            if btype == "title":
                parts.append(f'<h1 class="block title">{text}</h1>')
            elif btype == "subtitle":
                parts.append(f'<h2 class="block subtitle">{text}</h2>')
            elif btype == "equation":
                parts.append(f'<div class="block equation">{text}</div>')
            elif btype == "table":
                parts.append(f'<div class="block table">{text}</div>')
            elif btype == "figure":
                parts.append(f'<figure class="block"><figcaption>{text}</figcaption></figure>')
            elif btype == "caption":
                parts.append(f'<p class="block caption">{text}</p>')
            else:
                parts.append(f'<p class="block">{text}</p>')
            if conf < 0.7:
                parts.append(f'<span class="provenance">conf={conf:.2f}</span>')
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def write_html(doc, path, title: str = "Document") -> None:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document_to_html(doc, title=title), encoding="utf-8")


__all__ = ["document_to_html", "write_html"]
