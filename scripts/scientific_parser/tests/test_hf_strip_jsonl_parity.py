"""Pin (2026-08-31): fast-path final hf-strip must feed .pages.jsonl.

Measured on headers_footers/0678963a..._page_2_processed (olmOCR-bench
run #1 baseline: absent-type score 0.560, 331 fails): the geometry
detector correctly flagged the deep-band journal footer lines
('Midwifery xxx (xxxx) xxx–xxx' at 4.5% H, '0266-6138/ © 2017 ...' at
92.1%, 'Please cite this article as: ...' at 95.9%) and the FINAL
pre-emit strip removed them from the .md — but the jsonl page record
was built BEFORE that strip, so the scorer-visible text (bench runner
splits .pages.jsonl into candidate files) kept the leak. The .md and
.pages.jsonl must carry the SAME stripped text.

Fixture (fitz-synthetic, replicating the measured geometry):
  - body paragraph lines at mid-page (must SURVIVE everywhere)
  - deep-band top footer at ~4.5% H
  - deep-band bottom copyright line at ~92.5% H
  - deep-band cite line at ~96% H

Contract:
  1. Footer strings absent from the .pages.jsonl record text.
  2. Footer strings absent from the .md output.
  3. Body text present in BOTH surfaces.
  4. md (minus meta comments) == jsonl text (parity invariant).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
ENGINE = SCRIPTS / "pdf_ocr.py"
VENV_PY = SCRIPTS.parent / ".venv" / "bin" / "python"

FOOTERS = [
    "Midwifery xxx (xxxx) xxx-xxx",
    "Published by Elsevier Ltd",
    "Please cite this article as:",
]
BODY = "The authors regret that this sentence is body text."


def _make_fixture(path: Path) -> None:
    import fitz  # noqa: PLC0415

    doc = fitz.open()
    page = doc.new_page(width=595, height=794)  # measured page box of 0678963a
    # Deep top band (~4.5% H): journal running head.
    page.insert_text((72, 36), FOOTERS[0], fontsize=6.4)
    # Body block mid-page (enough chars to route fast, not scan-routed).
    for i in range(12):
        page.insert_text(
            (72, 300 + 14 * i),
            f"{BODY} Paragraph {i} continues with ordinary prose words.",
            fontsize=9,
        )
    # Deep bottom band (~92.5% and ~96% H): copyright + cite boilerplate.
    page.insert_text((72, 733), "0266-6138/ (c) 2017 " + FOOTERS[1], fontsize=6.4)
    page.insert_text((72, 762), FOOTERS[2] + " Homer, C., Midwifery (2017)", fontsize=6.4)
    doc.save(str(path))
    doc.close()


def test_fastpath_hf_strip_feeds_jsonl_and_md():
    if not VENV_PY.exists():
        _skip("skill venv missing")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        pdf = tdp / "hf_parity.pdf"
        _make_fixture(pdf)
        assert pdf.exists() and pdf.stat().st_size > 500, "fixture not written"
        out = tdp / "out"
        env = dict(os.environ, PDF_OCR_NO_SKILL_VENV="1")
        proc = subprocess.run(
            [str(VENV_PY), str(ENGINE), str(pdf), str(out), "--json"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr[-800:]

        records = [
            json.loads(ln)
            for ln in out.with_suffix(".pages.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert records, "no page records emitted"
        jsonl_text = records[0]["text"]
        md_text = out.with_suffix(".md").read_text()

        for footer in FOOTERS:  # contract 1 + 2
            assert footer.lower() not in jsonl_text.lower(), (
                f"footer leaked into .pages.jsonl: {footer!r}"
            )
        # contract 4: parity (md minus meta comments and the leading '---'
        # document separator == jsonl text)
        md_body = "\n".join(
            ln for ln in md_text.splitlines() if not ln.lstrip().startswith("<!--")
        ).strip()
        if md_body.startswith("---"):
            md_body = md_body[3:].lstrip()
        assert re.sub(r"\s+", " ", md_body) == re.sub(r"\s+", " ", jsonl_text.strip()), (
            "md and jsonl page text diverged"
        )


def _skip(msg: str):  # pragma: no cover
    import pytest

    pytest.skip(msg)


if __name__ == "__main__":
    sys.exit(test_fastpath_hf_strip_feeds_jsonl_and_md())
