# pdf-ocr

[![Verify](https://github.com/santanu20/pdf-ocr-skill/actions/workflows/verify.yml/badge.svg)](https://github.com/santanu20/pdf-ocr-skill/actions/workflows/verify.yml)

Research-grade **scientific PDF → Markdown** extraction. CPU-only, ONNX-first,
local: text-layer geometry is trusted when good, vision specialists are only
invoked where the text layer fails. No data leaves your machine.

Built for scientific documents: equations (Unicode → LaTeX), tables
(geometry-first reconstruction), multi-column reading order, subscripts/
superscripts (SiO₂), footnotes, captions, and domain validators that flag
suspicious values (oxide totals, REE sequences, implausible ages) instead of
silently shipping them.

## How it works

```
Input PDF
    ↓
preclassify_document()          ← born-digital vs scanned (one pass)
    ↓
Per page:
    ├─ Born-digital, good text → FAST path (pdftotext, milliseconds/page)
    │    column-aware order · table reconstruction · footnote relocation
    │    typography (italic/bold/mono/super/subscript) · caption matching
    ├─ Scanned / garbled → OCR path (seconds/page)
    │    adaptive DPI (150/250/300) · RapidOCR (PP-OCRv6)
    │    optional Ollama vision-model cascade / auto-fallback
    └─ Both paths → 12 scientific post-processors + validator warnings
```

Specialists run where they earn their cost: equations go native
Unicode → LaTeX first (most math needs no model), falling back to RapidLaTeX;
tables reconstruct from native geometry/pdfplumber first, falling back to
RapidTable. Layout detection uses DocLayout-YOLO (ONNX). Every unintended
routing decision emits a `PATH WARNING` to stderr and into the JSON summary —
nothing degrades silently.

## Install

Requirements: Python ≥ 3.11, [uv](https://docs.astral.sh/uv/),
[poppler-utils](https://poppler.freedesktop.org/) (`pdftotext`/`pdffonts`).
Optional: [Ollama](https://ollama.com) for vision-model escalation on
degraded scans.

```bash
git clone https://github.com/santanu20/pdf-ocr-skill.git
cd pdf-ocr-skill
python3 scripts/pdf_ocr.py input.pdf output.md
```

That's it — the first run self-bootstraps a project venv via
`uv sync` (specialist weights auto-download from Hugging Face; one-time cost)
and re-execs into it. To use an existing venv instead: `uv sync` yourself and
set `PDF_OCR_NO_SKILL_VENV=1`.

Missing dependencies? The script audits itself and exits with code 2,
printing the exact install commands. Run those, re-run.

## Usage

```bash
python3 scripts/pdf_ocr.py paper.pdf out.md                  # markdown (default)
python3 scripts/pdf_ocr.py paper.pdf out.md --json           # + .pages.jsonl / .summary.json
python3 scripts/pdf_ocr.py paper.pdf out.md --pages 5-7      # subset of pages
python3 scripts/pdf_ocr.py scan.pdf out.md --force-ocr       # skip text-layer path
python3 scripts/pdf_ocr.py paper.pdf out.md --crop 50,100,400,300 --pages 3
python3 scripts/pdf_ocr.py input.png out.md                  # images work too
```

Most useful flags (full list: `--help`):

| Flag | Purpose |
|------|---------|
| `--pages N` / `N-M` | Extract a subset of pages (1-indexed) |
| `--force-ocr` | Bypass the text-layer fast path (mangled text layers, raster-heavy PDFs) |
| `--crop X,Y,W,H` | Extract a region (PDF points, top-left origin; implies `--force-ocr`) |
| `--models a,b,c` | Ordered OCR model cascade (default `rapidocr`; any Ollama vision model name works) |
| `--json` | Structured output: `<base>.pages.jsonl` (per page) + `<base>.summary.json` |
| `--fast` | Skip quality routing; simple char-count threshold |
| `--dpi` / `--max-dim` | Override adaptive render DPI / cap image size (default 1024 px) |
| `--layout-backend` | `doclayout-yolo-onnx` (default) or `rapid-layout` |
| `--table-backend` | `rapidtable` (default), `ppstructure`, or `unitable` |
| `--no-cache` | Bypass the content-hash cache (default: re-runs return in <5% of original time) |
| `--parallel N` | Page-level parallelism for OCR-bound pages (default 1) |

### Output formats

**Markdown (default)** — page provenance comments, reconstructed tables,
typography, and inline warnings where a fix was applied or a validator fired.

**JSON (`--json`)** — one record per page (text, `route`, `route_reasons`,
`warnings`) plus an aggregate summary including `path_warnings` — the audit
trail of every fallback or degradation that occurred.

### Environment variables

| Variable | Effect |
|----------|--------|
| `PDF_OCR_NO_SKILL_VENV=1` | Don't bootstrap/re-exec into the project venv |
| `PDF_OCR_NO_BOOTSTRAP=1` | Skip auto-bootstrap entirely |
| `PDF_OCR_CACHE_DIR` | Move the extraction cache (default `~/.cache/pdf-ocr`) |
| `PDF_OCR_TABLEFORMER=0` | Disable the TableFormer table-structure backend |

### Optional backends

- **Ollama vision models** — degraded scans auto-escalate from RapidOCR to a
  vision model if the daemon is running (`ollama pull glm-ocr`, or any model;
  unknown models get a sane default profile). On failure the page degrades
  back to RapidOCR with a warning — never silently.
- **`uv sync --extra tier2`** — torch specialists (UniMERNet equations,
  DECIMER chemistry, torch DocLayout-YOLO), ~780 MB, opt-in; the default
  install never selects them.
- **`uv sync --extra tableformer`** — IBM TableFormer table-structure backend.

## Repository layout

```
scripts/pdf_ocr.py              CLI: routing, fast path, OCR path
scripts/scientific_parser/      perception · routing · science · postproc · verification
scripts/scientific_parser/tests/   full test suite (pytest)
scripts/shared.py               shared constants (single source of truth)
scripts/pdf_ocr_batch.py        multi-worker batch runner
SKILL.md                        full reference: routing, flags, warnings, troubleshooting
```

## Development

```bash
uv sync --dev                                        # deps + pytest/ruff
uv run python -m pytest scripts/scientific_parser/tests -q
uvx ruff check scripts                               # lint
uv build                                             # wheel/sdist (hatchling)
```

CI (`.github/workflows/verify.yml`) runs the same on every push/PR to `main`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Exit code 2 | Missing deps — read stderr, run the printed install commands |
| `model not found` | `ollama pull <model>` |
| `pdftotext failed` | Encrypted/corrupt PDF: `qpdf --decrypt in.pdf out.pdf` |
| Equations not in LaTeX | Page likely lacks math fonts (check `pdffonts file.pdf`) — try `--force-ocr` |
| Garbage/`---` output | Type-3 bitmap fonts — try a different `--models` entry |

Full warning-code table and routing diagnostics: [SKILL.md](SKILL.md).

## Use as an opencode skill

This repo doubles as an [opencode](https://opencode.ai) skill: clone it under
`~/.config/opencode/skills/pdf-ocr/` and opencode picks up `SKILL.md`
automatically.

## License

[MIT](LICENSE) — the ONNX/torch model weights it downloads carry their own
upstream licenses.
