---
name: pdf-ocr
description: >-
  Extract text from PDF or image to markdown. Hybrid CPU architecture:
  PP-OCRv6 (rapidocr) for OCR, RapidTable for tables, RapidLaTeX for
  equations, DocLayout-YOLO for layout — ONNX-first, torch specialists
  (UniMERNet/DECIMER) CPU-pinned. Ollama vision models (glm-ocr,
  chandra-ocr-2) auto-fallback for degraded scans where rapidocr fails.
  Per-page routing: born-digital pages use pdftotext fast path
  (0.1s/page), scanned pages use OCR. TableAgent
  (ensemble locate → extract → validate → emit), EquationAgent (native
  Unicode→LaTeX → validate → model fallback), column-aware reading order,
  geometry-driven table reconstruction, HTML block protection, 15+ scientific
  post-processors. Use when user wants PDF text extraction, OCR, or document
  digitization.
model: inherit
inheritProjectContext: true
inheritSkills: true
allowed-tools: Bash, Read, Write, Glob, Grep, AskUserQuestion
argument-hint: <path-to-pdf-or-image>
metadata:
  mcpmarket-version: 3.1.0
---

# PDF OCR — SOTA Hybrid Routing + Geometry-Driven Document Understanding

## Quick Start

```bash
python3 ~/.config/opencode/skills/pdf-ocr/scripts/pdf_ocr.py "<input>" "<output>" [options]
```

**Skill venv auto-bootstraps** on first run: `uv sync --project <skill-root>`
from `pyproject.toml` (the single source of dependency truth). Installs the
full dep tree (ONNX specialists + CPU-pinned torch specialists). One-time
cost. Re-execs into skill venv on every subsequent run. Opt-out env:
`PDF_OCR_NO_SKILL_VENV=1`.

The script **self-checks all dependencies** and prints actionable install
instructions for anything missing. You do NOT need to pre-check deps — just
run it. If it exits with code 2, read stderr for install commands, offer them
to the user, then re-run.

## Decision Tree — Which Flags?

| Situation | Flags | Why |
|-----------|-------|-----|
| **Unknown PDF (default)** | *(none)* | SOTA routing: born-digital→fast (pdftotext), scanned→OCR (rapidocr PP-OCRv6 small). |
| **Known scanned document** | *(none)* | Router detects scanned pages automatically + Ollama VL auto-fallback on failure. |
| **Math/equation-heavy paper** | *(none)* | EquationAgent: native Unicode→LaTeX first (70% skip model), RapidLaTeX fallback. |
| **Table-heavy document** | *(none)* | TableAgent: PyMuPDF+pdfplumber native first, RapidTable CNN fallback. 95% ensemble detection. |
| **Mangled equations/tables** | `--force-ocr` | Bypass pdftotext fast path; use OCR + specialists. |
| **Different OCR engine** | `--models rapidocr,glm-ocr` | Cascade: multi-model chain with confidence-floor advance + reflexive retry; on timeout/failure cascades to next model, rapidocr as final degrade. |
| **Table structure backend** | `--table-backend rapidtable` | RapidTable (SLANet-Plus, default). Alternatives: ppstructure, unitable. |
| **Layout detection backend** | `--layout-backend doclayout-yolo-onnx` | DocLayout-YOLO (default, SOTA). Alternatives: rapid-layout. |
| **Agent/programmatic use** | `--json` | Emits `.pages.jsonl` (streaming, crash-safe) + `.summary.json`. |
| **Specific page or figure** | `--pages 5` or `--pages 5-7` | Targeted extraction. Faster, less tokens. |
| **Crop equation/figure region** | `--crop X,Y,W,H --pages N` | PDF-points, top-left origin. Implies `--force-ocr`. |
| **Speed priority** | `--fast` | Disables quality scoring; chars<100→OCR only. |

## Execution Rules

1. **Output path**: choose based on use case:
   - Human reading: `<pdf_dir>/<basename>.md`
   - Agent processing: `<temp dir>/<basename>.md` or `--json` for structured

2. **Page count determines foreground/background**:
   - ≤10 pages: foreground, `timeout: 600000`
   - \>10 pages: background, `timeout: 600000`
   - Count requested pages (after `--pages`), not total document

3. **Never run the script twice on the same input** without checking output
   first — the first run may still be processing in background.

## What the Script Does (Automatic — No Flags Needed)

### Routing Pipeline

```
Input PDF
    ↓
preclassify_document() ← ONE PDF open, ONE pdffonts call
    ↓                    classifies born-digital vs scanned
    ↓
Per page:
    ├─ Born-digital + good text quality → FAST (pdftotext, 0ms/page)
    │   ├─ Column-aware reading order (PyMuPDF block sort)
    │   ├─ Geometry-driven table reconstruction (span clustering + grid)
    │   ├─ Footnote detection & relocation
    │   ├─ Typography preservation (italic/bold/mono/superscript)
    │   ├─ Caption association (Fig/Table markers)
    │   └─ 12 scientific post-processors (see below)
    ├─ Scanned / garbled / fragmented → OCR (vision model, ~5-15s/page)
    │   ├─ Adaptive DPI (150/250/300 by font analysis)
    │   ├─ Display equation detection → math-prompt OCR
    │   ├─ Thinking artifact stripping (</think> removal)
    │   └─ Same 12 scientific post-processors
    └─ Invisible text + good quality → FAST (trusted, Adobe Clearscan OK)
```

Post-processing: all pages (FAST + OCR) get the same scientific post-processors.

### Fast Path Enhancements (born-digital)

- **Column-aware reading order** (PyMuPDF): classifies text blocks as
  left/right/full-width, interleaves at correct Y positions. Handles 2-column
  scientific layouts (GCA, CMP, JPet, AmMin).
- **Geometry-driven table reconstruction**: `find_tables()` for table bbox only,
  then raw span coordinates clustered into grid cells (1D DBSCAN). Quality gate:
  ≥3 rows, ≥20% numeric cells. Fallback to fitz cell boundaries with narrow-column
  filtering for dense grids. No `cell.extract()` — perfect Unicode preservation.
- **Subscript detection** (PyMuPDF): chemistry `H$_{2}$O`, `YPO$_{4}$`,
  isotopes `$_{97}$Mo/$_{95}$Mo`. Works on 2-column papers.
- **Hybrid math pipeline**: detects private/control chars (encoding failures),
  renders just those regions via PyMuPDF `search_for()`, OCRs with math-aware
  prompt, splices OCR result back.
- **Typography preservation**: italic → `*text*`, bold → `**text**`,
  monospace → `` `text` ``, superscript → `$^{text}$`. Via fitz font flags.
- **Caption association**: detects Fig./Table captions, inserts structured markers.
- **Footnote reconstruction**: detects bottom-region + separator footnotes,
  moves after body under `**Footnotes:**`.
- **Header/footer learning** (multi-page): hash-normalized recurring header
  detection (>70% pages), strips across document.

### OCR Path Enhancements (scanned/garbled)

- **Adaptive DPI**: font analysis via PyMuPDF — 150 DPI normal (≥8pt),
  250 DPI dense scientific (6-8pt), 300 DPI tiny/scanned (<6pt).
- **Display equation detection**: centered blocks with math operators →
  crop + OCR with `MATH_OCR_PROMPT` (LaTeX-aware).
- **Thinking artifact stripping**: removes `<think>...</think>` blocks from
  Qwen3.5-based models (ovisocr2 etc.).
- **System prompt**: model profiles support per-model `system` field for
  behavior control (e.g., suppress thinking mode).

### Scientific Post-Processors (all pages)

| Processor | What It Does | Example |
|-----------|-------------|---------|
| **Per-mille fix** | `%` → `‰` in isotope context | `0.5% per amu` → `0.5‰ per amu` |
| **Chemical notation** | ASCII → Unicode subscripts | `SiO2` → `SiO₂`, `Fe2+` → `Fe²⁺` |
| **Chemical formula validator** | periodictable-validated repair | `Si02` → `SiO₂`, `AI2O3` → `Al₂O₃` |
| **Scientific symbol validator** | Context-aware OCR confusion fix | `10 W` → `10 Ω`, `5 x 3` → `5 × 3` |
| **REE sequence validator** | Detects anomalies, warns only | `Cd` for `Gd`, duplicates, reversed |
| **Mineral abbreviations** | Kretz/IUGS standard detection | `Bt.` → warns should be `Bt` |
| **Phantom decimal sanitizer** | Fixes OCR-inserted decimals in `<td>` | `8870.7` → `88707` |
| **Dehyphenation** | Joins broken words, preserves compounds | `meta-\nmorphism` → `metamorphism`, keeps `U-Pb` |
| **Citation repair** | Rejoins split citation brackets | `[12]-\net al.` → `[12] et al.` |
| **DOI preservation** | Joins DOIs split across lines | `10.1234/abc-\ndef` → `10.1234/abcdef` |
| **Bibliography mode** | Lighter processing for References | Skips chemical/symbol fixes in refs |
| **Oxide total validator** | Checks sum ≈ 100±5% | Flags anomalous totals |
| **Age validator** | Checks U-Pb ages within geological range | `5.2 Ga` → `age_implausible` |

### Document-Level Pre-Classification

`preclassify_document()` opens the PDF **once** and runs `pdffonts` **once**
for all pages. Returns cached `doc_info` that eliminates per-page PDF opens
in the routing loop. Born-digital documents route in O(1) per page.

### Model-Agnostic Architecture

```python
MODEL_PROFILES = {
    "glm-ocr": {"temperature": 0.0, "num_predict": 8192, "num_ctx": 16384,
                "repeat_penalty": 1.1},
    "ovisocr2": {"temperature": 0.0, "num_predict": 8192, "repeat_penalty": 1.1,
                 "system": "You are an OCR engine. Transcribe text exactly. No thinking."},
}
# DEFAULT_MODEL_PROFILE (unknown models): num_predict/num_ctx = 131072
# (paddleocr-vl needs the headroom; per-model profiles above tighten it).
```

Any Ollama vision model works out of the box. Unknown models use
`DEFAULT_MODEL_PROFILE`. Add new models by editing `MODEL_PROFILES`.

## Output Format

### Markdown (default)

```markdown
<!-- Source: pdftotext (text layer) -->

<!-- Page 1 | PageHeader: ... | PageNumber: ... -->

[page text with SiO₂ subscripts, markdown tables, *italic*, **bold**]

---

<!-- Page 2 | WARNING: encoding_artifacts (3 chars cleaned) | ree_ocr_confusion: 'Cd' likely 'Gd' -->

[page text]
```

### JSON (`--json`)

Two files: `<base>.pages.jsonl` (streamed, one JSON per line) +
`<base>.summary.json` (aggregate).

Page record includes `warnings` array with scientific validator output
(REE anomalies, oxide totals, mineral abbreviations, etc.).

## Warning Types and Agent Actions

| Warning | Meaning | Agent Action |
|---------|---------|-------------|
| `low_text_density (N chars)` | Page has very little text (figure-heavy) | Normal for figure pages. |
| `encoding_artifacts (N chars cleaned)` | Private chars fixed by hybrid pipeline | Already fixed. |
| `encoding_artifacts (N chars — run with --force-ocr)` | Not fully cleaned | Suggest `--force-ocr` for that page. |
| `ree_ocr_confusion: 'X' likely 'Y'` | OCR misread REE symbol | Check context, likely a typo. |
| `ree_duplicate` / `ree_reversed` / `ree_missing` | REE sequence anomaly | Review data table. |
| `oxide_total_anomalous: sum=N%` | Oxide weights don't sum to ~100 | Check for missing/extra values. |
| `age_implausible: N Ga` | Age exceeds Earth's age (4.54 Ga) | Check OCR or data entry. |
| `mineral_abbrev_dot: 'Bt.'` | Trailing dot on mineral abbreviation | Remove dot. |
| `no_text_recognized` | OCR returned empty | Page may be blank/image-only. |

### Path Warnings (unintended-path detection)

Whenever the workflow takes an unintended path — a born-digital page routed
to OCR, a doc-level fallback, an engine/model degradation, a specialist
fallback — the pipeline emits `PATH WARNING [code] ...` to stderr
immediately, aggregates them into an end-of-run block (stderr), and records
them under `path_warnings` in `.summary.json`. Page records also carry
`route` (`fast`/`ocr`) and `route_reasons` for audit.

| Code | Meaning | Agent Action |
|------|---------|-------------|
| `born_digital_ocr_route` | Page(s) with embedded fonts routed to the OCR path (routing false-positive or broken text layer; reasons included) | Check reasons: encoding issues = legitimate; layout false-positives = inspect `--force-ocr` usage / routing signals. |
| `pdftotext_unavailable` | pdftotext missing/failed — ALL pages fell back to OCR | Install poppler-utils; encrypted PDF → `qpdf --decrypt`. |
| `min_doc_ocr_forced` | Too few good pages — fast pages force-routed to OCR | Expected for genuinely bad text layers; born-digital docs = investigate. |
| `vl_fallback_used` | rapidocr output poor → Ollama VL re-OCR (or VL re-OCR failed) | Degraded-scan handling; check `model_failure_degraded` too. |
| `model_failure_degraded` | OCR model failed after retries → page degraded to rapidocr | Transient Ollama issue or model missing; retry or `ollama pull`. |
| `ocr_model_swap` | Chat-mode guard / self-corrective rescue / reflexive cascade swapped or retried the model | Informational; primary model underperformed. |
| `fast_path_ocr_escalation` | Fast-path text layer unusable → page re-OCR'd | Legitimate rescue; may indicate broken ToUnicode. |
| `layout_backend_fallback` | DocLayout-YOLO-ONNX unavailable → RapidLayout (or none) | Offline/weights issue; layout quality reduced. |
| `layout_detect_failed` | Layout detection threw — page fell back to text-layer/OCR | Informational; per-page degradation. |
| `region_routing_no_regions` | Layout found no regions — text-layer/OCR kept | Normal for simple pages. |
| `region_output_rejected` | Region-routed output rejected by adoption gates — page text kept | Quality gates working; no action. |
| `specialist_fallback` | Equation/table/chemical specialist failed — region fell back to text clip | Informational; specialist degraded that region. |
| `table_model_fallback` | Native table weak/absent → RapidTable CNN | Informational; slower path taken. |
| `equation_model_fallback` | Native Unicode→LaTeX weak → RapidLaTeX CNN | Informational; slower path taken. |
| `stage_skipped` | A pipeline stage was skipped by exception (stage named in message: polishers, crosschecks, rescue, conservation) | Inspect message; a skipped final pass may degrade output. |
| `page_count_fallback` | pdfinfo (or pdfinfo+PyMuPDF) failed — page count from fallback tier | Environmental; poppler health check. |
| `page_render_fallback` | PyMuPDF page render failed — pdftoppm subprocess used | Slower render path; check fitz install. |
| `script_extraction_fallback` | pypdfium2 subscript extraction mismatched/diverged/crashed — plain pdftotext used | Sub/superscript fidelity lost on that page. |
| `ocr_prompt_fallback` | VL model returned empty for task prompt — re-sent with plain transcription prompt | Model/prompt incompatibility; usually recovers. |
| `specialist_providers_missing` | Specialist modules absent at dep audit — heuristic fallback mode | `uv pip install -e .[specialists]`. |
| `cli_config_ignored` | CLI threshold overrides failed to wire — module defaults in effect | Flags silently ignored; check import wiring. |
| `font_math_cascade_failed` | Font-math cascade threw on a page — LaTeX reconstruction lost, raw text kept | Page outputs prose math; inspect exception. |
| `ocr_engine_legacy_api` | RapidOCR v1.4.4 legacy tuple API detected | Upgrade rapidocr to v3. |
| `table_detect_failed` | Table detection stage raised — page text kept, tables not appended (never silent; iter-8 class) | Inspect exception in logs; page needs table re-extraction. |
| `figure_table_reroute` | DocLayout-YOLO labelled a region "figure" but raster-coverage + OCR word-grid probe found a raster table — rerouted through TableAgent | Raster tables exported as images now extract as tables. |

| `span_repair_applied` | Font-math cascade dropped math-font spans during assembly — span-conservation pass re-injected them (iter-116; the 66% span-gap failure family) | Repair is telemetry-visible by design; if a repaired page misbehaves, set `PDF_OCR_SPAN_REPAIR=0` and inspect the cluster geometry. |

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Exit code 2 | Missing deps (Ollama/poppler/PIL/ftfy/pypdfium2/pymupdf/periodictable/pdfplumber/symspellpy) | Read stderr for install commands. **All deps are HARD** (no silent degradation). |
| Exit code 1, "model not found" | Model not pulled | `ollama pull <model>` |
| `--force-ocr` warning on stderr | Bypassing pdftotext on born-digital | Remove `--force-ocr` for born-digital PDFs. |
| `Multiple models specified` | Cascade chain | Reflexive loop advances on low confidence; `--always-model` pins one model instead. |
| Ollama transient HTTP 5xx | Auto-retried (2x backoff) | Persistent failure degrades page to rapidocr with `model_failure_degraded_rapidocr` warning. |
| `pdftotext failed` | Encrypted/corrupt PDF | `qpdf --decrypt in.pdf out.pdf` |
| Subscripts missing | born-digital used wrong font (not CMMI/CMMI7) | Check via `pdffonts file.pdf` — math fonts must be present |
| Chemical formulas not validated | periodictable not installed (HARD) | `uv add periodictable` |
| Table not detected | Page has non-standard layout | 6-strategy detector: pdfplumber + lines_strict + text_only_pattern + ocr_text_parse + camelot_lattice + camelot_stream |
| Math not extracted as LaTeX | Page doesn't use Computer Modern fonts | Math font detection covers CMMI/CMSY/CMEX/MSBM/MSAM/EU1/EU2/Frak |
| OCR returns garbage (`---` x N) | Vision model failed on Type 3 bitmap fonts | Auto-detected + collapsed. Try a different `--models` for math-heavy pages. |
| OCR repetition ("text contains no visible characters" x N) | Model hallucination loop | Auto-detected + collapsed with marker. |
| Positioned-glyph PDF (one word per line) | Old Elsevier scans, sakai1968-style | Auto-routed to OCR via `positioned_glyphs` signal (avg_wpl < 5) |

## Technical Details

- **Adaptive DPI**: 150 (≥8pt fonts) / 250 (6-8pt) / 300 (<6pt) via PyMuPDF font analysis
- **Max image dimension**: 1024px (Pillow LANCZOS resize). Override with `--max-dim`. Warning: >1024 may truncate OCR output on some vision models.
- **Model profiles**: `MODEL_PROFILES` dict — per-model Ollama options (`num_predict=32768`, `num_ctx=32768`, `repeat_penalty=1.1`)
- **Default model**: `glm-ocr` (override with `--models <name>`). Any Ollama vision model works.
- **Table reconstruction**: multi-strategy detector (strategies 1-4 always
  available; 5-6 camelot lattice/stream require `camelot-py` — NOT a core
  dep, install separately to enable)
  1. `pdfplumber_text` — decimal-bearing tables (oxide analyses, run data)
  2. `lines_strict` — ruled-line tables via PyMuPDF
  3. `text_only_pattern` — tables WITHOUT decimals (mineral abbreviations)
  4. `ocr_text_parse` — vision model's own markdown tables (validated)
  5. `camelot_lattice` — ruled tables via OpenCV line-intersection
  6. `camelot_stream` — whitespace tables via kmeans (with body-text false-positive guard)
  - Dedup by bbox overlap >50%, multi-row header collapse, cross-page table merging
- **Math extraction** (font-based, non-ML):
  - Detects math spans via font family (CMMI/CMSY/CMEX/MSBM/MSAM/EU1/EU2/Frak)
  - 100+ Unicode → LaTeX mappings (∑→`\sum`, ∏→`\prod`, ∫→`\int`, ≤→`\leq`, etc.)
  - Subscript/superscript via font size + Y-offset from baseline
  - Display vs inline via CMEX big-operator presence OR centered+wide bbox
- **Scientific post-processors**: degree-symbol restoration (8C→°C), SymSpell OCR correction (Scienees→Sciences via geology dict + English 82k), variable subscripts (RC→R$_{C}$), isotope superscripts (10Be→$^{10}$Be), unit powers (g cm-2→g cm$^{-2}$)
- **Figure region separation**: classify_text_block distinguishes BODY from FIGURE_GLYPH/FIGURE_TICK/FIGURE_LABEL_BLOCK/AXIS_TITLE; figure content moved to `<!-- FIGURE_REGION -->` block (preserved verbatim, no loss)
- **Column-aware reading order**: band-based Y-grouping (25pt), X-gap joining (<20pt = same line, handles positioned-glyph PDFs)
- **Multi-page PageHeader consensus**: requires ≥3-page recurrence; single-page ALLCAPS strings (author names on title page) NOT promoted
- **Prompt**: `"Text Recognition:"` (minimal, best quality)
- **Temp files**: `tempfile.mkdtemp()`, cleaned in `finally` block
- **Dependencies (ALL HARD — no silent degradation)**:
  - PIL/Pillow (image handling)
  - poppler-utils: pdftotext, pdftoppm, pdfinfo, pdffonts, pdfimages
  - Ollama + vision model (default glm-ocr, override via `--models`)
  - ftfy (Unicode fix_text)
  - pypdfium2 (invisible text detection, char-bbox analysis)
  - pymupdf/fitz (reading order, tables, DPI, typography, math fonts)
  - pymupdf4llm (SOTA markdown extraction)
  - pdfplumber (table detection, char-level geometry)
  - periodictable (chemical formula validation)
  - symspellpy (OCR error correction)
  - camelot-py (OPTIONAL — lattice + stream table strategies; not in core deps)
- **Platform**: Linux, macOS, Windows (cross-platform; venv bootstrap uses
  `Scripts\python.exe` on Windows)
- **Environment variables** (all optional):
  | Var | Effect |
  |---|---|
  | `PDF_OCR_NO_SKILL_VENV=1` | Skip skill-venv re-exec (use cwd interpreter) |
  | `PDF_OCR_NO_BOOTSTRAP=1` | Skip auto `uv sync` on first run |
  | `PDF_OCR_CACHE_DIR` | Extraction cache location (default `~/.cache/pdf-ocr`) |
  | `PDF_OCR_DEBUG_CASCADE=1` | Verbose cascade decision logging |
  | `PDF_OCR_KATEX_STRICT=1` | KaTeX strict-mode math gate (default off: drops ~3% valid math) |
  | `PDF_OCR_NO_CASCADE=1` | Disable equation cascade |
  | `PDF_OCR_STACKED_FRAC=1` | Experimental stacked-fraction merge (known CMEX false-merges) |
  | `DOCLAYOUT_YOLO_MODEL` | Override DocLayout-YOLO model path |
  | `BENCH_OCR_MODEL` | bench.py default `--models` value |
- **Skill version**: 2026-08-16 (v10 — audit-fixed: retry/degrade paths,
  events+ledger wiring, single-sourced constants)

## Self-improvement loop

This skill is designed to be self-improving. During use, if the AI agent
detects any issues, it should act based on severity:

**Minor issues** — the agent **should fix the skill itself directly** in
`~/.config/opencode/skills/pdf-ocr/scripts/pdf_ocr.py` (via Serena), then
re-run on the failing PDF to verify, then add a regression test to
`tests/test_pdf_ocr_qc.py` if the user's project has tests. **User
approval required** before applying any edit.
Minor issues include:

- **High-ROI bug fixes**: garbled output, missing equations, false-positive tables,
  silent failures, encoding artifacts (U+FFFD), repetition loops
- **New OCR models available** (e.g. via `ollama pull`) that may improve accuracy
  on specific document types (math-heavy, multi-language, handwriting)
- **Recurring failure patterns** on a document class (e.g. always fails on
  pre-1990 Elsevier scans, always drops equations from amminmin typeset)
- **Missing dictionary terms** for OCR correction (geology terms, mineral names,
  author surnames that should be in `_GEOLOGY_DICT_TERMS`)
- **Unicode→LaTeX mappings missing** (any math symbol rendered as raw Unicode
  instead of `\command` — extend `UNICODE_TO_LATEX` dict)
- **Layout patterns not handled** (multi-column with sidebars, footnotes in
  gutters, marginalia, multi-page tables with continued headers)

**Improvement protocol (minor)**:
1. Stop and describe the issue + proposed fix to the user
2. Wait for explicit user approval
3. Apply fix to `~/.config/opencode/skills/pdf-ocr/scripts/pdf_ocr.py` via Serena
4. Re-run on the failing PDF to verify
5. Add regression test to `tests/test_pdf_ocr_qc.py` (if user's project has tests)
6. Update this SKILL.md if behavior changed

**Major failures or architectural issues** (vision-model integration broken,
SOTA per-page routing failure, geometry/table-reconstruction regression,
new Ollama API breaking change, performance degradation, security
vulnerability, Unicode→LaTeX table wholesale drift): the agent **must STOP,
report the issue with full diagnosis** — what broke, why, blast radius,
evidence (logs/output), and a proposed architectural fix or improvement
with trade-offs. **Do not attempt a patchwork fix**. Wait
for user decision before proceeding.

**Never silently work around a bug** — surface it, propose a fix, await approval.
This skill is the AGENT'S tool — improving it improves every future invocation.
