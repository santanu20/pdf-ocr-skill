"""OCR engine — Ollama HTTP client + RapidOCR ONNX + image utilities.

Extracted from scripts/pdf_ocr.py during Phase R9 refactoring (2026-08-12).
"""

from __future__ import annotations

import re
import os
import sys
import time
import json
import base64
import struct
import logging
import urllib.error
import urllib.request
import subprocess
from scientific_parser.core.path_warnings import (
    PAGE_RENDER_FALLBACK,
    OCR_ENGINE_LEGACY_API,
    OCR_PROMPT_FALLBACK,
    record_path_warning,
)
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state

_RAPIDOCR_INSTANCE = None
_RAPIDOCR_MODEL_TYPE = None


def _pdf_ocr():
    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


def pdf_page_to_png(pdf_path, page_num, output_prefix, dpi=None):
    """Convert a single PDF page to PNG at given DPI.

    Uses PyMuPDF (fitz) for in-process rendering — 10x faster than pdftoppm
    subprocess (1.8s vs 18s for a large-format scan at 150 DPI). Falls back
    to pdftoppm if fitz is unavailable.
    """
    render_dpi = dpi or DPI
    out_path = f"{output_prefix}-{page_num}.png"

    # Primary: PyMuPDF in-process rendering (fast, no subprocess overhead).
    try:
        import fitz as _fitz

        with _fitz.open(pdf_path) as _doc:
            _pg = _doc[page_num - 1]
            _pix = _pg.get_pixmap(dpi=render_dpi)
            _pix.save(out_path)
        return out_path
    except Exception as e:
        log.debug("pdf_page_to_png: PyMuPDF render failed (%s) — trying pdftoppm", e)

        record_path_warning(
            PAGE_RENDER_FALLBACK,
            "PyMuPDF page render failed — pdftoppm subprocess fallback used",
        )

    # Fallback: pdftoppm subprocess (slower, but works without fitz).
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(render_dpi),
            pdf_path,
            output_prefix,
        ],
        capture_output=True,
        check=True,
    )
    import glob

    candidates = (
        glob.glob(output_prefix + f"-{page_num}*.png")
        + glob.glob(output_prefix + f"-{page_num:02d}*.png")
        + glob.glob(output_prefix + f"-{page_num:03d}*.png")
    )
    if not candidates:
        raise FileNotFoundError(
            f"pdftoppm produced no output matching {output_prefix}-*{page_num}*.png"
        )
    return candidates[0]


def resize_image(img_path, max_dim):
    """Resize image to fit within max_dim using Pillow (cross-platform).

    Falls back to `sips` on macOS if Pillow is unavailable.
    Returns the resized path if a new file was written, otherwise the input path.
    """
    from PIL import Image  # HARD dep

    with Image.open(img_path) as im:
        w, h = im.size
    if max(w, h) <= max_dim:
        return img_path
    resized_path = img_path.replace(".png", "_resized.png")
    with Image.open(img_path) as im:
        # Pillow >= 9.1: Image.Resampling.LANCZOS (enum). Older: Image.LANCZOS (int).
        try:
            resampling = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:
            resampling = Image.LANCZOS  # type: ignore[attr-defined]
        im.thumbnail((max_dim, max_dim), resampling)
        im.save(resized_path)
    return resized_path


# =============================================================================
# Model-Agnostic Profile System
# =============================================================================
# Per-model Ollama options. Unknown models fall back to DEFAULT_MODEL_PROFILE.
# Add profiles here when a model needs different temperature/token-limit/penalty.
# Validated via sweep (June 2026): defaults are quality-optimal for scientific
# PDFs (chemistry, citations, equations, tables, two-column layouts).
#
# 2026-07-25 update: bumped num_predict 8192 → 32768 after discovering it was
# the real bottleneck (not MAX_DIM). glm-ocr supports 131072-token context
# (verified via ollama show). At 8192, dense pages (long abstracts, multi-
# paragraph tables) truncated mid-output. 32768 = ~128KB text headroom.
# 2026-07-25: num_ctx is the REAL bottleneck — Ollama defaults to 4096 if
# not set explicitly, even when the model supports 131072 (glm-ocr verified).
# At 4096, a single 1024x1024 image (~2000-4000 image tokens) leaves only
# ~500 tokens for output → truncation on dense pages. Setting num_ctx=32768
# unlocks full num_predict headroom for any page density.
MODEL_PROFILES = {
    "chandra": {
        "temperature": 0.0,
        # 300-DPI page ≈ 4.3k vision tokens + layout prompt (~350) +
        # structured output (~3-6k) — needs ≥11k context. The server's
        # 4096 default silently TRUNCATED high-DPI reads to garbage
        # (measured 2026-08-19: request 4313 tokens > 4096 → HTTP 400;
        # our chandra old_scans 42.4% was a context-strangled model).
        # 16384 validated on 4MB 300-DPI renders; 131072 KV on a 13GB
        # host with parallel workers risks OOM.
        "num_predict": 8192,
        "num_ctx": 16384,
        "repeat_penalty": 1.1,
    },
    "paddleocr-vl": {
        # PaddleOCR-VL-1.6 (0.9B): page transcription is 300-1500 tokens.
        # DEFAULT profile's num_predict=131072 lets degenerate generations
        # (qquad/cdot loops on tiny crops) run for many minutes → bench
        # 120s timeouts. 8192 (was 4096, 2026-08-15): dense patent pages
        # hit ~4k output tokens and TRUNCATED mid-page (measured: 6.8k
        # chars cut, order tests failing on tail text). 8k caps degenerate
        # loops with 5x headroom over real pages.
        "num_predict": 8192,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
    "glm-ocr": {
        "temperature": 0.0,
        # num_ctx=16384: image (~2-4K tokens) + prompt (~200) + output (~3K) = ~6K used.
        # Reserving more VRAM (e.g. 131072) slows attention with no quality gain.
        # num_predict=8192: OCR output is rarely >3K tokens; 8K is comfortable headroom.
        "num_predict": 8192,
        "num_ctx": 16384,
        "repeat_penalty": 1.1,
    },
    # qwen2.5vl — 2026 OCR winner (nullmirror benchmark). Needs /api/chat
    # endpoint (auto-detected in ocr_image). num_ctx=8192 fits 8GB VRAM with
    # image tokens.
    "qwen2.5vl": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "qwen2.5vl:3b": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "qwen2.5vl:7b": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "qwen3-vl": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "qwen3-vl:4b": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "ovisocr2": {
        "temperature": 0.0,
        "num_predict": 131072,
        "num_ctx": 131072,
        "repeat_penalty": 1.1,
        # Suppress thinking-mode artifacts (Qwen3.5 chat template always emits <think>...</think>)
        "system": "You are an OCR engine. Transcribe text from the image exactly as shown. Do not add commentary, explanations, or reasoning.",
    },
    # dots.ocr (rednote-hilab) — HIGHEST ACCURACY on scientific PDFs:
    # correctly formats LaTeX subscripts ($f_{O_2}$), captures most decimals.
    # Trade-off: ~7x slower than glm-ocr (6.5 min vs 56s per page). Use for
    # small critical docs where accuracy > speed.
    # Pull: ollama pull hf.co/ggml-org/dots.ocr-GGUF:Q8_0
    # Alias created via: ollama create dots.ocr -f Modelfile (FROM hf.co/...)
    "hf.co/ggml-org/dots.ocr-GGUF:Q8_0": {
        "temperature": 0.0,
        "num_predict": 32768,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
    "dots.ocr": {  # short alias
        "temperature": 0.0,
        "num_predict": 32768,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
    # dots.mocr (SandLogic IQ4_XS quantization) — FASTER but less accurate
    # than Q8_0. Introduces "liquidous"/"komalites"-style errors due to 4-bit
    # quantization. Use only when speed > accuracy.
    "hf.co/SandLogicTechnologies/dots.mocr-GGUF:IQ4_XS": {
        "temperature": 0.0,
        "num_predict": 32768,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
    "dots.mocr": {  # short alias
        "temperature": 0.0,
        "num_predict": 32768,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
}

DEFAULT_MODEL_PROFILE = {
    "temperature": 0.0,
    "num_predict": 131072,
    "num_ctx": 131072,
    "repeat_penalty": 1.1,
}


def get_model_profile(model):
    """Return Ollama options dict for a given model.

    Prefix-matched against MODEL_PROFILES (so "paddleocr-vl:1.6" hits the
    "paddleocr-vl" profile); falls back to DEFAULT_MODEL_PROFILE for
    unregistered models — the skill works with any Ollama vision model
    out of the box.

    Profiles MERGE over DEFAULT_MODEL_PROFILE (iter-54): a partial profile
    (paddleocr-vl registered without 'temperature') made ocr_image's
    payload assembly raise KeyError BEFORE any HTTP call — every
    paddleocr-vl invocation died and silently degraded to rapidocr.
    """
    if model in MODEL_PROFILES:
        profile = MODEL_PROFILES[model]
    else:
        ml = model.lower()
        profile = next(
            (
                p
                for prefix, p in MODEL_PROFILES.items()
                if ml.startswith(prefix.lower())
            ),
            DEFAULT_MODEL_PROFILE,
        )
    merged = dict(DEFAULT_MODEL_PROFILE)
    merged.update(profile)
    return merged


# Models whose chat template makes bare task labels ("Text Recognition:")
# hallucinate reasoning instead of transcribing. These need an explicit
# instruction (measured: paddleocr-vl:1.6 rambled "Here are some examples"
# on bench old_scans with the default task prompt).
CHANDRA_LAYOUT_PROMPT = """OCR this image to HTML, arranged as layout blocks. Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format. Bboxes are normalized 0-1000. The data-label attribute is the label for the block.
Use labels: Caption, Footnote, Equation-Block, List-Group, Page-Header, Page-Footer, Image, Section-Header, Table, Text, Complex-Block, Code-Block, Form, Table-Of-Contents, Figure, Chemical-Block, Diagram, Bibliography, Blank-Page.
Only use tags math br i b u del sup sub table tr td p th div pre h1 h2 h3 h4 h5 ul ol li input a span img hr tbody small caption strong thead big code chem, and attributes class colspan rowspan display checked type border value style href alt align data-bbox data-label.
Guidelines:
* Inline math: Surround math with <math>...</math> tags. KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting, including subscripts/superscripts.
* Images: Describe images in the alt attribute of an <img> tag. Do not fill out the src property.
* Text: join lines together properly into paragraphs using <p>...</p> tags. Use <br> only when necessary.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure.
* Text accurate, reading order correct and natural."""

_PROMPT_OVERRIDE_MODELS = {
    "paddleocr": "OCR:",  # official task label per HF model card
    # chandra-ocr-2 official ocr_layout protocol (datalab-to/chandra
    # OLMOCR_BENCH.md, 85.8% reference): layout-block HTML. Measured
    # 17_pg4: bare label 0/7 gold, official prompt 7/7.
    "chandra": CHANDRA_LAYOUT_PROMPT,
}


def resolve_task_prompt(model, task):
    """Substitute hallucination-prone task prompts with explicit instructions.

    Only bare task LABELS ("Text Recognition:") are substituted — these
    make chat-templated OCR models ramble. Full instruction prompts
    (MATH_OCR_PROMPT for equation crops) pass through: callers control
    region-level prompts.
    """
    if len(task) >= 40 or "." in task.rstrip("."):
        return task
    for prefix, prompt in _PROMPT_OVERRIDE_MODELS.items():
        if prefix in model.lower():
            return prompt
    return task


# Module-level RapidOCR engine — PP-OCRv6 MEDIUM (default since 2026-08-12).
# arXiv 2606.13108 — "surpasses billion-scale VLMs on OCR tasks". PP-OCRv6
# medium: det Hmean 86.2%, rec acc 83.2% (+2-3 pts over small). ~132MB models
# (det 59MB + rec 73MB). In-process ONNX runtime, no HTTP, no daemon.
# ~6-12s/page on CPU. SOTA non-VLM OCR — works offline, zero data egress.
#
# Why PP-OCRv6 medium as default (was glm-ocr + Ollama):
#   1. No daemon required — works offline, zero setup beyond `uv add rapidocr`
#   2. Zero data egress (in-process, never leaves the machine)
#   3. +2-3 pts accuracy over PP-OCRv6 small on benchmark
#   4. CPU-friendly (no GPU needed for default operation)
#   5. Deterministic — same input always produces same output (VLMs vary)
#
# To force a different tier: pass model_type to _get_rapidocr() or use
# --models rapidocr:small / --models rapidocr:medium explicitly.
# To use a VLM instead: --models glm-ocr (still supported via Ollama).


def _get_rapidocr(model_type: str = "small"):
    """Lazily build RapidOCR engine. Returns None if not installed.

    Args:
        model_type: "small" (default — PP-OCRv6 small, 8MB, ~2s/page, fast for
                    CPU), "medium" (PP-OCRv6 medium, 134MB — SOTA accuracy but
                    >300s load on CPU, opt-in only via explicit param), "tiny".

    PP-OCRv6 small is the production default: fast + proven (69% on old_scans).
    For higher accuracy on degraded scans, the Ollama VL fallback (chandra-ocr-2,
    76%) fires automatically when small's output is poor — that path is both
    faster (small handles the common case) AND more accurate on the tail than
    medium. Medium sits in an awkward middle (slower than small, less accurate
    than VL) and its 134MB load is impractical for CPU-only.

    rapidocr auto-downloads the selected tier from modelscope.cn on first use
    (SHA256-verified, cached after). Uses the ``ModelType`` enum — NOT
    ``model_name`` (that cfg field doesn't exist + is silently ignored).
    """
    global _RAPIDOCR_INSTANCE, _RAPIDOCR_MODEL_TYPE
    if (
        _RAPIDOCR_INSTANCE is not None
        and getattr(_RAPIDOCR_INSTANCE, "_model_type", "") == model_type
    ):
        return _RAPIDOCR_INSTANCE
    try:
        from rapidocr import RapidOCR
        from rapidocr.utils.model_resolver import ModelType
    except ImportError:
        return None
    _TYPE_MAP = {
        "medium": ModelType.MEDIUM,
        "small": ModelType.SMALL,
        "tiny": ModelType.TINY,
        "mobile": ModelType.MOBILE,
        "server": ModelType.SERVER,
    }
    mt = _TYPE_MAP.get(model_type, ModelType.MEDIUM)
    _RAPIDOCR_INSTANCE = RapidOCR(params={"Det.model_type": mt, "Rec.model_type": mt})
    _RAPIDOCR_INSTANCE._model_type = model_type
    return _RAPIDOCR_INSTANCE


# Memoize the most-recent rapidocr output keyed on image path, so the table
# specialist can reuse page-level boxes WITHOUT re-running OCR (region routing
# runs immediately after the page OCR on the same image). Pure redundant-
# compute avoidance — not a behavior switch. Single entry is sufficient since
# processing is page-sequential.
_LAST_RAPIDOCR: dict = {}


def _rapidocr_output_to_result(output):
    """Parse a RapidOCROutput into (text, boxes, txts, scores).

    Guards ndarray truthiness (output.boxes is an ndarray — `or []` raises).
    """
    if hasattr(output, "txts"):
        txts = list(output.txts) if output.txts is not None else []
        scores = list(output.scores) if output.scores is not None else []
        boxes = list(output.boxes) if output.boxes is not None else []
        lines = [t for t, s in zip(txts, scores) if s >= 0.5] if scores else list(txts)
        return "\n".join(lines), boxes, txts, scores
    # v1.4.4 fallback (legacy)
    if isinstance(output, tuple) and output[0]:
        result = output[0]
        lines = [t for (_, t, c) in result if c >= 0.5]

        record_path_warning(
            OCR_ENGINE_LEGACY_API,
            "RapidOCR v1.4.4 legacy tuple API detected — upgrade rapidocr "
            "(v3 output shape expected)",
        )
        return "\n".join(lines), [], [], []
    return "", [], [], []


def ocr_image_rapidocr(img_path, model_type="small"):
    """OCR via PP-OCRv6 (RapidOCR v3). Returns plain text joined by newlines.

    model_type: 'medium' (default, SOTA), 'small' (faster), or 'tiny' (mobile).
    Memoizes the full result (boxes+texts+scores) for reuse by the table
    specialist — see :func:`ocr_image_rapidocr_full`.
    """
    engine = _get_rapidocr(model_type=model_type)
    if engine is None:
        raise RuntimeError("rapidocr not installed. Install via: uv pip install rapidocr")
    text, boxes, txts, scores = _rapidocr_output_to_result(engine(img_path))
    _LAST_RAPIDOCR[img_path] = (boxes, txts, scores)
    return text


def ocr_image_rapidocr_full(img_path, model_type="small"):
    """OCR via RapidOCR returning the full result (boxes + texts + scores).

    Reuses the memoized page-OCR result when available (the page was already
    recognised by :func:`ocr_image_rapidocr` during the OCR cascade — re-running
    would cost ~3-5s for nothing). Boxes are page-image pixel polygons
    (4 corner points each), aligned 1:1 with ``txts`` and ``scores``. Used to
    feed RapidTable's ``ocr_results`` so it skips its own per-crop re-OCR.

    Returns ``(text, boxes, txts, scores)``; empty lists on any failure.
    """
    cached = _LAST_RAPIDOCR.get(img_path)
    if cached is not None:
        boxes, txts, scores = cached
        lines = [t for t, s in zip(txts, scores) if s >= 0.5] if scores else list(txts)
        return "\n".join(lines), boxes, txts, scores
    engine = _get_rapidocr(model_type=model_type)
    if engine is None:
        raise RuntimeError("rapidocr not installed. Install via: uv pip install rapidocr")
    text, boxes, txts, scores = _rapidocr_output_to_result(engine(img_path))
    _LAST_RAPIDOCR[img_path] = (boxes, txts, scores)
    return text, boxes, txts, scores


def _ollama_post_json(url, data, timeout=600, retries=2):
    """POST JSON to Ollama with bounded retry on transient server errors.

    Retries HTTP 5xx and URLError (daemon cold-start / model load race —
    observed as one-shot HTTP 500 while a model is still loading). After
    `retries` attempts the last error propagates: callers decide fallback
    (cascade to next model / rapidocr) instead of crashing the document.
    Explicit policy per H16: max attempts, fixed backoff, no jitter needed
    for a single-client local daemon.
    """
    import time as _time

    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < retries:
                last_err = e
                _time.sleep(1.0 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < retries:
                last_err = e
                _time.sleep(1.0 * (attempt + 1))
                continue
            raise
    raise last_err  # unreachable: loop always returns or raises


def ocr_image(img_path, task="Text Recognition:", model="rapidocr"):
    """Send image to any Ollama vision model and return recognized text.

    Model-agnostic: options (temperature, num_predict, repeat_penalty) are
    resolved from MODEL_PROFILES via get_model_profile(). Unknown models
    use DEFAULT_MODEL_PROFILE — works out of the box with any Ollama
    vision model.

    Default prompt 'Text Recognition:' is minimal and lets the model
    interpret freely. For math-heavy regions, pass MATH_OCR_PROMPT as task.

    Special model names that bypass Ollama:
      - "rapidocr" → PP-OCRv6 medium via RapidOCR (CPU, in-process, SOTA
                     non-VLM OCR. Default since 2026-08-12 — no daemon needed,
                     zero data egress, deterministic, +2-3 pts accuracy over
                     PP-OCRv6 small per arXiv 2606.13108).
      - "rapidocr:medium" / "rapidocr:small" / "rapidocr:tiny" — explicit tier.
    """
    # RapidOCR fast path — in-process, no HTTP
    if model.startswith("rapidocr"):
        # Parse explicit tier from "rapidocr:medium" / "rapidocr:small" etc.
        # Bare "rapidocr" defaults to medium (SOTA per arXiv 2606.13108).
        if ":" in model:
            mtype = model.split(":", 1)[1].strip()
        else:
            mtype = "medium"
        return ocr_image_rapidocr(img_path, model_type=mtype)

    profile = get_model_profile(model)
    task = resolve_task_prompt(model, task)

    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model,
        "prompt": task,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": profile["temperature"],
            "num_predict": profile["num_predict"],
            "num_ctx": profile.get("num_ctx", 32768),
            "repeat_penalty": profile["repeat_penalty"],
        },
    }
    if "system" in profile:
        payload["system"] = profile["system"]

    # Some models (qwen2.5vl, qwen3-vl, llama-vision) require /api/chat
    # endpoint, not /api/generate. Detect by model name prefix.
    needs_chat_endpoint = (
        any(p in model.lower() for p in ("qwen", "llama", "minicpm", "ovis", "mplug"))
        and "glm-ocr" not in model.lower()
    )

    if needs_chat_endpoint:
        # Convert to /api/chat format
        chat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": task, "images": [img_b64]}],
            "stream": False,
            "options": payload["options"],
        }
        if "system" in profile:
            chat_payload["messages"] = [
                {"role": "system", "content": profile["system"]},
                chat_payload["messages"][0],
            ]
        data = json.dumps(chat_payload).encode()
        url = OLLAMA_URL.replace("/api/generate", "/api/chat")
        result = _ollama_post_json(url, data)
        content = result.get("message", {}).get("content", "") or result.get("response", "")
        # qwen3-vl and similar reasoning models may return empty with complex
        # prompts. Fallback to simple transcription prompt if empty.
        if not content.strip():
            fallback_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Transcribe this page exactly as shown.",
                        "images": [img_b64],
                    }
                ],
                "stream": False,
                "options": payload["options"],
            }
            data2 = json.dumps(fallback_payload).encode()
            result2 = _ollama_post_json(url, data2)
            content = result2.get("message", {}).get("content", "")

            record_path_warning(
                OCR_PROMPT_FALLBACK,
                "VL model returned empty for the task prompt — re-sent with "
                "plain transcription prompt",
            )
        return normalize_vlm_output(strip_thinking_artifacts(content))

    data = json.dumps(payload).encode()
    result = _ollama_post_json(OLLAMA_URL, data)
    return normalize_vlm_output(strip_thinking_artifacts(result.get("response", "")))


def strip_thinking_artifacts(text):
    """Remove <think>...</think> blocks emitted by thinking-mode models.

    Models based on Qwen3.5 (e.g. ovisocr2) always emit a thinking block
    before the actual response, per their chat template:
        <|im_start|>assistant\n<think>\n\n</think>\n

    The block may be empty or contain reasoning text. Either way, it is
    not OCR output and must be stripped.
    """
    # Strip <think>...</think> block at start of response (non-greedy)
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Catch bare </think> (Ollama sometimes strips <think> but leaves closing tag)
    text = re.sub(r"^\s*</think>\s*", "", text)
    # CHAT-NARRATION STRIP (2026-08-31): thinking models answer math-layout
    # prompts with a planning preamble ('I need to extract the page number,
    # the text before equation (11), …') before the structured extraction.
    # High-precision conjunction: FIRST-PERSON opener + extraction-verb +
    # extraction-object in the leading block; a body sentence like
    # 'I need the data' never carries the object triple.
    text = _strip_chat_narration(text)
    # <math>-TAG UNWRAP: our own prompts tell VL models to emit
    # '<math>LaTeX</math>'; every consumer extracts $-delimited math, so
    # tagged math is invisible (2503.07260_pg6: perfect equations scored
    # zero). Unwrap here at the single model-output choke point.
    text = _unwrap_math_tags(text)
    return text


_NARRATION_RE = re.compile(
    r"(?:I\s+(?:need|will|want|am\s+going|'ll|shall)\s+to|Let\s+me|"
    r"(?:The\s+)?[Uu]ser\s+wants\s+me|First[,,.]|Based\s+on\s+the|I\s+can\s+see)",
    re.IGNORECASE,
)
_EXTRACT_VERB_RE = re.compile(
    r"\b(extract|identify|parse|recognize|locate|list|read)\b", re.IGNORECASE
)
_EXTRACT_OBJ_RE = re.compile(
    r"\b(page number|equation|section|title|text|table|figure|header|footer)\b",
    re.IGNORECASE,
)


def _strip_chat_narration(text: str) -> str:
    """Remove a leading model-planning block (see strip_thinking_artifacts).

    Only a LEADING RUN of up to 3 paragraphs is eligible, and only when a
    paragraph reads as a plan for extraction (first-person opener +
    extraction verb + extraction object). Body prose never satisfies all
    three, and a planning block is always at the front.
    """
    paras = text.split("\n\n")
    cut = 0
    for p in paras[:4]:
        if (
            _NARRATION_RE.search(p)
            and _EXTRACT_VERB_RE.search(p)
            and _EXTRACT_OBJ_RE.search(p)
        ):
            cut += 1
        else:
            break
    if cut:
        return "\n\n".join(paras[cut:]).lstrip("\n")
    return text


def _unwrap_math_tags(text: str) -> str:
    """'<math ...>LaTeX</math>' -> '$$LaTeX$$' (display) / '$LaTeX$'."""
    if "<math" not in text:
        return text
    import html as _html

    def _u(m):
        inner = _html.unescape(m.group(2)).strip()
        if not inner:
            return ""
        return f"$${inner}$$" if 'display="block"' in m.group(1) else f"${inner}$"

    return re.sub(r"<math([^>]*)>(.*?)</math>", _u, text, flags=re.DOTALL)


# Layout-HTML emitted by bbox-aware VL OCR models (chandra-ocr-2 style):
# <div data-box="x y w h" data-label="Text"><p>line<br/>line</p></div>
# Attribute name is NOT stable across runs/prompts: observed both
# "data-box" and "data-bbox". Match any data-* layout div.
_VLM_LAYOUT_DIV_RE = re.compile(r"<div[^>]*\sdata-(box|bbox|label)[^>]*>|</div>")
_VLM_BR_RE = re.compile(r"<br\s*/?>")
_VLM_P_RE = re.compile(r"</?p>")


def normalize_vlm_output(text: str) -> str:
    """Unwrap layout-HTML emitted by bbox-aware VL OCR models.

    chandra-ocr-2 and similar models return text wrapped in
    ``<div data-box="..." data-label="..."><p>...<br/>...</p></div>``
    (also spelled ``data-bbox``). The bbox metadata is extraction noise —
    it breaks exact/fuzzy substring scoring in downstream consumers
    (olmOCR-bench) and hurts readability. Unwrap to plain reading-order
    lines. No-op for models that emit plain text or markdown (guarded on
    the data-* div marker). Tables and other intentional HTML are
    preserved (only layout div/p/br tags are touched).
    """
    # CHANDRA FORMAT CORRECTIONS on ALL VL output (iter-62: they were
    # div-gated, but chandra also emits plain text — the IPA-phonetic
    # repair ('/t\varepsilonmprərəli/'→'ɛ' on ə-bearing lines) must run
    # regardless of layout-HTML presence).
    text = _chandra_format_corrections(text)
    if "<div" not in text:
        return text.strip()
    if not re.search(r"<div[^>]*\sdata-(box|bbox|label)", text):
        return text.strip()
    out = _VLM_LAYOUT_DIV_RE.sub("\n", text)
    out = _VLM_BR_RE.sub("\n", out)
    out = _VLM_P_RE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


_SUP_MAP = {c: u for c, u in zip("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")}
_SUB_MAP = {c: u for c, u in zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")}


def _chandra_format_corrections(s: str) -> str:
    import html as _html

    def _map_sub(m):
        return "".join(_SUB_MAP.get(c, c) for c in m.group(1))

    def _map_sup(m):
        return "".join(_SUP_MAP.get(c, c) for c in m.group(1))

    s = re.sub(r"<sub>(.*?)</sub>", _map_sub, s, flags=re.S | re.I)
    s = re.sub(r"<sup>(.*?)</sup>", _map_sup, s, flags=re.S | re.I)
    s = re.sub(r"</?su[bp]>", "", s, flags=re.I)
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: "$$" + _html.unescape(m.group(1)) + "$$", s, flags=re.S)
    s = re.sub(r"\$([^$\n]+)\$", lambda m: "$" + _html.unescape(m.group(1)) + "$", s)
    # Synthesized figure descriptions are chandra inventions, not page text.
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    # IPA-PHONETIC REPAIR (iter-62): chandra transcribes dictionary
    # phonetics with LaTeX ('/t\varepsilonmprərəli/') — the model's math
    # habit applied to /…/ transcriptions. On a LINE carrying IPA-only
    # vowels (ə ɑ ɜ ɔ ʊ ɪ ŋ — never math glyphs), \varepsilon is the
    # open-e ɛ the gold carries (10a: 'temporarily /temprərəli/').
    _ipa_anchor = re.compile(r"[əɑɜɔʊɪŋ]")

    def _ipa_line(ln):
        if not _ipa_anchor.search(ln):
            return ln
        # plain 'e': dictionary phonetics spell the simple vowel as 'e'
        # (gold '/temprərəli/'), unlike the model's LaTeX epsilon habit.
        return ln.replace(r"\varepsilon", "e")

    s = "\n".join(_ipa_line(ln) for ln in s.split("\n"))
    # LONE-\to REPAIR (iter-62): '\to' glued directly to an uppercase
    # word ('\toAPART') is a prose cross-reference arrow the model
    # LaTeX-ified, not math (math \to has operands/spaces). Restore '→'.
    s = re.sub(r"\\to(?=[A-Z])", "→ ", s)
    return s


__all__ = [
    "pdf_page_to_png",
    "resize_image",
    "get_model_profile",
    "_get_rapidocr",
    "ocr_image_rapidocr",
    "ocr_image",
    "strip_thinking_artifacts",
    "normalize_vlm_output",
]
