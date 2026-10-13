"""RapidLayoutProvider — ONNX layout region detection.

Extracted from scripts/pdf_ocr.py during Phase R4 refactoring (2026-08-12).
Pure ONNX via RapidAI family. ~10-30MB model auto-downloads on first use.
"""

from __future__ import annotations

import logging
import tempfile

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state


def _pdf_ocr():
    import sys

    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


_CONSTS_LOADED = False


def _load_consts():
    global _CONSTS_LOADED
    if _CONSTS_LOADED:
        return
    p = _pdf_ocr()
    g = globals()
    own = set(g.keys())
    for k, v in vars(p).items():
        if k not in own and not k.startswith("__"):
            g[k] = v
    _CONSTS_LOADED = True


# Module-level singleton engine
_RAPIDLAYOUT_INSTANCE = None


class RapidLayoutProvider:
    """ONNX layout detection via RapidLayout — PP-DocLayoutV3 model.

    Uses pp_doc_layoutv3 (25-class SOTA model, arXiv 2503.17213) for
    English + multilingual scientific documents. Replaces the default
    pp_layout_cdla (Chinese Document Layout Analysis) which was missing
    table/equation regions on English scientific PDFs.

    PP-DocLayoutV3 classes: abstract, algorithm, aside_text, chart, content,
    display_formula, doc_title, figure_title, footer, footer_image, footnote,
    formula_number, header, header_image, image, inline_formula, number,
    paragraph_title, reference, reference_content, seal, table, text,
    vertical_text, vision_footnote.
    """

    PROVIDES = "rapid-layout"

    def __init__(self):
        self._engine = None

    def _ensure_loaded(self):
        global _RAPIDLAYOUT_INSTANCE
        if _RAPIDLAYOUT_INSTANCE is not None:
            self._engine = _RAPIDLAYOUT_INSTANCE
            return
        try:
            from rapid_layout import RapidLayout

            # PP-DocLayoutV3 — 25-class SOTA for scientific documents.
            # Auto-downloads ~100MB ONNX model on first use.
            _RAPIDLAYOUT_INSTANCE = RapidLayout(model_type="pp_doc_layoutv3")
            self._engine = _RAPIDLAYOUT_INSTANCE
            log.info("RapidLayout loaded (pp_doc_layoutv3, 25 classes)")
        except ImportError as e:
            raise RuntimeError(
                f"rapid-layout not installed. Install: uv pip install 'rapid-layout>=1.2.0'. ({e})"
            )
        except Exception as e:
            raise RuntimeError(f"RapidLayout init failed: {e}")

    def available(self) -> bool:
        try:
            import rapid_layout  # noqa: F401

            return True
        except ImportError:
            return False

    # PP-DocLayoutV3 25-class labels → our region types
    _LABEL_MAP = {
        # Text-like
        "text": "text",
        "content": "text",
        "paragraph_title": "title",
        "doc_title": "title",
        "abstract": "text",
        "aside_text": "text",
        "vertical_text": "text",
        "reference": "text",
        "reference_content": "text",
        "number": "page_number",
        # Lists
        "algorithm": "code",
        # Tables
        "table": "table",
        # Figures
        "image": "figure",
        "chart": "figure",
        "figure_title": "figure_caption",
        "vision_footnote": "figure_caption",
        "footer_image": "figure",
        "header_image": "figure",
        # Equations (KEY — these trigger RapidLaTeX specialist)
        "display_formula": "equation",
        "inline_formula": "equation",
        "formula_number": "equation",
        # Headers/footers
        "header": "header",
        "footer": "footer",
        "footnote": "footer",
        # Misc
        "seal": "figure",
    }

    def detect_regions(self, pdf_path: str, page_num: int) -> list:
        """Return list of Region dicts. page_num is 1-indexed.

        Each region: {page, bbox (x0,y0,x1,y1), type, confidence, text}
        """
        _load_consts()
        self._ensure_loaded()
        if self._engine is None:
            return []

        # Late imports from pdf_ocr (avoid circular import at module load)
        from pdf_ocr import pdf_page_to_png

        prefix = tempfile.mkdtemp(prefix="rlayout_") + f"/page_{page_num}"
        img_path = pdf_page_to_png(pdf_path, page_num, prefix)
        if not img_path:
            return []
        # No resize — RapidLayout handles any image size internally.
        # Resizing here would cause bbox coordinate mismatch with the crop
        # step in route_page_by_region (which uses the same img_path).
        try:
            result = self._engine(img_path)
            regions = []
            boxes = (
                getattr(result, "boxes", None)
                or (result.get("boxes") if isinstance(result, dict) else None)
                or []
            )
            labels = (
                getattr(result, "class_names", None)
                or getattr(result, "labels", None)
                or (result.get("labels") if isinstance(result, dict) else None)
                or []
            )
            scores = (
                getattr(result, "scores", None)
                or (result.get("scores") if isinstance(result, dict) else None)
                or []
            )
            for i, box in enumerate(boxes):
                label = (labels[i] if i < len(labels) else "unknown").lower()
                region_type = self._LABEL_MAP.get(label, "text")
                if hasattr(box, "__len__") and len(box) >= 4:
                    bbox = tuple(box[:4])
                else:
                    bbox = (0, 0, 0, 0)
                regions.append(
                    {
                        "page": page_num,
                        "bbox": bbox,
                        "type": region_type,
                        "confidence": float(scores[i]) if i < len(scores) else 1.0,
                        "text": "",
                    }
                )
            return regions
        except Exception as e:
            log.warning("RapidLayout detection failed p%s: %s", page_num, e)
            return []


_DOCLAYOUT_INSTANCE = None


class DocLayoutYOLOProvider:
    """DocLayout-YOLO — current SOTA document layout detection.

    arXiv 2410.12628: real-time, high-accuracy layout analysis via a
    coordination-cascade YOLO. A small model (not heavy ML) — fits the
    research-level-algorithm thesis. Superior to RapidLayout on complex
    multi-column / nested layouts. Dep: ``doclayout-yolo`` (pip). Graceful-
    degrade until installed; ``detect_regions`` returns [] on failure.
    """

    PROVIDES = "doclayout-yolo"

    def __init__(self):
        self._engine = None

    def _ensure_loaded(self):
        global _DOCLAYOUT_INSTANCE
        if _DOCLAYOUT_INSTANCE is not None:
            self._engine = _DOCLAYOUT_INSTANCE
            return
        try:
            from doclayout_yolo import YOLOv10  # noqa: F401

            model_path = self._resolve_model_path()
            _DOCLAYOUT_INSTANCE = YOLOv10(model_path)
            self._engine = _DOCLAYOUT_INSTANCE
            log.info("DocLayout-YOLO loaded (SOTA layout)")
        except ImportError as e:
            raise RuntimeError(
                f"doclayout-yolo not installed. Install: uv pip install doclayout-yolo. ({e})"
            )

    @staticmethod
    def _resolve_model_path() -> str:
        import os

        p = os.environ.get("DOCLAYOUT_YOLO_MODEL", "")
        if p and os.path.exists(p):
            return p
        # Default: auto-download to ~/.cache via the package's default weights.
        return "doclayout_yolo/doclayoutyolo_docstructbench_im2latex_yolo_ft.pt"

    def available(self) -> bool:
        try:
            import doclayout_yolo  # noqa: F401

            return True
        except ImportError:
            return False

    def detect_regions(self, pdf_path: str, page_num: int) -> list:
        """Detect layout regions for a page. Returns list of region dicts.

        Each region: ``{"type": <type>, "bbox": (x0,y0,x1,y1)}`` in render px.
        Returns [] on any failure (caller falls back to text-layer).
        """
        if not self.available():
            return []
        try:
            self._ensure_loaded()
            if self._engine is None:
                return []
            # Render page to PNG then run YOLO. Reuses the monolith's renderer
            # when available; falls back to PyMuPDF pixmap.
            import fitz

            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            zoom = 2.0  # ~144 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            pix.save(tmp.name)
            doc.close()
            res = self._engine.predict(tmp.name)
            regions = self._parse_yolo_result(res, page.rect.width, page.rect.height, zoom)
            try:
                import os

                os.unlink(tmp.name)
            except OSError:
                pass
            return regions
        except Exception as e:
            log.warning("DocLayout-YOLO detect failed: %s", e)
            return []

    @staticmethod
    def _parse_yolo_result(res, pdf_w_pt, pdf_h_pt, zoom):
        """Convert YOLO boxes to region dicts in render-px bbox."""
        out = []
        try:
            boxes = getattr(res[0], "boxes", None) if res else None
            if boxes is None:
                return out
            names = getattr(res[0], "names", {}) or {}
            for b in boxes:
                xyxy = getattr(b, "xyxy", None)
                cls = getattr(b, "cls", None)
                if xyxy is None or cls is None:
                    continue
                import numpy as np

                coords = np.asarray(xyxy).flatten().tolist()
                if len(coords) != 4:
                    continue
                # YOLO ran on the rendered image (zoomed) — coords already in
                # render px (pdf_pt * zoom).
                x0, y0, x1, y1 = coords
                label = (
                    names.get(int(cls.flatten().tolist()[0]), "text")
                    if hasattr(cls, "flatten")
                    else "text"
                )
                out.append({"type": str(label), "bbox": (x0, y0, x1, y1)})
        except Exception as e:
            log.debug("_parse_yolo_result: Exception degraded: %s", e)
        return out

from scientific_parser.core.path_warnings import (
    record_path_warning,
    LAYOUT_BACKEND_FALLBACK,
)

def get_layout_detector(backend: str = "default"):
    """Factory: layout detector by backend name. All CPU/ONNX — no torch.

    backends:
      - "default": best-available CPU layout — DocLayout-YOLO-ONNX (SOTA,
        arXiv 2410.12628: faster + better table/figure recall) with automatic
        fallback to RapidLayout if its weights are unavailable (offline /
        first-run before download).
      - "rapid-layout": RapidLayout ONNX (pp_doc_layoutv3, bundled in package).
      - "doclayout-yolo-onnx": DocLayout-YOLO served as pre-exported ONNX via
        onnxruntime — CPU, NO torch. Loads weights from HuggingFace.
      - "doclayout-yolo": torch-based DocLayout-YOLO (GPU extra, graceful-degrade)
      - "off" / other: returns None
    """
    if backend == "default":
        # SOTA first; fall back to the bundled RapidLayout if the DocLayout-YOLO
        # weights can't be obtained (offline / download failure).
        try:
            p = DocLayoutYOLOOnnxProvider()
            if p.available():
                return p
        except Exception as e:
            log.debug("DocLayout-YOLO-ONNX unavailable, trying RapidLayout: %s", e)
        # Unreachable weights OR unavailable() — both mean the SOTA backend
        # is not usable and the page set falls back to RapidLayout.
        record_path_warning(
            LAYOUT_BACKEND_FALLBACK,
            "DocLayout-YOLO-ONNX unavailable — falling back to RapidLayout",
        )
        try:
            return RapidLayoutProvider()
        except Exception as e:
            log.warning("RapidLayout backend unavailable: %s", e)
            record_path_warning(
                LAYOUT_BACKEND_FALLBACK,
                "RapidLayout ALSO unavailable — no layout detector active",
            )
            return None
    if backend == "rapid-layout":
        try:
            return RapidLayoutProvider()
        except Exception as e:
            log.warning("RapidLayout backend unavailable: %s", e)
            return None
    if backend == "doclayout-yolo-onnx":
        try:
            return DocLayoutYOLOOnnxProvider()
        except Exception as e:
            log.warning("DocLayout-YOLO-ONNX backend unavailable: %s", e)
            return None
    if backend == "doclayout-yolo":
        try:
            return DocLayoutYOLOProvider()
        except Exception as e:
            log.warning("DocLayout-YOLO backend unavailable: %s", e)
            return None
    return None


_DOCLAYOUT_ONNX_SESSION = None
_DOCLAYOUT_CACHE: dict = {}  # input_size / input_name / names (instance-reuse)
_DOCLAYOUT_ONNX_REPO = "welcomyou/doclayout-yolo-onnx-dynamic"
# Pinned revision (audit M5): unpinned snapshot_download silently changes
# behavior when the repo owner pushes new weights. Bump deliberately after
# re-benching. sha = main @ 2026-05-07 (lastModified), fetched 2026-08-16.
_DOCLAYOUT_ONNX_REVISION = "997d2c6f44c742bc6e3ef94c795a62e57f2228b6"
# Weights + external data live in a subfolder; snapshot_download fetches all
# three files (.onnx + .onnx.data + .names.json) together — single-file
# hf_hub_download misses the .onnx.data external-weights blob → load fails.
_DOCLAYOUT_ONNX_SUBDIR = "doclayout_yolo_onnx_dynamic"
_DOCLAYOUT_ONNX_FILE = "doclayout_yolo_docstructbench_imgsz1024_dynamic.onnx"
_DOCLAYOUT_ONNX_NAMES = "doclayout_yolo_docstructbench_imgsz1024_dynamic.names.json"
# DocLayout-YOLO labels → dispatcher-expected region types. "plain text" and
# "isolate_formula" would otherwise miss the text/equation branches.
_DOCLAYOUT_LABEL_MAP = {
    "plain text": "text",
    "isolate_formula": "equation",
}


class DocLayoutYOLOOnnxProvider:
    """DocLayout-YOLO (arXiv 2410.12628) — SOTA layout, **CPU, NO torch**.

    Runs the pre-exported ONNX via :mod:`onnxruntime` (same runtime as the
    RapidAI family). Weights download once from HuggingFace
    (``welcomyou/doclayout-yolo-onnx-dynamic``) to ``~/.cache/huggingface``.

    YOLOv10 contract: output is already NMS-free (boxes + conf + class per
    detection). Input contract is read at runtime from the session
    (``get_inputs()[0]``) so no hardcoded input shape. Falls back to [] on
    any failure — the caller (route_page_by_region) then uses RapidLayout or
    the text-layer heuristic, so a decode miss never breaks the pipeline.
    """

    PROVIDES = "doclayout-yolo-onnx"

    def __init__(self):
        self._session = None
        self._input_name = None
        self._input_size = None  # (h, w) the model expects, read at load
        self._names = None  # class-index → label, loaded from .names.json

    def _ensure_loaded(self):
        global _DOCLAYOUT_ONNX_SESSION
        if _DOCLAYOUT_ONNX_SESSION is not None:
            # Copy the FULL cached state, not just the session — a fresh
            # provider instance taking this early-return used to keep
            # _input_size/_names = None → "cannot unpack NoneType" on every
            # detect after the first (measured: 90 failures in bench run8).
            self._session = _DOCLAYOUT_ONNX_SESSION
            self._input_size = _DOCLAYOUT_CACHE["input_size"]
            self._input_name = _DOCLAYOUT_CACHE["input_name"]
            self._names = _DOCLAYOUT_CACHE["names"]
            return
        try:
            import json
            import os

            import onnxruntime as ort
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                f"onnxruntime + huggingface-hub required for DocLayout-YOLO-ONNX. ({e})"
            )
        # snapshot_download fetches the whole subfolder so the .onnx and its
        # external .onnx.data weights land co-located (single-file download
        # misses the data blob → onnxruntime fails to restore weights).
        local_dir = snapshot_download(
            repo_id=_DOCLAYOUT_ONNX_REPO,
            allow_patterns=[f"{_DOCLAYOUT_ONNX_SUBDIR}/*"],
            revision=_DOCLAYOUT_ONNX_REVISION,
        )
        sub = os.path.join(local_dir, _DOCLAYOUT_ONNX_SUBDIR)
        model_path = os.path.join(sub, _DOCLAYOUT_ONNX_FILE)
        names_path = os.path.join(sub, _DOCLAYOUT_ONNX_NAMES)
        with open(names_path) as f:
            raw = json.load(f)
        self._names = [raw[str(i)] for i in range(len(raw))]
        _DOCLAYOUT_ONNX_SESSION = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._session = _DOCLAYOUT_ONNX_SESSION
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape  # e.g. [1, 3, 1024, 1024] or [1,3,H,W] dynamic
        # Prefer concrete dims; fall back to 1024 if dynamic.
        h = shape[2] if len(shape) > 2 and isinstance(shape[2], int) and shape[2] > 0 else 1024
        w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) and shape[3] > 0 else 1024
        self._input_size = (int(h), int(w))
        _DOCLAYOUT_CACHE["input_size"] = self._input_size
        _DOCLAYOUT_CACHE["input_name"] = self._input_name
        _DOCLAYOUT_CACHE["names"] = self._names
        log.info(
            "DocLayout-YOLO-ONNX loaded (CPU, input %dx%d, %d classes)", h, w, len(self._names)
        )

    def available(self) -> bool:
        try:
            import huggingface_hub  # noqa: F401
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False

    def detect_regions(self, pdf_path: str, page_num: int) -> list:
        """Detect layout regions. Returns list of ``{"type","bbox"}`` (render px).

        Self-contained: renders the page via PyMuPDF, letterbox-resizes to the
        model input, runs ONNX inference, decodes YOLOv10 NMS-free output.
        """
        if not self.available():
            return []
        try:
            self._ensure_loaded()
            if self._session is None:
                return []
            import fitz

            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            zoom = 2.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            pix.save(tmp.name)
            doc.close()
            from PIL import Image

            img = Image.open(tmp.name).convert("RGB")
            regions = self._infer(img, zoom)
            try:
                import os

                os.unlink(tmp.name)
            except OSError:
                pass
            return regions
        except Exception as e:
            log.warning("DocLayout-YOLO-ONNX detect failed: %s", e)
            return []

    def _infer(self, pil_img, zoom: float) -> list:
        """Letterbox + ONNX run + YOLOv10 decode. Returns region dicts."""
        import numpy as np
        from PIL import Image

        ih, iw = self._input_size
        # Letterbox resize preserving aspect ratio.
        src = pil_img
        sw, sh = src.size
        scale = min(iw / sw, ih / sh)
        nw, nh = int(sw * scale), int(sh * scale)
        resized = src.resize((nw, nh))
        canvas = Image.new("RGB", (iw, ih), (114, 114, 114))
        canvas.paste(resized, (0, 0))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC
        arr = arr.transpose(2, 0, 1)[None]  # 1CHW
        out = self._session.run(None, {self._input_name: arr})
        return self._decode_yolov10(out, scale, zoom, self._names)

    @staticmethod
    def _decode_yolov10(outputs, scale: float, zoom: float, names: list) -> list:
        """Decode YOLOv10 NMS-free output → region dicts (render px).

        YOLOv10 returns ``[1, N, >=6]`` with rows ``(x0, y0, x1, y1, conf,
        class_id)`` already filtered. scale maps model px → rendered-image px
        (undo letterbox); zoom maps rendered px → the page-raster coords the
        dispatcher uses. ``names`` is the class-index→label list from the
        model's .names.json. Labels are normalized via _DOCLAYOUT_LABEL_MAP
        so the dispatcher sees its expected types (text/equation/table).
        """
        import numpy as np

        try:
            preds = outputs[0]
            arr = np.asarray(preds)
            if arr.ndim == 3:
                arr = arr[0]
            regions = []
            for row in arr:
                row = np.asarray(row).flatten()
                if row.size < 6:
                    continue
                x0, y0, x1, y1, conf = row[0], row[1], row[2], row[3], float(row[4])
                if conf < 0.25:
                    continue
                cls = int(row[5]) if row.size > 5 else 0
                # Undo letterbox (scale) — model coords are in input px.
                rx0, ry0, rx1, ry1 = (x0 / scale, y0 / scale, x1 / scale, y1 / scale)
                label = names[cls] if 0 <= cls < len(names) else "text"
                label = _DOCLAYOUT_LABEL_MAP.get(label, label)
                regions.append(
                    {"type": label, "bbox": (float(rx0), float(ry0), float(rx1), float(ry1))}
                )
            return regions
        except Exception as e:
            log.warning("DocLayout-YOLO-ONNX decode failed: %s", e)
            return []


__all__ = [
    "RapidLayoutProvider",
    "DocLayoutYOLOProvider",
    "DocLayoutYOLOOnnxProvider",
    "get_layout_detector",
]
