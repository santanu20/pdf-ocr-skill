"""RapidLaTeXProvider — ONNX formula → LaTeX via RapidLaTeXOCR.

Extracted from scripts/pdf_ocr.py during Phase R4 refactoring (2026-08-12).
Pure ONNX via RapidAI family — no torch/paddlepaddle. Lazy-loaded.
~80MB model auto-downloads on first use to ~/.cache/.

Backwards compat: pdf_ocr.py re-exports all names below.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Module-level singleton engine — survives across provider instances.
# Set on first successful _ensure_loaded(); reused on subsequent calls.
_RAPIDLATEX_INSTANCE = None
_NP2_PATCHED = False


def _patch_rapidlatex_numpy2() -> None:
    """Fix rapid_latex_ocr's numpy>=2 incompatibility at runtime.

    rapid_latex_ocr <= 0.0.9 does ``int(np.argmax(arr, axis=-1))`` in its
    image resizer. On numpy>=2 this raises ``TypeError: only 0-dimensional
    arrays can be converted to Python scalars`` (argmax along the last axis
    returns a 1-d array). The fix is ``.item()``.

    numpy<2 can't be pinned because the torch specialists (decimer/unimernet)
    require numpy>=2.3 — so the two coexist only by patching rapid_latex_ocr
    in place. Idempotent: re-checks + re-applies on every load (survives
    reinstalls). Edits the installed source file + reloads the module.
    """
    global _NP2_PATCHED
    if _NP2_PATCHED:
        return
    import importlib
    import inspect

    try:
        import rapid_latex_ocr.main as rlm

        src_file = inspect.getsourcefile(rlm.LatexOCR.loop_image_resizer)
        with open(src_file) as f:
            src = f.read()
        bad = "int(np.argmax(resizer_res, axis=-1))"
        good = "int(np.argmax(resizer_res, axis=-1).item())"
        if bad in src and good not in src:
            with open(src_file, "w") as f:
                f.write(src.replace(bad, good))
            importlib.reload(rlm)
            log.info("Patched rapid_latex_ocr for numpy>=2 (.item() fix)")
    except Exception as e:
        log.debug("rapid_latex_ocr numpy2 patch skipped: %s", e)
    finally:
        _NP2_PATCHED = True


class RapidLaTeXProvider:
    """ONNX formula → LaTeX via RapidLaTeXOCR.

    Lazy-loaded. Auto-activates when specialist deps installed (CORE per
    pyproject.toml). ~80MB model auto-downloads on first use to ~/.cache/.
    """

    PROVIDES = "rapidlatex"

    def __init__(self):
        self._engine = None

    def _ensure_loaded(self):
        global _RAPIDLATEX_INSTANCE
        if _RAPIDLATEX_INSTANCE is not None:
            self._engine = _RAPIDLATEX_INSTANCE
            return
        _patch_rapidlatex_numpy2()  # fix numpy>=2 incompat BEFORE constructing
        try:
            from rapid_latex_ocr import LatexOCR

            _RAPIDLATEX_INSTANCE = LatexOCR()
            self._engine = _RAPIDLATEX_INSTANCE
            log.info("RapidLaTeXOCR loaded (ONNX, ~80MB)")
        except ImportError as e:
            raise RuntimeError(
                f"rapid-latex-ocr not installed. Install: uv pip install rapid-latex-ocr. ({e})"
            )
        except Exception as e:
            raise RuntimeError(f"RapidLaTeXOCR init failed: {e}")

    def available(self) -> bool:
        try:
            import rapid_latex_ocr  # noqa: F401

            return True
        except ImportError:
            return False

    def recognize(self, img_path_or_array) -> tuple[str, float]:
        """Recognize formula in image, return (latex, confidence).

        Passes a PATH to the engine (LatexOCR.load_img handles str/Path via
        cv2.imread — reliable on OpenCV 4.x; the prior "avoid imread / convert
        to ndarray" workaround caused cv2 to receive an empty array on equation
        crops, breaking every equation). Arrays/PIL images are materialized to
        a temp PNG first. Small images (<100px height) are upscaled first — the
        resizer otherwise raises "image resizer meets error".
        """
        self._ensure_loaded()
        if self._engine is None:
            return "", 0.0
        try:
            import os
            import tempfile
            from pathlib import Path

            import numpy as np
            from PIL import Image

            path_like = isinstance(img_path_or_array, (str, Path)) or hasattr(
                img_path_or_array, "__fspath__"
            )
            if path_like:
                src_path = str(img_path_or_array)
                pil_img = Image.open(src_path).convert("RGB")
            elif isinstance(img_path_or_array, np.ndarray):
                pil_img = Image.fromarray(img_path_or_array).convert("RGB")
            elif hasattr(img_path_or_array, "mode"):
                pil_img = img_path_or_array.convert("RGB")
            else:
                pil_img = img_path_or_array  # let load_img handle it

            # Upscale short images so the resizer doesn't choke.
            w, h = pil_img.size
            if h < 100 and h > 0:
                scale = 100.0 / h
                pil_img = pil_img.resize((int(w * scale), 100), Image.Resampling.LANCZOS)

            # Always materialize to a temp PNG and pass the PATH to the engine —
            # the ndarray path is unreliable across rapid_latex_ocr versions.
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            pil_img.save(tmp.name, format="PNG")
            try:
                latex, conf = self._engine(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            return (latex or "").strip(), float(conf)
        except Exception as e:
            log.warning("RapidLaTeX recognition failed: %s", e)
            return "", 0.0


def get_formula_recognizer(backend: str = "default"):
    """Factory: formula recognizer by backend name.

    backends:
      - "default" / "rapidlatex": RapidLaTeX ONNX (auto-activates if installed)
      - "unimernet": UniMERNet (SOTA equation recognition, heavier)
      - "off" / other: returns None (specialist disabled)
    """
    if backend in ("rapidlatex", "default"):
        try:
            return RapidLaTeXProvider()
        except Exception as e:
            log.warning("RapidLaTeX backend unavailable: %s", e)
            return None
    if backend == "unimernet":
        try:
            return UniMERNetProvider()
        except Exception as e:
            log.warning("UniMERNet backend unavailable: %s", e)
            return None
    return None


# TexoProvider removed 2026-08-16: dep never published, zero call sites
# (audit M11 — speculative dead surface).


_UNIMERNET_INSTANCE = None


class UniMERNetProvider:
    """UniMERNet — SOTA equation recognition (heavier, highest accuracy).

    Dep: ``unimernet`` (pip). Handles complex/handwritten math better than
    RapidLaTeX-class CNN. Planned Phase 3 model. Graceful-degrade until installed.
    """

    PROVIDES = "unimernet"

    def __init__(self):
        self._engine = None

    def _ensure_loaded(self):
        global _UNIMERNET_INSTANCE
        if _UNIMERNET_INSTANCE is not None:
            self._engine = _UNIMERNET_INSTANCE
            return
        try:
            import unimernet
            from unimernet.models import get_model  # noqa: F401

            _UNIMERNET_INSTANCE = unimernet
            self._engine = unimernet
            log.info("UniMERNet loaded (SOTA equations)")
        except ImportError as e:
            raise RuntimeError(f"unimernet not installed. Install: uv pip install unimernet. ({e})")

    def available(self) -> bool:
        try:
            import unimernet  # noqa: F401

            return True
        except ImportError:
            return False

    def recognize(self, img_path_or_array) -> tuple[str, float]:
        self._ensure_loaded()
        if self._engine is None:
            return "", 0.0
        try:
            res = self._engine.recognize(str(img_path_or_array))
            latex = (res.get("latex") or "").strip() if isinstance(res, dict) else str(res)
            conf = float(res.get("confidence", 0.85)) if isinstance(res, dict) else 0.85
            return latex, conf
        except Exception as e:
            log.warning("UniMERNet recognize failed: %s", e)
            return "", 0.0


__all__ = [
    "RapidLaTeXProvider",
    "UniMERNetProvider",
    "get_formula_recognizer",
]
