"""Chemical-structure recognition providers (reviewer #5 / specialized models).

Optical Chemical Structure Recognition (OCSR): convert an image of a drawn
molecule into a SMILES string (or molecular graph). This fills the largest
specialist gap in the parser — previously only textual formulas like ``H2O``
were parsed; molecule IMAGES got nothing.

Two adapters, both graceful-degrade (``.available()`` checks the dep; the
dispatcher unconditionally TRIES them, so they are utilized the moment the
user installs the dep):

  - **DECIMER** (DECIMER.ai): image-transformer OCSR → SMILES.
    Open platform, Nature-published (Raján et al.). Dep: ``decimer``.
  - **MolScribe**: image → molecular graph (robust, end-to-end).
    Dep: ``molscribe`` (pip). Produces SMILES + atom/bond graph.

Both follow the RapidLaTeXProvider pattern: lazy-loaded, module-level singleton
engine, ``recognize(img) -> (smiles, confidence)``.

Install to activate:
    uv pip install decimer        # or: molscribe
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_DECIMER_INSTANCE = None
_MOLSCRIBE_INSTANCE = None


class DecimerProvider:
    """DECIMER OCSR — chemical structure image → SMILES."""

    PROVIDES = "decimer"

    def __init__(self):
        self._engine = None

    def _ensure_loaded(self):
        global _DECIMER_INSTANCE
        if _DECIMER_INSTANCE is not None:
            self._engine = _DECIMER_INSTANCE
            return
        try:
            import decimer
            from decimer import DECIMER_V2  # noqa: F401

            _DECIMER_INSTANCE = decimer
            self._engine = decimer
            log.info("DECIMER OCSR loaded")
        except ImportError as e:
            raise RuntimeError(f"decimer not installed. Install: uv pip install decimer. ({e})")

    def available(self) -> bool:
        try:
            import decimer  # noqa: F401

            return True
        except ImportError:
            return False

    def recognize(self, img_path_or_array) -> tuple[str, float]:
        """Recognize a chemical structure, return (smiles, confidence).

        confidence is heuristic (DECIMER does not emit one): 0.8 if non-empty
        SMILES with a carbon atom, else 0.3.
        """
        self._ensure_loaded()
        if self._engine is None:
            return "", 0.0
        try:
            from DECIMER import decimer_interface  # DECIMER import path varies

            smiles = decimer_interface.decode_image(str(img_path_or_array))
            smiles = (smiles or "").strip()
            conf = 0.8 if smiles and ("C" in smiles or "c" in smiles) else 0.3
            return smiles, conf
        except Exception as e:
            log.warning("DECIMER recognize failed: %s", e)
            return "", 0.0


class MolScribeProvider:
    """MolScribe OCSR — chemical structure image → SMILES + graph."""

    PROVIDES = "molscribe"

    def __init__(self):
        self._engine = None

    def _ensure_loaded(self):
        global _MOLSCRIBE_INSTANCE
        if _MOLSCRIBE_INSTANCE is not None:
            self._engine = _MOLSCRIBE_INSTANCE
            return
        try:
            import molscribe
            from molscribe import MolScribe  # noqa: F401

            _MOLSCRIBE_INSTANCE = molscribe
            self._engine = molscribe
            log.info("MolScribe OCSR loaded")
        except ImportError as e:
            raise RuntimeError(f"molscribe not installed. Install: uv pip install molscribe. ({e})")

    def available(self) -> bool:
        try:
            import molscribe  # noqa: F401

            return True
        except ImportError:
            return False

    def recognize(self, img_path_or_array) -> tuple[str, float]:
        """Recognize a chemical structure, return (smiles, confidence)."""
        self._ensure_loaded()
        if self._engine is None:
            return "", 0.0
        try:
            from molscribe import MolScribe
            from molscribe.inference import Inference

            model = Inference(MolScribe)
            output = model.predict_image(str(img_path_or_array))
            smiles = (
                (output.get("smiles") or "").strip() if isinstance(output, dict) else str(output)
            )
            conf = float(output.get("score", 0.85)) if isinstance(output, dict) else 0.85
            return smiles, conf
        except Exception as e:
            log.warning("MolScribe recognize failed: %s", e)
            return "", 0.0


def get_chemical_recognizer(backend: str = "default"):
    """Factory: chemical-structure recognizer by backend name.

    backends:
      - "default" / "decimer": DECIMER OCSR (auto-activates if installed)
      - "molscribe": MolScribe OCSR
      - "off" / other: returns None (no OCSR; structure images untreated)
    """
    if backend in ("decimer", "default"):
        try:
            return DecimerProvider()
        except Exception as e:
            log.warning("DECIMER backend unavailable: %s", e)
            return None
    if backend == "molscribe":
        try:
            return MolScribeProvider()
        except Exception as e:
            log.warning("MolScribe backend unavailable: %s", e)
            return None
    return None


__all__ = ["DecimerProvider", "MolScribeProvider", "get_chemical_recognizer"]
