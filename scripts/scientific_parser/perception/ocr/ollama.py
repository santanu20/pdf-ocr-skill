"""Ollama-based VLM OCR backend — glm-ocr / qwen2.5vl / dots.ocr.

Wraps the existing ocr_image() function in the skill monolith, but exposed
as a clean OCRBackend for the package.

Two modes:
  - INLINE: imports skill's ocr_image (requires skill context)
  - STANDALONE: implements Ollama HTTP directly (for lib-only use)
"""

from __future__ import annotations
from typing import Optional
from .interface import (
    OCRBackend,
    OCRResult,
    CAP_TEXT,
    CAP_FORMULA,
    CAP_MULTILINGUAL,
    CAP_TABLE,
)


class OllamaBackend(OCRBackend):
    """Ollama-hosted VLM OCR (glm-ocr default)."""

    def __init__(
        self,
        model: str = "glm-ocr",
        endpoint: str = "http://127.0.0.1:11434",
        capabilities: Optional[frozenset] = None,
    ):
        self.name = model  # instance name overrides class-level "abstract"
        self.model = model
        self.endpoint = endpoint
        if capabilities:
            self.capabilities = capabilities
        else:
            # Infer from model name
            if "qwen" in model.lower():
                self.capabilities = frozenset({CAP_TEXT, CAP_MULTILINGUAL, "layout"})
            elif "glm-ocr" in model.lower():
                self.capabilities = frozenset({CAP_TEXT, CAP_FORMULA, CAP_TABLE})
            elif "dots" in model.lower():
                self.capabilities = frozenset({CAP_TEXT, CAP_MULTILINGUAL, CAP_FORMULA})
            else:
                self.capabilities = frozenset({CAP_TEXT})

    def available(self) -> bool:
        """Check Ollama is running and model is pulled."""
        import urllib.request, json

        try:
            req = urllib.request.Request(f"{self.endpoint}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            # Match by prefix (glm-ocr:latest matches "glm-ocr")
            return any(m.split(":")[0] == self.model.split(":")[0] for m in models)
        except Exception:
            return False

    def ocr(self, image_path: str, task: Optional[str] = None) -> OCRResult:
        """Send image to Ollama, return OCRResult."""
        import base64, json, urllib.request, time

        t0 = time.monotonic()
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        prompt = task or "Text Recognition:"

        # Pick endpoint based on model (qwen needs /api/chat)
        needs_chat = any(p in self.model.lower() for p in ("qwen", "llama", "minicpm"))
        if needs_chat:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 8192, "num_ctx": 8192},
            }
            url = f"{self.endpoint}/api/chat"
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 8192, "num_ctx": 16384},
            }
            url = f"{self.endpoint}/api/generate"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                result = json.loads(r.read())
            text = result.get("message", {}).get("content", "") or result.get(
                "response", ""
            )
            elapsed = time.monotonic() - t0
            return OCRResult(
                text=text,
                confidence=0.85,
                backend=self.model,
                metadata={"elapsed_s": round(elapsed, 2)},
            )
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                backend=self.model,
                metadata={"error": str(e)[:200]},
            )


__all__ = ["OllamaBackend"]
