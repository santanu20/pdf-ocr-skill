"""Model artifact manifest — self-describing model format.

Per architecture proposal item #36 (2026-08-12):
  "Model artifact should be self-describing. Instead of `model.onnx`,
   have a manifest.json that bundles model + dict + preprocessing +
   checksums. Then your runtime doesn't care whether the model came
   from Paddle, RapidOCR or Hugging Face."

Each model lives in a directory:
    models/
      <model_name>/
        manifest.json     ← this file's schema
        model.onnx (or .pth, .gguf, .safetensors)
        dictionary.txt    ← for CTC-based recognizers
        preprocessing.json
        postprocessing.json
        checksums.sha256

Skill behavior:
  - Models registered in MODEL_PROFILES get a manifest entry
  - On first use: verify sha256 matches manifest (anti-tampering per §5 H7)
  - On mismatch: refuse to load (fail-fast per §5 H1)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelManifest:
    """Self-describing model artifact manifest."""

    name: str  # e.g. "PP-OCRv6_small_rec"
    task: str  # "detection" | "recognition" | "classification" | "ocr_full"
    format: str  # "onnx" | "torch" | "gguf" | "safetensors" | "ollama"
    version: str  # e.g. "v6.0", "v3.9.2"
    source: str = ""  # upstream URL or ollama tag
    license: str = ""  # SPDX ID (Apache-2.0, MIT, etc.)
    capabilities: list = field(default_factory=list)  # from OCREngine.CAPABILITIES
    input_shape: list = field(default_factory=list)  # e.g. [1, 3, 48, 320]
    dictionary_path: Optional[str] = None  # rel path to char dict
    preprocessing: dict = field(default_factory=dict)
    postprocessing: dict = field(default_factory=dict)
    checksum_sha256: Optional[str] = None  # for primary model file
    notes: str = ""

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: Path) -> "ModelManifest":
        data = json.loads(path.read_text())
        return cls(**data)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute sha256 of a file (1MB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(model_dir: Path) -> tuple[bool, str]:
    """Load manifest, verify primary model file sha256 matches.

    Returns (ok, message). Per §5 H1: fail-fast on mismatch.
    """
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return False, f"no manifest.json in {model_dir}"
    try:
        manifest = ModelManifest.from_json(manifest_path)
    except Exception as e:
        return False, f"manifest parse error: {e}"

    if not manifest.checksum_sha256:
        return True, "manifest present, no checksum declared (skip verify)"

    # Find primary model file
    candidates = [
        model_dir / "model.onnx",
        model_dir / "model.safetensors",
        model_dir / "model.pth",
        model_dir / "model.gguf",
    ]
    model_file = next((p for p in candidates if p.exists()), None)
    if model_file is None:
        return False, f"no model file found in {model_dir}"

    actual = sha256_file(model_file)
    if actual.lower() != manifest.checksum_sha256.lower():
        return False, (
            f"checksum mismatch for {model_file.name}: "
            f"expected {manifest.checksum_sha256[:16]}…, got {actual[:16]}…"
        )
    return True, f"verified {model_file.name} ({manifest.name} {manifest.version})"


# === Built-in model manifests (shipped with skill) ========================= #
# These are the known-good configurations. Users can override by adding
# their own manifest.json next to a model directory.

BUILTIN_MANIFESTS = {
    "glm-ocr": ModelManifest(
        name="glm-ocr",
        task="ocr_full",
        format="ollama",
        version="ollama-pull",
        source="ollama pull glm-ocr",
        license="proprietary-Ollama-LLama2-style",
        capabilities=["text", "formula", "table"],
        input_shape=[1, 3, 1024, 1024],  # adaptive
        preprocessing={"normalize": False, "max_dim": 4096},
        postprocessing={"strip_thinking": True, "math_prompt_required": True},
        checksum_sha256=None,  # Ollama manages integrity
        notes="Default Ollama vision model. HTTP API at /api/generate.",
    ),
    "qwen2.5vl:3b": ModelManifest(
        name="qwen2.5vl:3b",
        task="ocr_full",
        format="ollama",
        version="ollama-pull",
        source="ollama pull qwen2.5vl:3b",
        license="Apache-2.0 (Qwen-LICENSE)",
        capabilities=["text", "multilingual", "layout"],
        input_shape=[1, 3, 1024, 1024],
        preprocessing={"normalize": False, "max_dim": 4096},
        postprocessing={"endpoint": "/api/chat", "strip_thinking": False},
        checksum_sha256=None,
        notes="2026 OCR winner for plain text (nullmirror benchmark).",
    ),
    "rapidocr": ModelManifest(
        name="PP-OCRv6_small",
        task="ocr_full",
        format="onnx",
        version="v3.9.2",
        source="uv pip install rapidocr  (bundled ONNX)",
        license="Apache-2.0",
        capabilities=["text", "multilingual"],
        input_shape=[1, 3, 48, 320],  # rec model
        preprocessing={"normalize": [(0.5,) * 3, (0.5,) * 3]},
        postprocessing={"ctc_decode": True},
        checksum_sha256=None,  # RapidOCR manages integrity
        notes=(
            "PP-OCRv6 small detector + recognizer. arXiv 2606.13108. "
            "12s/pg on CPU. Speed tier; qwen2.5vl:3b beats on accuracy."
        ),
    ),
}


def get_manifest(model_name: str) -> Optional[ModelManifest]:
    """Return manifest for known model. None if unknown."""
    return BUILTIN_MANIFESTS.get(model_name)


def list_known_models() -> list[str]:
    """Names of all models with built-in manifests."""
    return sorted(BUILTIN_MANIFESTS.keys())


__all__ = [
    "ModelManifest",
    "sha256_file",
    "verify_manifest",
    "BUILTIN_MANIFESTS",
    "get_manifest",
    "list_known_models",
]
