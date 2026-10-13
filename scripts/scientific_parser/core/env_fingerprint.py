"""Environment fingerprint for extraction-cache keys (P0.3, ROADMAP-SOTA).

Problem (era-delta class, run-51): the cache key encoded engine versions
only. Raw extraction output also depends on the dependency stack
(rapidocr/onnxruntime behavior, poppler's pdftotext) and, for OCR-routed
pages, the exact local ollama model blobs. Two checkouts sharing a version
string but differing in uv.lock or a re-pulled model would replay each
other's cached raw output — silent cross-environment staleness.

Component set (each memoized per process):
  - uv.lock sha256 (dependency pin set for the whole engine)
  - python major.minor (interpreter semantics)
  - pdftotext version line (poppler release, fast-path text source)
  - ollama manifest digests for the models in the requested chain,
    resolved against a disk-persisted registry snapshot so a transient
    ollama outage never flaps the key (last-known digests are reused;
    a first-ever run with ollama down records "absent", which can only
    cause later misses, never stale hits)

Fingerprint scheme is versioned ("env1") so the component set can evolve
without colliding with old values.

This module must NEVER raise into the cache path: any unexpected error
degrades to a constant "env1-error" component (worst case: no env
discrimination — same as the pre-P0.3 world).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import urllib.request
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

log = logging.getLogger(__name__)

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
SCHEME = "env1"


def _repo_root() -> str | None:
    # .../scripts/scientific_parser/core/env_fingerprint.py -> repo root
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    return root if os.path.isdir(root) else None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=4)
def _uv_lock_tag(repo_root: str | None = None) -> str:
    root = repo_root or _repo_root()
    if not root:
        return "no-repo"
    p = os.path.join(root, "uv.lock")
    if not os.path.exists(p):
        return "no-uvlock"
    try:
        return _sha256_file(p)[:16]
    except OSError:
        return "uvlock-unreadable"


def _python_tag() -> str:
    return f"py{sys.version_info.major}.{sys.version_info.minor}"


@lru_cache(maxsize=1)
def _pdftotext_tag() -> str:
    try:
        out = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, timeout=10)
        text = (out.stderr or "") + (out.stdout or "")
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("pdftotext version"):
                return line
        return "pdftotext-unknown"
    except Exception:
        return "pdftotext-missing"


def _fetch_registry(timeout_s: float = 3.0) -> dict[str, str] | None:
    """Live {model_name: digest} from the local ollama, or None if unreachable."""
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.load(resp)
        return {
            str(m.get("name")): str(m.get("digest") or "nodigest")
            for m in data.get("models", [])
            if m.get("name")
        }
    except Exception:
        return None


def _registry_path() -> str:
    try:
        import shared  # scripts/ on sys.path in engine and test runs

        cache_dir = shared.DEFAULT_CACHE_DIR
    except Exception:
        cache_dir = os.environ.get(
            "PDF_OCR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "pdf-ocr")
        )
    return os.path.join(cache_dir, "model_registry.json")


def _load_persisted_registry(path: str) -> dict[str, str]:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if isinstance(doc, dict) and isinstance(doc.get("models"), dict):
            return {str(k): str(v) for k, v in doc["models"].items()}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _persist_registry(path: str, registry: Mapping[str, str]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"scheme": SCHEME, "models": dict(registry)}, f)
        os.replace(tmp, path)
    except OSError as e:
        log.warning("model registry persist failed (non-fatal): %s", e)


def _digest_for(registry: Mapping[str, str], model: str) -> str:
    if model in registry:
        return registry[model][:16]
    base = model.split(":")[0]
    for name, digest in sorted(registry.items()):
        if name.split(":")[0] == base:
            return digest[:16]
    return "absent"


def _model_digests(models: Sequence[str], registry_path: str | None = None) -> str:
    """Stable per-model digest string. Refreshes the persisted snapshot when
    ollama answers; reuses the last-known snapshot when it doesn't."""
    wanted = sorted(models or ())
    if not wanted:
        return "nomodels"
    live = _fetch_registry()
    path = registry_path or _registry_path()
    if live is not None:
        _persist_registry(path, live)
        registry = live
    else:
        registry = _load_persisted_registry(path)
    return ";".join(f"{m}={_digest_for(registry, m)}" for m in wanted)


def compute_fingerprint(
    models: Iterable[str],
    *,
    uv_lock: str,
    python_tag: str,
    pdftotext_tag: str,
    model_digests: str,
) -> str:
    """Pure core: combine component tags into the fingerprint string."""
    blob = "\n".join([uv_lock, python_tag, pdftotext_tag, model_digests])
    return f"{SCHEME}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"


@lru_cache(maxsize=64)
def _env_fingerprint_cached(models_key: tuple[str, ...]) -> str:
    return compute_fingerprint(
        models_key,
        uv_lock=_uv_lock_tag(),
        python_tag=_python_tag(),
        pdftotext_tag=_pdftotext_tag(),
        model_digests=_model_digests(models_key),
    )


def env_fingerprint(models: Iterable[str] | None = None) -> str:
    """Memoized fingerprint for the model chain (sorted, order-insensitive).

    Never raises: unexpected errors degrade to the constant
    "env1-error" component (see module docstring).
    """
    try:
        key = tuple(sorted(models or ()))
        return _env_fingerprint_cached(key)
    except Exception as e:  # cache is throughput, never correctness (H1)
        log.warning("env_fingerprint degraded (non-fatal): %s", e)
        return f"{SCHEME}-error"


__all__ = [
    "env_fingerprint",
    "compute_fingerprint",
    "_model_digests",
    "_fetch_registry",
    "_registry_path",
    "_uv_lock_tag",
]
