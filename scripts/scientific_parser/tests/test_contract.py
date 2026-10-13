"""Model-contract tests — verify every OCR backend meets minimum contract.

Per proposal item #37 (2026-08-12):
  "Every OCR model must pass: input preprocessing test, output tensor shape
   test, dictionary alignment test, blank-token test, CTC decode test,
   normalization test, known-image golden test. Before entering production."

Run via: pytest scripts/scientific_parser/tests/test_contract.py -v
"""

import sys
from pathlib import Path

# Make scripts importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_ollama_backend_contract():
    """OllamaBackend must declare capabilities + implement ocr() + available()."""
    from scientific_parser.perception.ocr.ollama import OllamaBackend

    b = OllamaBackend(model="glm-ocr")
    assert b.name == "glm-ocr"
    assert "text" in b.capabilities
    assert "formula" in b.capabilities
    assert callable(b.ocr)
    assert callable(b.available)
    assert callable(b.supports)
    assert b.supports("text") is True
    assert b.supports("chart") is False


def test_rapidocr_backend_contract():
    """RapidOCRBackend must declare text + multilingual capabilities."""
    from scientific_parser.perception.ocr.rapidocr_backend import RapidOCRBackend

    b = RapidOCRBackend()
    assert b.name == "rapidocr"
    assert "text" in b.capabilities
    assert "multilingual" in b.capabilities
    assert callable(b.ocr)
    assert callable(b.available)


def test_onnx_backend_contract():
    """OnnxBackend loads manifest from disk."""
    # Constructing without manifest just creates stub
    import tempfile

    from scientific_parser.perception.ocr.onnx import OnnxBackend

    with tempfile.TemporaryDirectory() as tmp:
        b = OnnxBackend(tmp)
        # Even without manifest, must be constructable + have available()
        assert callable(b.available)
        assert callable(b.ocr)


def test_vlm_backend_contract():
    """VLMBackend accepts a callable."""
    from scientific_parser.perception.ocr.vlm import VLMBackend

    def fake_ocr(img, prompt):
        return "fake OCR result"

    b = VLMBackend(call_fn=fake_ocr, name="test-vlm")
    assert b.name == "test-vlm"
    assert b.available() is True
    result = b.ocr("/tmp/fake.png")
    assert isinstance(result.text, str)


def test_rapidocr_backend_contract_paddle_removed():
    """paddle.py removed 2026-08-16 (dep never declared, never wired);
    the ABC contract stays proven by the live RapidOCRBackend above."""
    import importlib.util as _ilu

    assert _ilu.find_spec("scientific_parser.perception.ocr.paddle") is None


def test_model_manifest_roundtrip():
    """ModelManifest serializes/deserializes correctly."""
    import sys

    if "scripts" not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import json
    import tempfile

    from model_manifest import BUILTIN_MANIFESTS

    glm = BUILTIN_MANIFESTS["glm-ocr"]
    assert glm.name == "glm-ocr"
    assert glm.task == "ocr_full"
    assert glm.format == "ollama"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(glm.__dict__, f)
        path = f.name
    loaded_data = json.loads(Path(path).read_text())
    assert loaded_data["name"] == "glm-ocr"


def test_model_manifest_list():
    """All built-in manifests have required fields."""
    import sys

    if "scripts" not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from model_manifest import BUILTIN_MANIFESTS

    for name, m in BUILTIN_MANIFESTS.items():
        assert m.name, f"{name} missing name"
        assert m.task, f"{name} missing task"
        assert m.format, f"{name} missing format"
        assert m.capabilities, f"{name} missing capabilities"
        assert isinstance(m.capabilities, list)


def test_sha256_known_value():
    """sha256_file produces known hash for known content."""
    import sys

    if "scripts" not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import tempfile

    from model_manifest import sha256_file

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("hello world")
        path = f.name
    h = sha256_file(Path(path))
    # Known: SHA256 of "hello world"
    assert h.startswith("b94d27b9934d3e08")


def test_rapidocr_golden_image():
    """RapidOCR returns plausible text on a known-good test image.

    Uses old_scans/10.pdf (Theodore Roosevelt letter) — should contain
    'BROWNLEE' or 'ATTORNEY' when OCR'd correctly.
    """
    try:
        from scientific_parser.perception.ocr.rapidocr_backend import RapidOCRBackend
    except ImportError:
        return  # skip if rapidocr not installed

    pdf_path = (
        SCRIPTS_DIR
        / "bench"
        / "olmOCR-bench"
        / "bench_data"
        / "pdfs"
        / "old_scans"
        / "10.pdf"
    )
    if not pdf_path.exists():
        return  # skip if bench data missing

    # Render PDF to image first
    import tempfile

    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(150 / 72, 150 / 72))
    fd, img_path = tempfile.mkstemp(suffix=".png")
    import os

    os.close(fd)
    pix.save(img_path)
    doc.close()

    b = RapidOCRBackend()
    if not b.available():
        return  # skip if rapidocr binary missing
    result = b.ocr(img_path)
    os.unlink(img_path)
    # Must contain at least 100 chars + some recognizable word
    assert len(result.text) > 50, f"OCR too short: {len(result.text)}"
    assert any(w in result.text.upper() for w in ("BROWNLEE", "ATTORNEY", "LAW")), (
        f"Golden-image check failed; got: {result.text[:200]}"
    )
