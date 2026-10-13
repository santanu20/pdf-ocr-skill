"""rapidfuzz is a hard runtime dependency of the header/footer strip.

Regression pins (2026-10-13): the strip's pass-1 whole-line blanker used to
run UNVERIFIED when rapidfuzz was missing (fail-open import → difflib or no
check at all), silently destroying body text on clean installs that lacked
the then-optional package. These pins keep (1) the base-dependency
declaration and (2) the fail-closed import shape.
"""

import pathlib
import tomllib

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2]
REPO_ROOT = SCRIPTS_DIR.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
ENGINE_SRC = SCRIPTS_DIR / "pdf_ocr.py"


def test_rapidfuzz_declared_as_base_dependency():
    with open(PYPROJECT, "rb") as f:
        base_deps = tomllib.load(f)["project"]["dependencies"]
    names = [d.split(">=")[0].split("==")[0].strip() for d in base_deps]
    assert "rapidfuzz" in names, (
        "rapidfuzz moved out of base [project].dependencies — the footer "
        "strip hard-requires it; moving it to an extra reintroduces the "
        f"clean-install defect. Current base deps: {names}"
    )


def test_strip_import_fails_closed_without_rapidfuzz():
    src = ENGINE_SRC.read_text(encoding="utf-8")
    idx = src.index("from rapidfuzz import fuzz as _rfz")
    window = src[idx : idx + 300]
    assert "except ImportError" not in window, (
        "footer-strip rapidfuzz import is fail-open again — a missing "
        "install must crash the strip, not silently skip containment checks"
    )
