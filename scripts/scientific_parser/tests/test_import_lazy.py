"""PERF-1 pins: pypdfium2 import stays lazy (PEP 562 via shared.__getattr__).

Guard class: import-time contract. pypdfium2 costs ~85ms of interpreter
startup and is unneeded on --help / cache-replay paths; eager regression
here would silently slow every CLI invocation (Run 27 measurement).
"""

import sys

import pytest


def test_shared_all_excludes_pdfium_names():
    import shared

    assert "_pdfium" not in shared.__all__
    assert "_pdfium_c" not in shared.__all__


def test_shared_getattr_lazy_loads_pdfium():
    import shared

    was_loaded = "pypdfium2" in sys.modules
    try:
        pdfium = shared._pdfium
        assert pdfium is not None
        assert hasattr(pdfium, "PdfDocument")
        assert "pypdfium2" in sys.modules
        # cached: second access returns the same module object
        assert shared._pdfium is pdfium
        assert shared._pdfium_c is sys.modules["pypdfium2.raw"]
    finally:
        if not was_loaded and "pypdfium2" in sys.modules:
            # keep the interpreter clean for the not-loaded assertion below
            for m in list(sys.modules):
                if m == "pypdfium2" or m.startswith("pypdfium2."):
                    del sys.modules[m]


def test_shared_getattr_unknown_name_raises():
    import shared

    with pytest.raises(AttributeError):
        shared._definitely_not_a_real_attr_


def test_wildcard_import_does_not_load_pdfium():
    for m in list(sys.modules):
        if m == "pypdfium2" or m.startswith("pypdfium2."):
            del sys.modules[m]
    # fresh namespace wildcard import, mirroring pdf_ocr.py:76
    ns = {}
    exec("from shared import *", ns)  # noqa: S102
    assert "pypdfium2" not in sys.modules
    assert "_pdfium" not in ns
