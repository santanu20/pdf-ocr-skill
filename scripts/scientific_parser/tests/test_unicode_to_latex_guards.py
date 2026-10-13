"""iter-62 conversion contracts (deterministic unit pins).

The 10a dictionary fixture routes to the chandra VL model (nondeterministic
reads) — fixture pins flake; these unit pins hold the invariant instead.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PDF_OCR_NO_SKILL_VENV"] = "1"

from scientific_parser.science.unicode_to_latex import convert_unicode_to_latex


def test_ipa_phonetic_stays_verbatim():
    """Dictionary transcriptions keep their phonetic chars; ɛ spells 'e'
    (gold '/temprərəli/'), never \\varepsilon — both with ə-anchors and
    inside /.../ without any (10a: temporarily, template)."""
    out = convert_unicode_to_latex("temporarily /'tɛmprərəli/ adv.")
    assert "temprərəli" in out and "\\varepsilon" not in out, out
    out2 = convert_unicode_to_latex("template /'templɛt/ noun")
    assert "templet" in out2 and "\\varepsilon" not in out2, out2


def test_crossref_arrow_stays_arrow():
    """'→ APART' (arrow + capitalized token) is a dictionary cross-
    reference, not math; a math arrow with operands still converts."""
    out = convert_unicode_to_latex("tell A and B apart → APART")
    assert "→ APART" in out and "\\to" not in out, out
    math_out = convert_unicode_to_latex("the map x → y is smooth")
    assert "\\to" in math_out, math_out
