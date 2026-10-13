"""Unit tests for font_style (docling MIT port) + hierarchy wiring.

Ported-algorithm contract: conservative weight/slant parsing where a
false "bold" would silently promote body text to a heading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scientific_parser.science.font_style import (  # noqa: E402
    FontStyle,
    bold_heading_lines,
    parse_font_style,
    weight_class,
)


class TestParseFontStyle:
    def test_parse_explicit_bold_suffix(self):
        assert parse_font_style("/Helvetica-Bold") == FontStyle(700, False, True)

    def test_parse_subset_prefix_and_abbreviation(self):
        assert parse_font_style("NKDKGK+HelveticaNeueLTPro-Bd").weight == 700

    def test_parse_foundry_tag_not_bold(self):
        # TB is glued to the family name — not a whole part
        assert parse_font_style("LinLibertineTB").known is False

    def test_parse_italic_slant(self):
        assert parse_font_style("/KIDKQO+Times-Italic").italic is True

    def test_parse_italic_abbreviation(self):
        assert parse_font_style("Garamond-It").italic is True

    def test_parse_book_weight_is_regular(self):
        assert parse_font_style("Avenir-Book").weight == 400

    def test_parse_camel_case_modifier_pair(self):
        assert parse_font_style("SemiBold").weight == 600

    def test_parse_bare_resource_key_unknown(self):
        assert parse_font_style("/F1").known is False

    def test_parse_roman_known_regular(self):
        # "Roman" is a recognized weight token (upright regular)
        assert parse_font_style("Times-Roman") == FontStyle(400, False, True)

    def test_parse_empty_and_none_safe(self):
        assert parse_font_style("") == FontStyle()
        assert parse_font_style(None) == FontStyle()  # type: ignore[arg-type]


class TestWeightClass:
    def test_weight_class_buckets(self):
        assert weight_class(100) == 0
        assert weight_class(400) == 0
        assert weight_class(500) == 1
        assert weight_class(600) == 1
        assert weight_class(700) == 2
        assert weight_class(900) == 2


class TestBoldHeadingLines:
    def _body(self, font="Times-Roman", n=4):
        return [{"text": f"body line {i}", "size": 10.0, "font": font} for i in range(n)]

    def test_bold_same_size_minority_promotes(self):
        lines = (
            [{"text": "Chapter One", "size": 10.0, "font": "Times-Bold"}]
            + self._body()
            + [{"text": "Section Two", "size": 10.0, "font": "Times-Bold"}]
            + self._body()
        )
        assert bold_heading_lines(lines) == ["Chapter One", "Section Two"]

    def test_all_bold_page_inert(self):
        lines = [{"text": f"line {i}", "size": 10.0, "font": "Arial-Bold"} for i in range(8)]
        assert bold_heading_lines(lines) == []

    def test_bold_different_size_excluded(self):
        lines = [{"text": "Big Title", "size": 14.0, "font": "Arial-Bold"}] + self._body(
            font="Arial"
        )
        assert bold_heading_lines(lines) == []

    def test_too_few_lines_inert(self):
        assert bold_heading_lines([{"text": "Only", "size": 10.0, "font": "B-Bold"}]) == []

    def test_empty_lines_list_error_free(self):
        assert bold_heading_lines([]) == []


class TestBoldFlagOrdering:
    """Pins (2026-09-20, EXP-6c): flags-before-name bold recognition.

    NimbusRomNo9L-Medi is the LaTeX \\textbf face — the family has no
    Bold cut and the name parses to weight 500 (class 1). PyMuPDF still
    sets the span bold flag (bit 16). LiteParse is_bold_item ordering:
    descriptor flags first, name parsing second.
    """

    def _page(self, bold_lines, font="NimbusRomNo9L-Regu", bold_share=None):
        lines = [{"text": f"body line {i}", "size": 9.0, "font": font} for i in range(8)]
        for t in bold_lines:
            rec = {"text": t, "size": 9.0, "font": font}
            if bold_share is not None:
                rec["bold_share"] = bold_share
            lines.append(rec)
        return lines

    def test_medi_flag_bold_share_promotes(self):
        lines = self._page(["Hidden Data Display"], font="NimbusRomNo9L-Medi", bold_share=1.0)
        assert bold_heading_lines(lines) == ["Hidden Data Display"]

    def test_medi_name_alone_not_bold(self):
        # Weight 500 by name = class 1 — name parsing alone must NOT promote
        lines = self._page(["Hidden Data Display"], font="NimbusRomNo9L-Medi")
        assert bold_heading_lines(lines) == []

    def test_regular_font_flag_alone_promotes(self):
        # Flag bit overrides a regular-parsing name (ordering pin)
        lines = self._page(["Hidden Data Display"], font="Times-Roman", bold_share=1.0)
        assert bold_heading_lines(lines) == ["Hidden Data Display"]

    def test_bold_share_below_threshold_inert(self):
        lines = self._page(["Hidden Data Display"], font="NimbusRomNo9L-Medi", bold_share=0.5)
        assert bold_heading_lines(lines) == []


class TestHierarchyWiring:
    def test_detect_section_hierarchy_imports_and_has_bold_pass(self):
        import inspect

        from scientific_parser.postproc.text_pipeline import detect_section_hierarchy

        src = inspect.getsource(detect_section_hierarchy)
        assert "bold_heading_lines" in src
        assert "_promote_standalone" in src
