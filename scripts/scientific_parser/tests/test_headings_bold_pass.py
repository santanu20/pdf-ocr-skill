"""Pins (2026-09-20, EXP-6c / det18): bold-same-size heading pass.

Synthetic-PDF end-to-end pins for detect_section_hierarchy + the
preserve_typography heading mask. Root causes fixed:

1. MEDI INVISIBLE (font_style): NimbusRomNo9L-Medi (LaTeX \\textbf
   face) parses to weight 500 by name — never class 2. Fixed by
   honoring the PyMuPDF span bold flag (bold_share >= 0.9). Unit-pinned
   in test_font_style.TestBoldFlagOrdering; here the pass is exercised
   end-to-end with Times-Bold spans.

2. CAPTION/ATTRIBUTION VETO (LiteParse headings.rs:374-381): bold
   body-size "Figure 7" captions and "Source:" lines satisfy every
   other heading rule — they are figure furniture.

3. FIRST-MATCH GATE ABORT: the numbered-pair promote searched once and
   bailed on a gate-failed occurrence, masking a later clean one (a
   page can carry the same pair twice — TOC echo). finditer + first
   PASSING occurrence.

4. SHAPE B INLINE: "2.5 Spreadsheet Integration" set bold as ONE line
   promotes at the level its own prefix dictates (### not ##).

5. ** INJECTION (preserve_typography): order 710 runs after dsh 620,
   wrapping the promoted heading's bold span inside the '###' line.
   Heading lines are masked (sentinel discipline, same as math).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scientific_parser.postproc.text_pipeline import (  # noqa: E402
    detect_section_hierarchy,
    preserve_typography,
)


def _make_pdf(tmp_path, lines, name="pin.pdf"):
    """One line per (text, bold) tuple, Times-Bold/Times-Roman at 9pt."""
    import fitz

    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for text, bold in lines:
        page.insert_text((72.0, y), text, fontname="tibo" if bold else "tiro", fontsize=9.0)
        y += 12.0
    doc.save(str(path))
    doc.close()
    return str(path)


class TestCaptionAttributionVeto:
    def test_figure_and_source_lines_not_promoted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PDF_OCR_BOLD_HEADINGS", "1")
        lines = [
            ("Introduction text goes here.", False),
            ("This manual covers entry.", False),
            ("Fields are validated on save.", False),
            ("See the appendix notes.", False),
            ("Hidden Data Display", True),  # control: must promote
            ("Figure 7", True),  # caption furniture: must NOT promote
            ("Source: USGS", True),  # attribution furniture: must NOT promote
            ("Body text continues here.", False),
            ("Even more body text now.", False),
            ("Additional paragraphs follow.", False),
            ("The index lists topics.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "veto.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "## Hidden Data Display" in out
        assert "## Figure 7" not in out
        assert "## Source: USGS" not in out
        assert "Figure 7\nSource: USGS" in out  # both survive verbatim

    def test_kill_switch_disables_bold_pass(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PDF_OCR_BOLD_HEADINGS", "0")
        lines = [
            ("Introduction text goes here.", False),
            ("This manual covers entry.", False),
            ("Fields are validated on save.", False),
            ("See the appendix notes.", False),
            ("Hidden Data Display", True),
            ("Body text continues here.", False),
            ("Even more body text now.", False),
            ("Additional paragraphs follow.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "kill.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "## Hidden Data Display" not in out


class TestNumberedPairSecondOccurrence:
    def test_gate_failed_first_occurrence_falls_through_to_clean_second(self, tmp_path):
        lines = [
            ("Chapter intro sentence here.", False),
            ("2.2", True),
            ("Interactive Data Entry", True),
            ("follows the entry workflow.", False),  # lowercase next: occ 1 must fail
            ("Fields save on exit.", False),
            ("Data validation runs next.", False),
            ("A second passage ends here.", False),
            ("2.2", True),
            ("Interactive Data Entry", True),
            ("Begin the tour now.", False),
            ("Final body line here.", False),
            ("Additional notes follow.", False),
            ("The index lists topics.", False),
            ("Appendix ends the page.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "pair.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert out.count("### 2.2 Interactive Data Entry") == 1
        # occurrence 1 stays unmerged (gate-failed), not collateral-damaged
        assert "2.2\nInteractive Data Entry\nfollows the entry workflow." in out


class TestShapeBInlineNumbered:
    def test_inline_numbered_bold_line_promotes_at_own_level(self, tmp_path):
        lines = [
            ("Intro paragraph line one ends.", False),
            ("Another body sentence here.", False),
            ("Third body line sits here.", False),
            ("Fourth body line follows.", False),
            ("2.5 Spreadsheet Integration", True),
            ("Body continues after this.", False),
            ("Seventh body line now.", False),
            ("Eighth body line here.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "shapeb.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n### 2.5 Spreadsheet Integration\n" in out
        assert (
            "\n## 2.5 Spreadsheet Integration\n" not in out
        )  # level from own prefix, not fixed ##


class TestPreserveTypographyHeadingMask:
    def test_bold_span_inside_promoted_heading_not_wrapped(self, tmp_path):
        # bold runs separated by a body line (y-gap 24 > 1.5x12 merge
        # threshold) so Phase-2 cross-line merging cannot fuse them
        lines = [
            ("Hidden Data Display", True),
            ("Read the notes before.", False),
            ("Important Caveat", True),
            ("Final body line here.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "mask.pdf")
        text = (
            "Important Caveat applies to this.\n\n"
            "### 2.3 Hidden Data Display\n\n"
            "Read the notes before."
        )
        out = preserve_typography(pdf, 1, text)
        assert "### 2.3 Hidden Data Display" in out  # heading intact
        assert "**Hidden Data Display**" not in out  # no ** injection
        assert "**Important Caveat**" in out  # body wrapping still active


class TestWrappedHeadingMerge:
    """Pins (2026-09-20, det19): a heading WRAPPED across two adjacent
    bold PDF lines merges into ONE heading. Root cause: each half
    promoted separately ('## Leaking Ducts Can Also Be Hazardous to' +
    '## Your Health') broke reading-order probes — the wrapped title
    lost contiguity (ducts-flyer 0d767c7c)."""

    def test_two_adjacent_bold_lines_merge_into_one_heading(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PDF_OCR_BOLD_HEADINGS", "1")
        lines = [
            ("Body text opens the page.", False),
            ("More body text follows.", False),
            ("Then a third body line.", False),
            ("Leaking Ducts Can Also Be Hazardous to", True),  # wrap 1/2
            ("Your Health", True),  # wrap 2/2
            ("Body text resumes after.", False),
            ("Even more body follows.", False),
            ("A final line of body.", False),
            ("Extra body line added.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "wrap.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Leaking Ducts Can Also Be Hazardous to Your Health\n" in out
        assert "## Your Health\n" not in out  # second half never alone

    def test_terminal_punct_first_line_blocks_merge(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PDF_OCR_BOLD_HEADINGS", "1")
        lines = [
            ("Body text opens the page.", False),
            ("More body text follows.", False),
            ("Then a third body line.", False),
            ("Overview", True),  # ends clean... make pair with punct below
            ("Results.", True),  # terminal '.' -> excluded from titles
            ("Body text resumes after.", False),
            ("Even more body follows.", False),
            ("A final line of body.", False),
            ("Extra body line added.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "nopunct.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        # 'Overview Results.' must NOT be emitted as a merged heading
        assert "## Overview Results." not in out

    def test_body_line_between_bold_lines_blocks_merge(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PDF_OCR_BOLD_HEADINGS", "1")
        lines = [
            ("Body text opens the page.", False),
            ("More body text follows.", False),
            ("Then a third body line.", False),
            ("System Configuration", True),
            ("Body text interrupts here.", False),  # adjacency broken
            ("Overview Panel", True),
            ("Even more body follows.", False),
            ("A final line of body.", False),
            ("Extra body line added.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "gap.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        # never a cross-body merged heading
        assert "## System Configuration Overview Panel" not in out


def _make_pdf_sized(tmp_path, lines, name="sized.pdf"):
    """One line per (text, bold, size) tuple — size is per-line (EXP-8).
    Optional 4th tuple element overrides the y step: tight steps
    (≈ size + 4) emulate a wrapped title set inside ONE text block;
    the default wide step (max(size, 14) + 16) emulates spaced lines."""
    import fitz

    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for entry in lines:
        text, bold, size = entry[0], entry[1], entry[2]
        y_step = entry[3] if len(entry) > 3 else max(size, 14.0) + 16.0
        page.insert_text((72.0, y), text, fontname="tibo" if bold else "tiro", fontsize=size)
        y += y_step
    doc.save(str(path))
    doc.close()
    return str(path)


class TestSparseTitleSizeFallback:
    """det20: when the heading cluster is <10% of lines, p90 reads body
    size and the old size statistic went blind. The cluster walk admits
    size levels >=1.15x median with cumulative share <=15% (docling
    size clustering)."""

    def test_display_title_promoted_and_dropcap_blocked(self, tmp_path):
        lines = [("22", False, 9.0)]  # page furniture line
        lines += [("Field Methods in Chuj", False, 20.0)]  # 1/22 big -> p90 dead
        lines += [("T", False, 40.0)]  # dropcap: alpha<3 gate must block
        for i in range(19):
            lines.append((f"Body sentence number {i} continues here.", False, 9.0))
        pdf = _make_pdf_sized(tmp_path, lines, "sparse.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Field Methods in Chuj\n" in out
        assert "\n## T\n" not in out

    def test_mid_tier_promoted_and_114pct_excluded(self, tmp_path):
        # Garamond-007 shape: body 9.0, mid heading 11.0 (1.22x),
        # 'Chapter 2' at 10.2 (1.13x) below the 1.15 floor.
        lines = [
            ("22", False, 9.0),
            ("Narratives in Chuj", False, 11.0),
            ("Body first sentence flows on.", False, 9.0),
            ("Chapter 2", False, 10.2),
            ("Body second sentence flows.", False, 9.0),
            ("More body keeps flowing on.", False, 9.0),
        ]
        for i in range(6, 20):
            lines.append((f"Body sentence number {i} goes here.", False, 9.0))
        pdf = _make_pdf_sized(tmp_path, lines, "midtier.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Narratives in Chuj\n" in out
        assert "\n## Chapter 2\n" not in out

    def test_dense_heading_page_keeps_p90_path(self, tmp_path):
        # 12/30 lines at 1.56x: share 40% > 15% cap, but the p90 branch
        # fires FIRST (p90 lands on the big tier) — fallback cap must
        # not regress dense multi-heading pages.
        lines = [("22", False, 9.0)]
        for i in range(12):
            lines.append((f"Section {i} Heading Line", False, 14.0))
        for i in range(18):
            lines.append((f"Body sentence number {i} flows.", False, 9.0))
        pdf = _make_pdf_sized(tmp_path, lines, "dense.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Section 0 Heading Line\n" in out
        assert "\n## Section 11 Heading Line\n" in out


class TestPrevGateForms:
    """det20: predecessor gate accepts sentence periods inside closing
    parens/quotes ('...transcription.)') and heading-shaped lines
    ('Chapter 2' above a mid-tier heading)."""

    def test_paren_period_predecessor(self, tmp_path):
        lines = [
            ("Head and Shoulders", True),
            ("This sentence closes a parenthetical (see note 3.)", False),
            ("Introduction to the Texts", True),
            ("Two of the stories are of foreign origin.", False),
            ("More body text after that sentence.", False),
            ("Further body text keeps this going.", False),
            ("One more body line for safety.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "paren.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Introduction to the Texts\n" in out

    def test_heading_shaped_predecessor(self, tmp_path):
        lines = [
            ("Chapter 2", False),
            ("Narratives in Chuj", True),
            ("This collection of narratives is varied.", False),
            ("More body text follows the first.", False),
            ("Further body text keeps this going.", False),
            ("One more body line for safety.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "hshape.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Narratives in Chuj\n" in out

    def test_long_nonterminal_predecessor_still_blocks(self, tmp_path):
        lines = [
            ("This is a long body line without any terminal punctuation", False),
            ("Summary of Findings", True),
            ("The findings were mixed across sites.", False),
            ("More body text after the heading.", False),
            ("Further body text keeps this going.", False),
            ("One more body line for safety.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "block.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Summary of Findings\n" not in out


class TestInlineTabNumberedPair:
    """det20: fitz and the text extractor disagree on line splits (Brill
    002). fitz carries '8' / 'Choosing between...' as TWO lines (the
    bold-pair source) while the extracted text joins them with a tab —
    the pair promoter must accept a same-line tab separator."""

    def test_tab_joined_pair_merges_at_own_level(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PDF_OCR_BOLD_HEADINGS", "1")
        span_lines = [
            ("Body opens the section here.", False),
            ("8", True),
            ("Choosing between Observer Models", True),
            ("Body follows the new heading.", False),
            ("More body text after that.", False),
            ("Further body text here now.", False),
            ("A fourth body line added.", False),
            ("A fifth body line as well.", False),
        ]
        pdf = _make_pdf(tmp_path, span_lines, "tabpair.pdf")
        text = "\n".join(
            [
                "Body opens the section here.",
                "8\tChoosing between Observer Models",
                "Body follows the new heading.",
                "More body text after that.",
                "Further body text here now.",
            ]
        )
        out = detect_section_hierarchy(pdf, 1, text)
        assert "# 8 Choosing between Observer Models" in out
        assert "## 8" not in out


class TestPrevGateRegressions:
    """det21 (odlbench docs 51/146): the det20 heading-shaped prev
    exception over-admitted two prev classes — numbered list entries
    ('2.3 Problem framing' above a numbered heading) and abbreviation
    continuations ('No. of Seats Aquino' above a table stub). And
    single-word bold stubs ('Ramos', 'Administration') that echo in
    table rows are furniture, unlike page-unique 'Contents'."""

    def test_numbered_item_prev_blocks(self, tmp_path):
        lines = [
            ("Body opens the section here.", False),
            ("2.3 Problem framing", False),  # numbered list entry, no punct
            ("3. Envisioning sustainable futures", True),  # must NOT promote
            ("Learners practice these skills.", False),
            ("More body text after that.", False),
            ("Further body text here now.", False),
            ("A fourth body line added.", False),
            ("A fifth body line as well.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "numprev.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        # '## 3.' is a substring of '### 3.' — one check covers both forms
        assert "## 3. Envisioning" not in out
        assert "3. Envisioning sustainable futures" in out  # survives verbatim

    def test_abbreviation_prev_blocks(self, tmp_path):
        lines = [
            ("Body opens the section here.", False),
            ("No. of Seats Aquino", False),  # abbreviation continuation
            ("Position", True),  # table column stub: must NOT promote
            ("Body text resumes after stub.", False),
            ("More body text after that.", False),
            ("Further body text here now.", False),
            ("A fourth body line added.", False),
            ("A fifth body line as well.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "abbrevprev.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "## Position" not in out
        assert "No. of Seats Aquino\nPosition" in out  # pair survives verbatim

    def test_single_word_echo_veto_blocks_table_stub(self, tmp_path):
        lines = [
            ("Body opens the section here.", False),
            ("The prior section ended.", False),
            ("Ramos", True),  # single-word stub echoing below: must NOT promote
            ("The Ramos administration continued reforms.", False),
            ("More body text after that.", False),
            ("Further body text here now.", False),
            ("A fourth body line added.", False),
            ("A fifth body line as well.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "echo.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "## Ramos" not in out
        assert "\nRamos\n" in out  # stub survives verbatim

    def test_single_word_unique_survives(self, tmp_path):
        lines = [
            ("Body opens the section here.", False),
            ("The front matter ends here.", False),
            ("Contents", True),  # page-unique single word: promotes
            ("A preface follows the listing.", False),
            ("More body text after that.", False),
            ("Further body text here now.", False),
            ("A fourth body line added.", False),
            ("A fifth body line as well.", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "unique.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Contents\n" in out


class TestCascadeAndSizeLaneDet22:
    """det22 pins: display-size multiline merges, display-size numbered
    pairs, pos-0 page openers, and the two non-merge guards (spans-index
    gap, >10% size delta) — the 075cc4e9 / 145 / 9530001d shapes."""

    BODY = [f"Body sentence number {i} flows along here." for i in range(12)]

    def test_multiline_merge_with_interior_colon(self, tmp_path):
        # 075cc4e9 shape: display title wrapped over 4 span lines;
        # interior ':' fragments must stay in the chain, only the FINAL
        # fragment needs to be terminal-free.
        lines = [("22", False, 9.0)]
        lines += [(b, False, 9.0) for b in self.BODY[:4]]
        lines += [
            ("Non-Specific", False, 14.0, 18.0),
            ("Inflammatory", False, 14.0, 18.0),
            ("Conjunctivitis:", False, 14.0, 18.0),
            ("Diagnosis", False, 14.0),
        ]
        lines += [(b, False, 9.0) for b in self.BODY[4:]]
        pdf = _make_pdf_sized(tmp_path, lines, "mln.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Non-Specific Inflammatory Conjunctivitis: Diagnosis\n" in out
        assert "## Non-Specific\n" not in out  # no shattered fragments

    def test_display_size_numbered_pair(self, tmp_path):
        # 145 shape: '4.1' + 'Introduction' at display size — the bold
        # lane's body-size window can never see them; the size lane pair
        # shape must emit the numbered level (###).
        lines = [("22", False, 9.0)]
        lines += [(b, False, 9.0) for b in self.BODY[:4]]
        lines += [("4.1", False, 14.0, 18.0), ("Introduction", False, 14.0)]
        lines += [(b, False, 9.0) for b in self.BODY[4:]]
        pdf = _make_pdf_sized(tmp_path, lines, "pair14.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n### 4.1 Introduction\n" in out

    def test_pos0_page_opener_promotes(self, tmp_path):
        # 145 cascade shape: 'Chapter 4' is the very first line of the
        # text — pos==0 must satisfy the whole-line gate.
        lines = [("Advanced Topics in Geology", False, 16.0)]
        lines += [(b, False, 9.0) for b in self.BODY]
        pdf = _make_pdf_sized(tmp_path, lines, "pos0.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Advanced Topics in Geology\n" in "\n" + out

    def test_index_gap_breaks_merge(self, tmp_path):
        # A body line BETWEEN two display lines breaks the run by spans
        # index gap: two separate headings, never one merged.
        lines = [("22", False, 9.0)]
        lines += [(b, False, 9.0) for b in self.BODY[:3]]
        lines += [("Study Methods", False, 14.0)]
        lines += [("We describe the approach here.", False, 9.0)]
        lines += [("Key Results", False, 14.0)]
        lines += [(b, False, 9.0) for b in self.BODY[3:]]
        pdf = _make_pdf_sized(tmp_path, lines, "gap.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Study Methods\n" in out
        assert "\n## Key Results\n" in out
        assert "## Study Methods Key Results" not in out

    def test_size_delta_breaks_merge(self, tmp_path):
        # 145 shape: 'Chapter 4'@20.7 directly above 'Nonlinear
        # equations'@24.8 — adjacent spans but >10% size delta = stacked
        # heading levels, not one wrapped title.
        lines = [("Chapter 4", False, 20.7, 28.0)]
        lines += [("Nonlinear equations", False, 24.8)]
        lines += [(b, False, 9.0) for b in self.BODY]
        pdf = _make_pdf_sized(tmp_path, lines, "delta.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Chapter 4\n" in "\n" + out
        assert "\n## Nonlinear equations\n" in out
        assert "## Chapter 4 Nonlinear equations" not in out


class TestDet23StructuralGuards:
    """det23: PageHeader single-column rescue, region heading-loss guard,
    caption-prev block boundary."""

    def test_caption_prev_accepts(self, tmp_path):
        """Bold heading directly after a figure caption promotes (111/154
        class): a caption closes a block even ending lowercase."""
        lines = [
            ("Earlier work established the baseline.", False),
            ("Figure 7.5: Georgina Ballantine holds the British record for a salmon", False),
            ("Sportfishing Demographics", True),
            ("Among those who fish for sport, few are female", False),
            ("participation rates vary by region and year", False),
            ("subsequent surveys confirmed the trend", False),
            ("closing note on data sources follows below", False),
        ]
        pdf = _make_pdf(tmp_path, lines, "caption_prev.pdf")
        from scientific_parser.postproc.text_pipeline import (
            detect_section_hierarchy,
        )

        out = detect_section_hierarchy(str(pdf), 1, "\n".join(t for t, _ in lines))
        assert "## Sportfishing Demographics" in out

    def test_region_heading_loss_wrap_rejected(self):
        """_region_heading_loss: heading words surviving only inside pipe
        cells count as LOST (junk table wrap, odlbench 133/123)."""
        import pdf_ocr as po

        pre = "Intro text.\n\n## 7.6 Examples of Women's Impact\n\nbody"
        wrapped = (
            "| 7.6 Examples of Women's Impact body starts here wrapped into one cell |\n"
            "|---|---|\n| more | cells |"
        )
        assert po._region_heading_loss(pre, wrapped) == ["## 7.6 Examples of Women's Impact"]

    def test_region_heading_loss_plain_line_kept(self):
        """Heading text surviving as its own non-pipe line = not lost
        (real table pages: heading region survives; adoption allowed)."""
        import pdf_ocr as po

        pre = "Intro text.\n\n## 7.6 Examples of Women's Impact\n\nbody"
        rebuilt = "## 7.6 Examples of Women's Impact\n\n| A | B |\n|---|---|\n| 1 | 2 |"
        assert po._region_heading_loss(pre, rebuilt) == []

    def test_region_demoted_plain_line_restored(self):
        """det27 (01030000000168): E0 text-region re-extract keeps the
        heading as a PLAIN line — markup-aware loss flags it and
        _restore_heading_markup re-marks it in place (adoption keeps
        the E4/equation value a reject would destroy)."""
        import pdf_ocr as po

        pre = "Intro text.\n\n## Activity 1: Determining pH\n\nbody\n\n$pH = -\\log[H+]$"
        rebuilt = "Activity 1: Determining pH\n\nbody\n\n$pH = -\\log[H+]$"
        lost = po._region_heading_loss(pre, rebuilt)
        assert lost == ["## Activity 1: Determining pH"]
        restored, unresolved = po._restore_heading_markup(rebuilt, lost)
        assert unresolved == []
        assert "\n## Activity 1: Determining pH\n" in "\n" + restored + "\n"
        assert "$pH = -\\log[H+]$" in restored  # routed value preserved
        # pipe-cell-only survival stays unresolvable (123 class)
        wrapped = (
            "| Activity 1: Determining pH body wrapped into one cell |\n|---|---|\n| more | cells |"
        )
        lost2 = po._region_heading_loss(pre, wrapped)
        _, unresolved2 = po._restore_heading_markup(wrapped, lost2)
        assert unresolved2 == ["## Activity 1: Determining pH"]

    def test_pageheader_single_column_rescue(self, tmp_path):
        """105-class: distinct-size top-band title with body flowing
        beneath is a section title — NOT stripped as a masthead header."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        y = 30.0  # top band (<10% of 842pt A4)
        page.insert_text((72, y), "The Scholarly Publishing Cycle", fontsize=18)
        y += 24
        for ln in [
            "Scholarly publishing is the cycle of",
            "review, editing, and dissemination.",
        ]:
            page.insert_text((72, y), ln, fontsize=10)
            y += 14
        y += 20
        for i in range(8):
            page.insert_text((72, y), f"Body paragraph {i} of the page text.", fontsize=10)
            y += 14
        p = tmp_path / "ph.pdf"
        doc.save(str(p))
        doc.close()

        text = "The Scholarly Publishing Cycle\n" + "Body line.\n" * 10
        meta = extract_page_metadata(text, page_num=1, pdf_path=str(p))
        assert "The Scholarly Publishing Cycle" not in meta["hf_lines"]


class TestDet24ChromeAndTableAdopt:
    """det24: running-head chrome veto on the det23a rescue/tight gates;
    substantial-table adoption under the det23b heading-loss guard."""

    def test_chrome_journal_line_stripped(self, tmp_path):
        """Journal vol/no/year running head is chrome even with rescue
        geometry (title-sized, body beneath) — stays a header (531)."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        chrome = (
            "International Journal of Humanities and Social Science Vol. 9 No. 12 December 2019"
        )
        page.insert_text((72, 30.0), chrome, fontsize=12)
        y = 54.0
        for ln in ["Review, editing, and dissemination.", "Second body line here."]:
            page.insert_text((72, y), ln, fontsize=10)
            y += 14
        for i in range(8):
            page.insert_text((72, y), f"Body paragraph {i} of the page text.", fontsize=10)
            y += 14
        p = tmp_path / "chrome.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            chrome + "\n" + "Body line.\n" * 10, page_num=1, pdf_path=str(p)
        )
        assert any("International Journal" in l for l in meta["hf_lines"])

    def test_chrome_report_id_stripped(self, tmp_path):
        """Report/document-ID line at body size with body flowing beneath
        (tight-continuation geometry) is still chrome (bec1f712)."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        chrome = "ANEC-R&T-2010-ENV-003final June 2010"
        page.insert_text((72, 30.0), chrome, fontsize=10)
        y = 70.0
        for i in range(10):
            page.insert_text((72, y), f"Body paragraph {i} of the report text.", fontsize=10)
            y += 14
        p = tmp_path / "report.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            chrome + "\n" + "Body line.\n" * 10, page_num=1, pdf_path=str(p)
        )
        assert any("ANEC" in l for l in meta["hf_lines"])

    def test_region_header_cell_survives(self):
        """A lost heading that equals a routed table's header CELL survived
        (bold table header promoted by dsh, 137297b4) — no heading loss."""
        import pdf_ocr as po

        pre = "## Degree Requirements\n\nRequirements"
        rebuilt = (
            "Requirements\n\n| Degree Requirements | Credit Hours |\n"
            "|---|---|\n| University Core Curriculum Requirements | 39 |\n"
            "| Requirements for Major in Mechanical Engineering | (9) + 87 |"
        )
        assert po._region_heading_loss(pre, rebuilt) == []

    def test_region_one_row_wrap_still_lost(self):
        """Whole-page 1-cell wrap still counts the heading as lost — header
        CELL equality is required, containment is not survival (123)."""
        import pdf_ocr as po

        pre = "## The Data Journey\n\nintro text"
        wrapped = (
            "| The Data Journey To get started follow the steps and "
            "see the fruit production numbers Strawberries |\n|---|---|\n| 40,000 | kg |"
        )
        assert po._region_heading_loss(pre, wrapped) == ["## The Data Journey"]


class TestDet25ChromeJoinQuote:
    """det25: chrome widenings (URL/date/attribution/trailnum/et-al),
    right-float letterhead, citation-block rescue veto, word-gap span
    join, and quote-normalized title matching."""

    def test_chrome_url_and_date_stripped(self, tmp_path):
        """URL and month-day-year band lines are chrome (ojrd URL,
        'Reference Document - 1 April 2022')."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 30.0), "http://www.ojrd.com/content/5/1/10", fontsize=8)
        page.insert_text((72, 44.0), "Reference Document - 1 April 2022", fontsize=10)
        y = 70.0
        for i in range(10):
            page.insert_text((72, y), f"Body paragraph {i} of the report text.", fontsize=10)
            y += 14
        p = tmp_path / "url_date.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            "http://www.ojrd.com/content/5/1/10\nReference Document - 1 April 2022\n"
            + "Body line.\n" * 10,
            page_num=1,
            pdf_path=str(p),
        )
        joined = "\n".join(meta["hf_lines"])
        assert "ojrd.com" in joined
        assert "Reference Document" in joined

    def test_chrome_attribution_footer_stripped(self, tmp_path):
        """'Prepared by:' attribution byline above a date line (tight-
        continuation geometry) is chrome, not body flow (d461e0d4)."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((400, 796.0), "Prepared by:  Becky Geltz,", fontsize=9)
        page.insert_text((450, 808.0), "April 28, 2012", fontsize=9)
        for i in range(12):
            page.insert_text((72, 100.0 + i * 14), f"Data row {i} of the table.", fontsize=9)
        p = tmp_path / "attr.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            "\n".join(f"Data row {i} of the table." for i in range(12))
            + "\nPrepared by:  Becky Geltz,\nApril 28, 2012",
            page_num=1,
            pdf_path=str(p),
        )
        assert any("Prepared by" in l for l in meta["hf_lines"])

    def test_right_float_letterhead_not_rescued(self, tmp_path):
        """Short top-band line floating past the page midline is
        letterhead chrome, never a section title (fd9f7d29 'Mesto
        Sobrance')."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((414, 40.0), "Mesto Sobrance", fontsize=14)
        y = 64.0
        for ln in [
            "Projekt pilotna implementacia programoveho rozpoctovania",
            "efektivnosti vynakladania verejnych zdrojov a ucinnosti",
        ]:
            page.insert_text((102, y), ln, fontsize=11)
            y += 12
        for i in range(8):
            page.insert_text((106, y), f"Ukazovatel {i} meria plnenie ciela.", fontsize=12)
            y += 14
        p = tmp_path / "float.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            "Mesto Sobrance\n" + "Body text.\n" * 10, page_num=1, pdf_path=str(p)
        )
        assert any("Mesto Sobrance" in l for l in meta["hf_lines"])

    def test_span_gap_join_promotes_kerned_title(self, tmp_path):
        """Two spans on one line joined by a word gap ('Basis' + 'Fields')
        must not glue into 'BasisFields' — the numbered pair promotes
        (035)."""
        import fitz

        from scientific_parser.postproc.text_pipeline import (
            detect_section_hierarchy,
        )

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 40.0), "22", fontsize=10, fontname="tiro")
        page.insert_text((72, 70.0), "4", fontsize=25, fontname="tibo")
        page.insert_text((72, 110.0), "Basis", fontsize=17, fontname="tibo")
        page.insert_text((122, 110.0), "Fields", fontsize=17, fontname="tibo")
        for i, ln in enumerate(
            [
                "The basis fields generate the kernel.",
                "We define them axiomatically below.",
                "Several examples illustrate the case.",
                "Further constructions extend the idea.",
                "The proofs appear in the appendix.",
                "See also the earlier chapters.",
                "The first axiom fixes the unit.",
                "The second axiom fixes closure.",
                "The third axiom fixes duality.",
                "Continuity follows from compactness.",
                "Uniqueness holds under mild conditions.",
                "The remaining cases are analogous.",
                "Full statements appear later.",
                "This concludes the overview.",
            ]
        ):
            page.insert_text((72, 140.0 + i * 16), ln, fontsize=10, fontname="tiro")
        p = tmp_path / "kerned.pdf"
        doc.save(str(p))
        doc.close()

        out = detect_section_hierarchy(str(p), 1, "22\n4\nBasis Fields\n" + "Body line.\n" * 14)
        assert "# 4 Basis Fields" in out
        assert "BasisFields" not in out

    def test_quote_normalized_matching(self, tmp_path):
        """Span title with a typographic apostrophe matches a straight-
        quote text line (133 'Women's Impact') — base-14 PDF fonts cannot
        round-trip the curly apostrophe through span extraction, so the pin
        uses the mirror of production (straight in SPANS, curly in text) —
        same 1:1 normalization contract."""
        import fitz

        from scientific_parser.postproc.text_pipeline import (
            detect_section_hierarchy,
        )

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 40.0), "22", fontsize=10, fontname="tiro")
        page.insert_text((72, 70.0), "7.6 Examples of Women's Impact", fontsize=14, fontname="tibo")
        for i, ln in enumerate(
            [
                "The river record stood for decades.",
                "Anglers reported the catches here.",
                "Further notes on the survey follow.",
                "Data collection ended in the spring.",
                "The authors thank the record keepers.",
                "See the appendix for full tables.",
            ]
        ):
            page.insert_text((72, 100.0 + i * 16), ln, fontsize=10, fontname="tiro")
        p = tmp_path / "quote.pdf"
        doc.save(str(p))
        doc.close()

        out = detect_section_hierarchy(
            str(p), 1, "22\n7.6 Examples of Women\u2019s Impact\n" + "Body line.\n" * 6
        )
        assert "## 7.6 Examples of Women" in out and "Impact" in out


class TestDet26StandaloneAndAdjacency:
    """det26: standalone promoter must match the FULL title (the old
    t[:50] prefix match emitted the whole title while leaving the line
    remainder in place, doubling the tail); chrome-cluster adjacency
    must not capture lines LARGER than body (section titles under
    journal chrome stay rescuable)."""

    def test_long_title_no_tail_doubling(self, tmp_path):
        """121-class: a >50-char display title promotes as ONE clean
        heading — the tail past char 50 is never re-emitted."""
        title = "II. Set Up the Restriction Digests of the Suspect and the Evidence DNA Samples"
        lines = (
            [("22", False, 9.0)]
            + [(title, True, 12.0)]
            + [(f"Body text line number {i} flows on here.", False, 10.0) for i in range(9)]
        )
        pdf = _make_pdf_sized(tmp_path, lines, "longtitle.pdf")
        text = (
            "22\n"
            + title
            + "\n"
            + "\n".join(f"Body text line number {i} flows on here." for i in range(9))
        )
        out = detect_section_hierarchy(pdf, 1, text)
        assert f"## {title}" in out
        # The prefix-match bug doubled the tail past char 50
        # ('...DNA\u201d and \u201cEvidence\u201d DNA').
        assert "\u201d and \u201cEvidence\u201d DNA" not in out.replace(title, "")

    def test_chrome_adjacency_size_guard(self, tmp_path):
        """029-class: a DOI/URL chrome line directly above a
        larger-than-body section title must NOT pull the title into the
        chrome cluster — the title rescues, the DOI strips."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72.0, 30.0), "DOI: http://dx.doi.org/10.5772/intechopen.90696", fontsize=9
        )
        page.insert_text((72.0, 53.0), "5. The dynamics", fontsize=11)
        page.insert_text((72.0, 78.0), "The next step is to construct a model.", fontsize=10)
        y = 92.0
        for i in range(8):
            page.insert_text((72.0, y), f"Body paragraph {i} of the page text.", fontsize=10)
            y += 14
        p = tmp_path / "adj.pdf"
        doc.save(str(p))
        doc.close()

        text = (
            "DOI: http://dx.doi.org/10.5772/intechopen.90696\n"
            "5. The dynamics\n"
            "The next step is to construct a model.\n" + "Body line.\n" * 8
        )
        meta = extract_page_metadata(text, page_num=1, pdf_path=str(p))
        assert "5. The dynamics" not in meta["hf_lines"]
        assert any("DOI" in l for l in meta["hf_lines"])


class TestDet28TocAndStrategy2:
    """det28: TOC/list-block veto in the bold lane, promote_allcaps
    Strategy-2 standing down when dsh already emitted a heading, and
    restore preferring the LAST plain-line match (01030000000198)."""

    def test_strategy2_stands_down_when_dsh_spoke(self):
        from scientific_parser.postproc.text_pipeline import (
            promote_allcaps_headings,
        )

        t = (
            "## Existing Heading\n\n"
            "1. Introduction of OCR Pack\n"
            "2. Methods Used Here\n"
            "3. Results Overview\n"
        )
        out = promote_allcaps_headings(t)
        assert "## 1. Introduction of OCR Pack" not in out
        assert "## Existing Heading" in out

    def test_strategy2_legacy_path_preserved(self):
        from scientific_parser.postproc.text_pipeline import (
            promote_allcaps_headings,
        )

        # dsh emitted nothing (no '#' line): legacy numbered heuristic
        # still fires — it is the safety net for pages without span
        # evidence.
        t = "1. Introduction of OCR Pack\n\nbody text here follows it.\n"
        out = promote_allcaps_headings(t)
        assert "## 1. Introduction of OCR Pack" in out

    def test_toc_entries_not_promoted(self, tmp_path):
        # 198 shape: display title + adjacent numbered entries, all
        # bold at body size. The adjacency run (>=3 numbered titles on
        # consecutive PDF lines) vetoes every entry.
        lines = [("Contents Heading", True, 26.0)]
        for n, ttl in [
            ("1.", "Overview of OCR Pack"),
            ("2.", "Introduction of Product Services"),
            ("3.", "Product - Detail Specification"),
            ("4.", "Integration Policy"),
        ]:
            lines.append((f"{n} {ttl}", True, 14.0))
        pdf = _make_pdf_sized(tmp_path, lines, "toc198.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "## Contents Heading" in out
        assert "## 1. Overview of OCR Pack" not in out
        assert "## 3. Product - Detail Specification" not in out
        # entries survive verbatim as plain lines
        assert "\n1. Overview of OCR Pack\n" in "\n" + out + "\n"

    def test_toc_veto_spaced_sections_survive(self, tmp_path):
        # Real numbered sections carry body text between them — gaps
        # in spans indices break the adjacency run and promotion must
        # continue (002 Brill / 007 class).
        lines = [("Intro body opener here.", False, 11.0)]
        lines += [(f"Body filler number {i} keeps spacing wide.", False, 11.0) for i in range(4)]
        lines.append(("2.2 Interactive Data Entry", True, 11.0))
        lines += [(f"More body filler number {i} wide.", False, 11.0) for i in range(4)]
        lines.append(("2.3 Hidden Data Display", True, 11.0))
        lines += [(f"Tail body filler number {i} wide.", False, 11.0) for i in range(3)]
        pdf = _make_pdf_sized(tmp_path, lines, "secs.pdf")
        text = "\n".join(t for t, *_ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "## 2.2 Interactive Data Entry" in out
        assert "## 2.3 Hidden Data Display" in out

    def test_restore_prefers_last_match(self):
        import pdf_ocr as po

        rebuilt = "Contents\nintro line\n\nContents\n"
        restored, unresolved = po._restore_heading_markup(rebuilt, ["## Contents"])
        assert unresolved == []
        lines = restored.rstrip("\n").split("\n")
        assert lines[0] == "Contents"  # TOC echo stays plain
        assert lines[-1] == "## Contents"  # last occurrence re-marked


class TestDet29CorroborationAndCaptionTail:
    """det29: masthead-strip requires chrome corroboration (bare
    Title-Case display lines over a gapped body are page TITLES);
    caption-continuation prev gate (pattern on prev2 when the caption
    wraps)."""

    def test_masthead_without_chrome_is_title(self, tmp_path):
        """156-class: top-band display title, body starts FAR beneath
        (outside every rescue window), zero chrome evidence — must NOT
        be stripped as a masthead."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 58.0), "Fact-Checking", fontsize=14)
        y = 128.0  # 70pt gap: far outside the 2.2x-size rescue window
        for i in range(10):
            page.insert_text((72, y), f"Body paragraph {i} of the page.", fontsize=9)
            y += 14
        p = tmp_path / "masthead_title.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            "Fact-Checking\n" + "Body paragraph.\n" * 10, page_num=1, pdf_path=str(p)
        )
        assert "Fact-Checking" not in meta["hf_lines"]

    def test_allcaps_masthead_still_stripped(self, tmp_path):
        """All-caps journal styling corroborates a masthead — stays a
        header even with the same gapped-body geometry."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 58.0), "JOURNAL OF APPLIED MECHANICS", fontsize=14)
        y = 128.0
        for i in range(10):
            page.insert_text((72, y), f"Body paragraph {i} of the page.", fontsize=9)
            y += 14
        p = tmp_path / "masthead_caps.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            "JOURNAL OF APPLIED MECHANICS\n" + "Body paragraph.\n" * 10,
            page_num=1,
            pdf_path=str(p),
        )
        assert any("JOURNAL OF APPLIED MECHANICS" in l for l in meta["hf_lines"])

    def test_caption_tail_prev_promotes(self, tmp_path):
        """111-class: caption wraps ('Figure 8.1: ... force vortex' +
        'measuring probes'); the bold heading's direct prev is the
        caption TAIL — pattern sits one line up."""
        lines = [
            ("Figure 8.1: a) apparatus with free vortex", False),
            ("measuring probes and calipers on site", False),
            ("7. THEORY", True),
            ("Two types of vortices are distinguished here.", False),
        ]
        lines += [(f"Body sentence number {i} follows the theory.", False) for i in range(8)]
        pdf = _make_pdf(tmp_path, lines, "captail.pdf")
        text = "\n".join(t for t, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## 7. THEORY\n" in "\n" + out


class TestDet30StackedPrev:
    """det30: size-lane stacked-prev allowance. Doc 200: kicker@11 above
    title@17 — the title's prev is a mixed-case >6-word kicker, so every
    built-in prev gate fails; the allowance admits prev lines that are
    themselves size_titles. Doc 156 FP: a lowercase-initial body line in
    size_titles must stay unpromoted (uppercase-initial guard)."""

    def test_kicker_prev_admits_title(self, tmp_path):
        lines = [("22", False, 9.0)]  # furniture so title is not at pos 0
        lines += [
            # kicker: 8 words > heading-shaped prev word cap, no punct
            ("Introduction of product services and key features", False, 11.0),
            ("Key Functions by Service Flow", False, 17.0),
        ]
        for i in range(17):
            lines.append((f"Body sentence number {i} continues here.", False, 9.0))
        pdf = _make_pdf_sized(tmp_path, lines, "stacked.pdf")
        text = "\n".join(t for t, _, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "\n## Key Functions by Service Flow\n" in "\n" + out
        assert "\n## Introduction of product services and key features\n" in "\n" + out

    def test_lowercase_initial_line_not_promoted(self, tmp_path):
        lines = [("22", False, 9.0)]
        lines += [
            # >6-word Title-Case line: fails heading-shaped prev, sits in
            # size_titles so the stack allowance would otherwise fire
            ("A longer mixed-case reference line that wraps here", False, 11.0),
            ('entry "Fact-checking" reference line for the wiki', False, 17.0),
        ]
        for i in range(17):
            lines.append((f"Body sentence number {i} continues here.", False, 9.0))
        pdf = _make_pdf_sized(tmp_path, lines, "lower.pdf")
        text = "\n".join(t for t, _, _ in lines)
        out = detect_section_hierarchy(pdf, 1, text)
        assert "## entry" not in out
        assert 'entry "Fact-checking" reference line for the wiki' in out


class TestDet31ChromeClusterAndCiteVeto:
    """det31: masthead label blocks (label BESIDE text) are chrome
    clusters; running heads over citation blocks are chrome; lowercase
    decks are never titles."""

    def test_masthead_label_block_stripped(self, tmp_path):
        """9530001d class: all-caps label left of a same-size block,
        second block line stacked beneath — the whole topic block is
        masthead chrome (horizontal arm + one-level chain)."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((30, 34.0), "THEMENTHEMA BLOCK", fontsize=11)
        page.insert_text((145, 47.0), "Kontrollierte Teststrome -", fontsize=11)
        page.insert_text((145, 60.0), "Zweite Zeile der Blocktext", fontsize=11)
        y = 110.0
        for i in range(8):
            page.insert_text((72, y), f"Body text paragraph {i} runs here.", fontsize=8.8)
            y += 13
        p = tmp_path / "masthead_label.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            "THEMENTHEMA BLOCK\nKontrollierte Teststrome -\n"
            "Zweite Zeile der Blocktext\n" + "Body text paragraph runs here.\n" * 8,
            page_num=1,
            pdf_path=str(p),
        )
        assert "THEMENTHEMA BLOCK" in meta["hf_lines"]
        assert "Kontrollierte Teststrome -" in meta["hf_lines"]
        assert "Zweite Zeile der Blocktext" in meta["hf_lines"]

    def test_cite_block_running_head_stripped(self, tmp_path):
        """e71f5c59 p6 class: a Title-Case display line above a
        hanging-indent citation block is a references-page running
        head, not the page title."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((210, 70.0), "Evaluate the Role of Citations", fontsize=11)
        page.insert_text((72, 94.0), "[14].", fontsize=8)
        page.insert_text((99, 94.0), "Author A (2014) work title here.", fontsize=8)
        y = 140.0
        for i in range(8):
            page.insert_text((72, y), f"Citation body line {i} of the list.", fontsize=8)
            y += 12
        p = tmp_path / "cite_head.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            "Evaluate the Role of Citations\n[14]. "
            "Author A (2014) work title here.\n"
            + "Citation body line of the list.\n" * 8,
            page_num=1,
            pdf_path=str(p),
        )
        assert "Evaluate the Role of Citations" in meta["hf_lines"]

    def test_lowercase_deck_stripped(self, tmp_path):
        """0b045ab8 class: lowercase-initial display lines are kickers/
        decks, never page titles — masthead arm owns them on every
        path (corpus census: zero lowercase-initial GT headings)."""
        import fitz

        from scientific_parser.routing.page_metadata import extract_page_metadata

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((60, 44.0), "focus on testing", fontsize=13)
        y = 90.0
        for i in range(8):
            page.insert_text((72, y), f"Body content line {i} flows on.", fontsize=9)
            y += 14
        p = tmp_path / "lowercase_deck.pdf"
        doc.save(str(p))
        doc.close()

        meta = extract_page_metadata(
            "focus on testing\n" + "Body content line flows on.\n" * 8,
            page_num=1,
            pdf_path=str(p),
        )
        assert "focus on testing" in meta["hf_lines"]
