#!/usr/bin/env python3
"""Compare RapidLayout models — find the best for English scientific documents.

Tests each available model on 2 bench PDFs (tables + old_scans_math) and
reports: total regions, equation regions, table regions, time per page.
"""

import sys, os, time, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PDF_OCR_NO_SKILL_VENV"] = "1"


def render_page(pdf_path, page_num):
    import fitz

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=150)
    img_path = tempfile.mktemp(suffix=".png")
    pix.save(img_path)
    doc.close()
    return img_path


# Models to test (from RapidLayout docs/models.md)
MODELS = [
    (
        "pp_layout_cdla",
        "Chinese default",
        [
            "text",
            "title",
            "figure",
            "figure_caption",
            "table",
            "table_caption",
            "header",
            "footer",
            "reference",
            "equation",
        ],
    ),
    (
        "pp_doc_layoutv3",
        "PP-DocLayoutV3 25cls",
        [
            "abstract",
            "algorithm",
            "aside_text",
            "chart",
            "content",
            "display_formula",
            "doc_title",
            "figure_title",
            "footer",
            "footer_image",
            "footnote",
            "formula_number",
            "header",
            "header_image",
            "image",
            "inline_formula",
            "number",
            "paragraph_title",
            "reference",
            "reference_content",
            "seal",
            "table",
            "text",
            "vertical_text",
            "vision_footnote",
        ],
    ),
    ("pp_layout_publaynet", "PubLayNet English", ["text", "title", "list", "table", "figure"]),
    (
        "yolov8n_layout_paper",
        "YOLOv8 papers",
        [
            "Text",
            "Title",
            "Header",
            "Footer",
            "Figure",
            "Table",
            "Toc",
            "Figure caption",
            "Table caption",
        ],
    ),
    (
        "yolov8n_layout_general6",
        "YOLOv8 general",
        ["Text", "Title", "Figure", "Table", "Caption", "Equation"],
    ),
]

PDFS = [
    (
        "tables",
        "bench/olmOCR-bench/bench_data/pdfs/tables/0b100ba02aedbdb366a79ac387b4d6bf1774_pg4.pdf",
        1,
    ),
    ("math", "bench/olmOCR-bench/bench_data/pdfs/old_scans_math/1_pg10.pdf", 1),
]

print("=" * 90)
print("RapidLayout Model Comparison — English Scientific Documents")
print("=" * 90)

for model_type, desc, expected_labels in MODELS:
    print(f"\n{'─' * 90}")
    print(f"MODEL: {model_type} ({desc})")
    print(f"Labels: {expected_labels}")
    print(f"{'─' * 90}")

    try:
        from rapid_layout import RapidLayout

        t0 = time.time()
        engine = RapidLayout(model_type=model_type)
        load_time = time.time() - t0
        print(f"  Load: {load_time:.1f}s")
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        continue

    for cat, pdf_path, page_num in PDFS:
        if not os.path.exists(pdf_path):
            print(f"  {cat}: PDF not found")
            continue

        img_path = render_page(pdf_path, page_num)
        try:
            t1 = time.time()
            result = engine(img_path)
            elapsed = time.time() - t1

            boxes = (
                getattr(result, "boxes", None)
                or (result.get("boxes") if isinstance(result, dict) else None)
                or []
            )
            labels = (
                getattr(result, "labels", None)
                or (result.get("labels") if isinstance(result, dict) else None)
                or []
            )
            scores = (
                getattr(result, "scores", None)
                or (result.get("scores") if isinstance(result, dict) else None)
                or []
            )

            # Count by type
            from collections import Counter

            label_counts = Counter(labels)

            # Count equation/table regions
            eq_labels = [
                l
                for l in labels
                if any(
                    k in l.lower()
                    for k in ("equation", "formula", "display_formula", "inline_formula")
                )
            ]
            tbl_labels = [l for l in labels if "table" in l.lower()]

            print(f"\n  {cat}: {len(boxes)} regions in {elapsed:.1f}s")
            print(f"    Types: {dict(label_counts)}")
            print(f"    Equations: {len(eq_labels)}  Tables: {len(tbl_labels)}")
            if boxes:
                print(f"    First 5 bboxes: {boxes[:5]}")
                print(f"    First 5 scores: {[f'{s:.2f}' for s in scores[:5]]}")
        except Exception as e:
            print(f"  {cat}: DETECTION FAILED: {e}")
        finally:
            os.unlink(img_path)

print(f"\n{'=' * 90}")
print("DONE — pick the model with most equation + table regions detected.")
