"""Parquet output adapter — emit DocumentIR / entities as Parquet.

Requires pyarrow. Optional dep — graceful degradation if missing.
"""

from __future__ import annotations


def blocks_to_dataframe(doc):
    """Convert DocumentIR blocks to pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required: uv pip install pandas")
    rows = []
    for page in doc.pages:
        for block in page.blocks:
            winner = block.resolve()
            rows.append(
                {
                    "block_id": block.id,
                    "page_num": block.page_num,
                    "type": block.type,
                    "bbox_x0": block.bbox[0] if block.bbox else None,
                    "bbox_y0": block.bbox[1] if block.bbox else None,
                    "bbox_x1": block.bbox[2] if block.bbox else None,
                    "bbox_y1": block.bbox[3] if block.bbox else None,
                    "value": winner.value if winner else None,
                    "confidence": winner.confidence if winner else 0.0,
                    "evidence_count": len(block.evidence),
                    "mentions": ",".join(block.mentions),
                }
            )
    return pd.DataFrame(rows)


def write_parquet(doc, path) -> None:
    """Write all blocks to Parquet file."""
    from pathlib import Path

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("pyarrow required: uv pip install pyarrow")
    df = blocks_to_dataframe(doc)
    table = pa.Table.from_pandas(df)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out))


def entities_to_dataframe(entities: list):
    """Convert list of Entity objects to DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required")
    rows = []
    for e in entities:
        d = e.to_dict() if hasattr(e, "to_dict") else dict(e.__dict__)
        rows.append(d)
    return pd.DataFrame(rows)


__all__ = ["blocks_to_dataframe", "write_parquet", "entities_to_dataframe"]
