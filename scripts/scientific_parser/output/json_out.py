"""JSON output adapter — emit DocumentIR as structured JSON."""

from __future__ import annotations
import json
import sys

if "scripts" not in sys.path:
    sys.path.insert(0, "scripts")


def document_to_json(doc, pretty: bool = True) -> str:
    """Serialize DocumentIR (or compatible) to JSON string."""
    data = doc.to_dict() if hasattr(doc, "to_dict") else dict(doc)
    return json.dumps(
        data, ensure_ascii=False, indent=2 if pretty else None, default=str
    )


def write_json(doc, path) -> None:
    """Write DocumentIR JSON to file path."""
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document_to_json(doc, pretty=True), encoding="utf-8")


def entities_to_json(entities: list) -> str:
    """Serialize list of Entity objects to JSON."""
    data = []
    for e in entities:
        if hasattr(e, "to_dict"):
            data.append(e.to_dict())
        elif hasattr(e, "__dict__"):
            d = dict(e.__dict__)
            # Drop private/internal
            data.append({k: v for k, v in d.items() if not k.startswith("_")})
        else:
            data.append(str(e))
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


__all__ = ["document_to_json", "write_json", "entities_to_json"]
