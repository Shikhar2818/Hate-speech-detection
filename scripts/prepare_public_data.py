from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatexai.utils import read_jsonl, write_jsonl


def convert_records(in_path: Path, out_path: Path) -> None:
    """Convert HateXplain/HASOC-style records to the project schema."""
    if in_path.suffix == ".json":
        raw = json.loads(in_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = list(raw.values()) if raw and isinstance(next(iter(raw.values())), dict) else [raw]
    else:
        raw = read_jsonl(in_path)

    converted = []
    for i, item in enumerate(raw):
        label = item.get("label", item.get("gold_label", item.get("task_1", "normal")))
        y = 1 if str(label).lower() in {"hate", "hateful", "offensive", "hofs", "1"} else 0
        text = item.get("text") or " ".join(item.get("post_tokens", []))
        converted.append(
            {
                "id": str(item.get("id", f"pub_{i}")),
                "text": text,
                "label": y,
                "language": item.get("language", "en"),
                "code_mixed": bool(item.get("code_mixed", False)),
                "context": item.get("context", ""),
                "image_path": item.get("image_path"),
                "chosen_explanation": item.get("chosen_explanation", ""),
                "rejected_explanation": item.get("rejected_explanation", "The post is unlabeled noise."),
            }
        )
    write_jsonl(out_path, converted)
    print(f"wrote {len(converted)} records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert public hate-speech files to JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_records(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
