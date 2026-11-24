from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatexai.data.public_pipeline import build_public_dataset
from hatexai.utils import load_config, read_jsonl, write_jsonl


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
    parser = argparse.ArgumentParser(description="Prepare public hate-speech benchmark splits")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", help="Benchmark profile name")
    source.add_argument("--input", help="Legacy single-file conversion")
    parser.add_argument("--config", default=str(ROOT / "configs" / "public_data.yaml"))
    parser.add_argument("--authorized-sources", action="store_true",
                        help="Confirm authorized local copies for restricted corpora")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.input:
        convert_records(Path(args.input), Path(args.output))
        return
    profiles = load_config(args.config)
    if args.profile not in profiles:
        parser.error(f"Unknown profile {args.profile!r}; choose from {', '.join(profiles)}")
    try:
        manifest = build_public_dataset(profiles[args.profile], Path(args.output),
                                        authorized_sources=args.authorized_sources)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(", ".join(f"{name}={stats['total']}" for name, stats in manifest["splits"].items()))
    print(f"wrote benchmark manifest to {Path(args.output) / 'manifest.json'}")


if __name__ == "__main__":
    main()
