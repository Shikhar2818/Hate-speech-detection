"""Draw fixed stratified subsets for a matched CPU experiment."""

import argparse
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from hatexai.data.public_pipeline import statistics
from hatexai.data.public_sources import file_digest, load_json
from hatexai.utils import load_config, read_jsonl, write_json, write_jsonl
from scripts.run_pipeline import verify_benchmark_split


def prepare(config, root):
    if root.exists():
        raise FileExistsError(f"Choose a new output directory: {root}")
    source = Path(config["data"]["manifest_path"])
    manifest = load_json(source)
    manifest["pilot_source_sha256"] = file_digest(source)
    for split, n in {"train": 768, "val": 320, "test": 768}.items():
        verify_benchmark_split(config, manifest, split)
        rows = read_jsonl(config["data"][f"{split}_path"])
        rows, _ = train_test_split(rows, train_size=n, random_state=config["seed"],
                                   stratify=[(r["language"], r["label"]) for r in rows])
        path = root / f"{split}.jsonl"
        write_jsonl(path, rows)
        config["data"][f"{split}_path"] = path.as_posix()
        manifest["splits"][split] = {**statistics(rows), "file": path.name, "sha256": file_digest(path)}
    write_json(root / "manifest.json", manifest)
    config["data"]["manifest_path"] = (root / "manifest.json").as_posix()
    config.update(output_dir=root.as_posix(), model_version="linear-xlmr-matched-pilot", device="cpu")
    config["comparison_scope"] = ("CPU pilot: fixed stratified subsets of 768 train, 320 validation and "
                                  "768 test rows. Both models used identical subsets.")
    config["transformer"].update(
        checkpoint=(root / "checkpoints/classifier").as_posix(), evaluation_dir=(root / "eval").as_posix(),
        model_version="xlmr-bilingual-cpu-pilot", batch_size=8, max_length=64, epochs=2,
        learning_rate=0.00015, train_encoder_layers=1,
    )
    write_json(root / "config.yaml", config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/xlm_roberta.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(load_config(args.config), args.output)
