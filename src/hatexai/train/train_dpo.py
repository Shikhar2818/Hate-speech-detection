from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from hatexai.data.dataset import collate_batch
from hatexai.data.preprocess import normalize_text
from hatexai.data.public_sources import file_digest
from hatexai.data.public_splits import canonical_text, text_matches
from hatexai.models.llm_policy import ExplanationPolicy, completion_logprobs
from hatexai.reward.dpo import dpo_loss
from hatexai.utils import ensure_dir, read_jsonl, resolve_device, set_seed, write_json


def _sequence_logprob(policy: ExplanationPolicy, sources: list[str], targets: list[str],
                      **lengths) -> torch.Tensor:
    """Length-normalized completion log-probability used by the rationale objective."""
    totals, counts = completion_logprobs(policy, sources, targets, **lengths)
    return totals / counts.clamp(min=1)


def load_reviewed_preferences(path: Path, *, require_prompt=False) -> list[dict[str, str]]:
    records = []
    ids = set()
    fields = ("id", "text", "chosen_explanation", "rejected_explanation", "reviewer_id")
    for row in read_jsonl(path):
        if row.get("review_status") != "accepted" or any(
            not isinstance(row.get(field), str) or not row[field].strip() for field in fields
        ):
            raise ValueError(f"Preference {row.get('id')} needs accepted human review and two rationales")
        pair_id = row["id"].strip()
        if pair_id in ids or row["chosen_explanation"].strip() == row["rejected_explanation"].strip():
            raise ValueError(f"Preference {pair_id} repeats an ID or has identical rationales")
        context = row.get("context", "")
        if not isinstance(context, str):
            raise ValueError(f"Preference {pair_id} has invalid context")
        prompt = row.get("prompt", "")
        if not isinstance(prompt, str) or (require_prompt and not prompt.strip()):
            raise ValueError(f"Preference {pair_id} needs a reviewed classifier-evidence prompt")
        ids.add(pair_id)
        records.append({
            "id": pair_id, "prompt": prompt.strip(),
            "text": normalize_text(row["text"]),
            "context": context,
            "chosen_explanation": row["chosen_explanation"].strip(),
            "rejected_explanation": row["rejected_explanation"].strip(),
        })
    if not records:
        raise ValueError("DPO requires a non-empty file of human-reviewed preference pairs")
    return records


def verify_preference_isolation(train, test):
    if {row["id"] for row in train} & {row["id"] for row in test}:
        raise ValueError("Preference splits share IDs")
    left, right = ({canonical_text(row["text"]) for row in split} for split in (train, test))
    if left & right:
        raise ValueError("Preference splits share posts")
    texts = dict(enumerate([*sorted(left), *sorted(right)]))
    if any((i < len(left)) != (j < len(left)) for i, j in text_matches(texts)):
        raise ValueError("Preference splits share near-duplicate posts")


def train_dpo(config: dict[str, Any]) -> ExplanationPolicy:
    settings = config["dpo"]
    preference_path = Path(settings["preferences_path"])
    source_digest = file_digest(preference_path)
    require_prompt = bool(settings.get("require_prompt"))
    dataset = load_reviewed_preferences(preference_path, require_prompt=require_prompt)
    if settings.get("evaluation_path"):
        heldout = load_reviewed_preferences(Path(settings["evaluation_path"]), require_prompt=require_prompt)
        verify_preference_isolation(dataset, heldout)
    epochs = int(settings["epochs"])
    lengths = {
        "max_source_length": int(settings.get("max_source_length", 384)),
        "max_target_length": int(settings.get("max_target_length", 128)),
    }
    if epochs < 1 or lengths["max_source_length"] < 1 or lengths["max_target_length"] < 2:
        raise ValueError("DPO requires at least one epoch and valid source and target lengths")
    out_dir = Path(settings.get("output_dir", config["output_dir"])) / "checkpoints" / "dpo_policy"
    if out_dir.exists():
        raise FileExistsError(f"Choose a fresh experiment output directory; {out_dir} already exists")

    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    policy = ExplanationPolicy(settings["policy_model"], device)
    reference = ExplanationPolicy(settings["policy_model"], device)
    reference.model.eval()
    for param in reference.model.parameters():
        param.requires_grad = False

    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True, collate_fn=collate_batch)
    optimizer = AdamW(policy.trainable_parameters(), lr=float(settings["learning_rate"]))
    beta = float(settings["beta"])

    policy.model.train()
    history = []
    for epoch in range(epochs):
        total_loss, pair_count = 0.0, 0
        progress = tqdm(loader, desc=f"dpo epoch {epoch+1}")
        for batch in progress:
            sources = [
                prompt or policy.prompt_for(text, context)
                for prompt, text, context in zip(batch["prompt"], batch["text"], batch["context"])
            ]
            chosen = batch["chosen_explanation"]
            rejected = batch["rejected_explanation"]
            policy_chosen = _sequence_logprob(policy, sources, chosen, **lengths)
            policy_rejected = _sequence_logprob(policy, sources, rejected, **lengths)
            with torch.no_grad():
                ref_chosen = _sequence_logprob(reference, sources, chosen, **lengths)
                ref_rejected = _sequence_logprob(reference, sources, rejected, **lengths)
            loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)
            if not torch.isfinite(loss):
                raise ValueError("DPO loss became non-finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pair_count += len(sources)
            total_loss += float(loss.item()) * len(sources)
            progress.set_postfix(loss=float(loss.item()))
        history.append({"epoch": epoch + 1, "pairs": pair_count, "mean_loss": total_loss / pair_count})

    with TemporaryDirectory(dir=ensure_dir(out_dir.parent)) as temp:
        checkpoint = Path(temp) / "policy"
        policy.model.save_pretrained(checkpoint)
        policy.tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "training_history.json", {
            "settings": settings, "seed": config["seed"], "device": str(device),
            "preferences_sha256": source_digest, "history": history,
        })
        checkpoint.rename(out_dir.resolve())
    return policy


@torch.inference_mode()
def evaluate_dpo(config):
    settings = config["dpo"]
    output = Path(settings.get("output_dir", config["output_dir"]))
    checkpoint = output / "checkpoints" / "dpo_policy"
    from hatexai.data.public_sources import load_json

    metadata = load_json(checkpoint / "training_history.json")
    train_path, test_path = Path(settings["preferences_path"]), Path(settings["evaluation_path"])
    if metadata["preferences_sha256"] != file_digest(train_path):
        raise ValueError("Preference training digest does not match the DPO checkpoint")
    require_prompt = bool(settings.get("require_prompt"))
    train = load_reviewed_preferences(train_path, require_prompt=require_prompt)
    test = load_reviewed_preferences(test_path, require_prompt=require_prompt)
    verify_preference_isolation(train, test)
    device = resolve_device(config.get("device", "auto"))
    lengths = {k: int(settings.get(k, default)) for k, default in
               (("max_source_length", 384), ("max_target_length", 128))}
    report = {"n": len(test), "test_sha256": file_digest(test_path),
              "metric": "Held-out human preference agreement by completion score"}
    for name, path in (("before", metadata["settings"]["policy_model"]), ("after", str(checkpoint))):
        policy = ExplanationPolicy(path, device, local_only=True)
        margins = []
        for row in test:
            prompt = row["prompt"] or policy.prompt_for(row["text"], row["context"])
            scores = _sequence_logprob(policy, [prompt, prompt],
                                       [row["chosen_explanation"], row["rejected_explanation"]], **lengths)
            margins.append(float(scores[0] - scores[1]))
        report[name] = {"win_rate": sum(m > 0 for m in margins) / len(margins),
                        "mean_margin": sum(margins) / len(margins),
                        "pairs": [{"id": row["id"], "margin": m} for row, m in zip(test, margins)]}
        del policy
    write_json(output / "eval" / "dpo_comparison.json", report)
    return report
