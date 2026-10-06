from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from hatexai.data.dataset import MultimodalHateDataset, collate_batch
from hatexai.models.llm_policy import ExplanationPolicy
from hatexai.reward.dpo import dpo_loss
from hatexai.utils import ensure_dir, resolve_device, set_seed


def _sequence_logprob(policy: ExplanationPolicy, sources: list[str], targets: list[str]) -> torch.Tensor:
    enc_src = policy.tokenizer(
        sources, padding=True, truncation=True, max_length=384, return_tensors="pt"
    ).to(policy.device)
    enc_tgt = policy.tokenizer(
        targets, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(policy.device)
    labels = enc_tgt["input_ids"].clone()
    pad_id = policy.tokenizer.pad_token_id
    labels[labels == pad_id] = -100
    outputs = policy.model(**enc_src, labels=labels)
    logp = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    gathered = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = labels != -100
    return (gathered * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def train_dpo(config: dict[str, Any]) -> ExplanationPolicy:
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    policy = ExplanationPolicy(config["dpo"]["policy_model"], device)
    reference = ExplanationPolicy(config["dpo"]["policy_model"], device)
    reference.model.eval()
    for param in reference.model.parameters():
        param.requires_grad = False

    dataset = MultimodalHateDataset(config["data"]["train_path"], max_length=config["data"]["max_text_length"])
    loader = DataLoader(dataset, batch_size=int(config["dpo"]["batch_size"]), shuffle=True, collate_fn=collate_batch)
    optimizer = AdamW(policy.trainable_parameters(), lr=float(config["dpo"]["learning_rate"]))
    beta = float(config["dpo"]["beta"])

    policy.model.train()
    for epoch in range(int(config["dpo"]["epochs"])):
        progress = tqdm(loader, desc=f"dpo epoch {epoch+1}")
        for batch in progress:
            sources = [
                policy.prompt_for(text, context)
                for text, context in zip(batch["text"], batch["context"])
            ]
            chosen = batch["chosen_explanation"]
            rejected = batch["rejected_explanation"]
            if not chosen[0]:
                continue
            policy_chosen = _sequence_logprob(policy, sources, chosen)
            policy_rejected = _sequence_logprob(policy, sources, rejected)
            with torch.no_grad():
                ref_chosen = _sequence_logprob(reference, sources, chosen)
                ref_rejected = _sequence_logprob(reference, sources, rejected)
            loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=float(loss.item()))

    out_dir = ensure_dir(Path(config["output_dir"]) / "checkpoints" / "dpo_policy")
    policy.model.save_pretrained(out_dir)
    policy.tokenizer.save_pretrained(out_dir)
    return policy
