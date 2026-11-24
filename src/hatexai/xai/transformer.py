"""Explain the prediction for a fixed, truncated XLM-R token sequence."""

import numpy as np
import torch


def explain_tokens(classifier, text, label, method, settings):
    encoded = classifier.encode(text, return_special_tokens_mask=True, return_offsets_mapping=True)
    ids, attention = encoded["input_ids"], encoded["attention_mask"]
    positions = (attention[0].bool() & ~encoded["special_tokens_mask"][0].bool()).nonzero().flatten()
    offsets = encoded["offset_mapping"][0, positions].tolist()
    tokens = [text[start:end] for start, end in offsets]
    if not tokens:
        return [], [], {}
    model = classifier.model
    mask_id = classifier.tokenizer.mask_token_id

    @torch.inference_mode()
    def probabilities(visible):
        visible = torch.as_tensor(visible, device=ids.device, dtype=torch.bool)
        hidden = ids.expand(len(visible), -1).clone()
        hidden[:, positions] = torch.where(visible, ids[0, positions], mask_id)
        parts = []
        for chunk in hidden.split(classifier.batch_size):
            logits = model(input_ids=chunk, attention_mask=attention.expand(len(chunk), -1)).logits
            parts.append(logits.float().softmax(-1)[:, label].cpu().numpy())
        return np.concatenate(parts)

    if method == "integrated_gradients":
        from captum.attr import IntegratedGradients

        reference = ids.clone()
        reference[0, positions] = mask_id
        embeddings = model.get_input_embeddings()

        def class_logit(vectors, mask):
            return model(inputs_embeds=vectors, attention_mask=mask).logits[:, label]

        attribution = IntegratedGradients(class_logit).attribute(
            embeddings(ids).detach(), baselines=embeddings(reference).detach(),
            additional_forward_args=attention, n_steps=int(settings.get("ig_steps", 16)),
            internal_batch_size=classifier.batch_size,
        )
        scores = attribution[0, positions].sum(-1).detach().float().cpu().numpy()
    elif method == "shap":
        import shap

        reference = np.zeros((1, len(tokens)), dtype=int)
        explainer = shap.Explainer(probabilities, reference, algorithm="permutation",
                                   seed=int(settings.get("shap_seed", 42)))
        budget = max(int(settings.get("shap_evals", 65)), 2 * len(tokens) + 1)
        scores = explainer(np.ones_like(reference), max_evals=budget, silent=True).values[0]
    else:
        raise ValueError(f"Unknown transformer attribution method: {method}")

    # Compare equally sized masks on the same exact input and predicted class.
    k = min(int(settings.get("top_k_tokens", 8)), len(tokens))
    important = np.argsort(-scores, kind="stable")[:k]
    rng = np.random.default_rng(42)
    visible = np.ones((6, len(tokens)), dtype=bool)
    visible[1, important] = False
    visible[2] = ~visible[1]
    for row in visible[3:]:
        row[rng.choice(len(tokens), k, replace=False)] = False
    full, removed, kept, *random = probabilities(visible).tolist()
    return tokens, scores.astype(float).tolist(), {
        "comprehensiveness": full - removed, "sufficiency": full - kept,
        "random_mask_drop": full - float(np.mean(random)),
    }
