from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPModel, CLIPProcessor, AutoTokenizer


class GatedFusion(nn.Module):
    def __init__(self, text_dim: int, vision_dim: int, context_dim: int, hidden_size: int):
        super().__init__()
        self.proj_text = nn.Linear(text_dim, hidden_size)
        self.proj_vision = nn.Linear(vision_dim, hidden_size)
        self.proj_context = nn.Linear(context_dim, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, text: torch.Tensor, vision: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t = self.proj_text(text)
        v = self.proj_vision(vision)
        c = self.proj_context(context)
        weights = self.gate(torch.cat([t, v, c], dim=-1))
        return weights[:, 0:1] * t + weights[:, 1:2] * v + weights[:, 2:3] * c


class MultimodalHateDetector(nn.Module):
    """Prediction model f_theta(x_i) -> y_i for x_i = {T_i, V_i, C_i}."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model_cfg = config["model"]
        self.use_vision = bool(model_cfg.get("use_vision", True))
        self.use_context = bool(model_cfg.get("use_context", True))
        self.text_encoder_name = model_cfg["text_encoder"]
        self.vision_encoder_name = model_cfg["vision_encoder"]
        hidden_size = int(model_cfg["hidden_size"])
        dropout = float(model_cfg["dropout"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size

        if self.use_vision:
            self.clip = CLIPModel.from_pretrained(self.vision_encoder_name)
            self.image_processor = CLIPProcessor.from_pretrained(self.vision_encoder_name)
            vision_dim = self.clip.config.projection_dim
            for param in self.clip.parameters():
                param.requires_grad = False
        else:
            self.clip = None
            self.image_processor = None
            vision_dim = hidden_size

        context_dim = text_dim
        self.fusion = GatedFusion(text_dim, vision_dim, context_dim, hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, int(model_cfg["num_labels"])),
        )

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        cls = outputs.last_hidden_state[:, 0]
        return cls, outputs.attentions

    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]

    def encode_vision(self, pixel_values: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if not self.use_vision or pixel_values is None:
            return torch.zeros(batch_size, self.fusion.proj_vision.in_features, device=device)
        with torch.no_grad():
            return self.clip.get_image_features(pixel_values=pixel_values)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        context_ids: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_vec, attentions = self.encode_text(input_ids, attention_mask)
        if self.use_context and context_ids is not None:
            context_vec = self.encode_context(context_ids, context_mask)
        else:
            context_vec = text_vec
        vision_vec = self.encode_vision(pixel_values, input_ids.size(0), input_ids.device)
        fused = self.fusion(text_vec, vision_vec, context_vec)
        logits = self.classifier(fused)
        result = {"logits": logits, "fused": fused, "attentions": attentions}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
