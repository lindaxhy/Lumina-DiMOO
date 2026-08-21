#!/usr/bin/env python3
"""DreamBooth fine-tuning for Lumina-DiMOO's discrete diffusion objective.

This keeps the upstream FSDP trainer and replaces its token-global cross entropy
with a per-example weighted loss.  Instance examples have weight 1 and class
examples have ``prior_loss_weight``.  With balanced instance/class records this
is the DreamBooth prior-preservation objective, applied to masked VQ tokens.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import LLaDAForMultiModalGeneration  # noqa: E402
from train.train import ItemProcessor as UpstreamItemProcessor  # noqa: E402
from train.train import Solver as UpstreamSolver  # noqa: E402


class WeightedLabels(list):
    """List-like labels carrying a per-example DreamBooth loss weight."""

    def __init__(self, values: Iterable[int], loss_weight: float) -> None:
        super().__init__(values)
        self.loss_weight = float(loss_weight)


class DreamBoothItemProcessor(UpstreamItemProcessor):
    def process_item(self, data_item: dict, training_mode: bool = False):
        tokens, labels = super().process_item(data_item, training_mode=training_mode)
        weight = data_item.get("loss_weight", 1.0)
        if weight <= 0:
            raise ValueError(f"loss_weight must be positive, got {weight}")
        return tokens, WeightedLabels(labels, weight)


class DreamBoothLLaDAForMultiModalGeneration(LLaDAForMultiModalGeneration):
    """Lumina-DiMOO with weighted, per-example masked-token CE."""

    def get_checkpointing_wrap_module_list(self):
        """Expose transformer blocks expected by the upstream FSDP solver."""
        return list(self.model.transformer.blocks)

    def forward(
        self,
        input_ids=None,
        labels=None,
        infer=False,
        use_cache=False,
        to_compute_mask=None,
        cat="",
        **kwargs,
    ):
        if infer:
            return super().forward(
                input_ids=input_ids,
                labels=labels,
                infer=True,
                use_cache=use_cache,
                to_compute_mask=to_compute_mask,
                cat=cat,
                **kwargs,
            )

        weights = torch.tensor(
            [getattr(label, "loss_weight", 1.0) for label in labels],
            dtype=torch.float32,
            device=self.device,
        )
        max_tokens = max(len(example) for example in input_ids)
        lengths = [len(example) for example in input_ids]
        padded_inputs = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_tensor = torch.tensor(padded_inputs, dtype=torch.int64, device=self.device)
        valid = torch.zeros(len(lengths), max_tokens, dtype=torch.bool, device=self.device)
        for row, length in enumerate(lengths):
            valid[row, :length] = True
        attention_bias = (valid[:, :, None] & valid[:, None, :]).unsqueeze(1)

        output = super(LLaDAForMultiModalGeneration, self).forward(
            input_ids=input_tensor,
            attention_bias=attention_bias,
            use_cache=use_cache,
            to_compute_mask=to_compute_mask,
            cat=cat,
        )
        padded_labels = [list(label) + [-100] * (max_tokens - len(label)) for label in labels]
        label_tensor = torch.tensor(padded_labels, dtype=torch.int64, device=self.device)
        token_loss = F.cross_entropy(
            output.logits.transpose(1, 2), label_tensor, ignore_index=-100, reduction="none"
        )
        predicted = label_tensor.ne(-100)
        per_example = (token_loss * predicted).sum(dim=1) / predicted.sum(dim=1).clamp_min(1)
        # Average over examples, not over the sum of weights.  Dividing by the
        # weight sum would cancel prior_loss_weight entirely for batch size 1.
        return weighted_batch_mean(per_example, weights)


def weighted_batch_mean(per_example: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Apply per-example DreamBooth weights without renormalizing them away."""
    if per_example.ndim != 1 or weights.shape != per_example.shape:
        raise ValueError(
            f"expected matching 1D loss/weight tensors, got {per_example.shape} and {weights.shape}"
        )
    return (per_example * weights).mean()


class Solver(UpstreamSolver):
    def _model_func(self, init_from: str):
        tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=True)
        model = DreamBoothLLaDAForMultiModalGeneration.from_pretrained(
            init_from, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        model.model.set_activation_checkpointing("whole_layer")
        return model, tokenizer

    def _item_processor_func(self, tokenizer=None, max_len=None):
        return DreamBoothItemProcessor(tokenizer, max_len)


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    Solver(args).run()
