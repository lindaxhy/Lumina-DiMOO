#!/usr/bin/env python3
"""DreamBooth-LoRA for Lumina-DiMOO's discrete diffusion objective."""

from __future__ import annotations

import os
import sys
from types import MethodType

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_dreambooth import (  # noqa: E402
    DreamBoothItemProcessor,
    DreamBoothLLaDAForMultiModalGeneration,
    Solver as DreamBoothSolver,
)


DEFAULT_TARGETS = "q_proj,k_proj,v_proj,attn_out,ff_proj,up_proj,ff_out"


def _trainable_parameters(self):
    return {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}


class Solver(DreamBoothSolver):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        parser.add_argument("--lora_rank", type=int, default=16)
        parser.add_argument("--lora_alpha", type=int, default=16)
        parser.add_argument("--lora_dropout", type=float, default=0.0)
        parser.add_argument("--lora_targets", default=DEFAULT_TARGETS)
        return parser

    def _model_func(self, init_from: str):
        tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=True)
        base = DreamBoothLLaDAForMultiModalGeneration.from_pretrained(
            init_from, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        base.model.set_activation_checkpointing("whole_layer")
        config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            target_modules=[item.strip() for item in self.args.lora_targets.split(",") if item.strip()],
        )
        model = get_peft_model(base, config)
        # Frozen base parameters are replicated.  FSDP shards only the LoRA
        # parameters, avoiding mixed bf16/fp32 flattening within PEFT layers.
        model._fsdp_ignore_frozen_params = True
        model._cast_all_frozen_to_mixed_precision = True
        model.get_trainable_params = MethodType(_trainable_parameters, model)
        model.print_trainable_parameters()
        return model, tokenizer

    def _item_processor_func(self, tokenizer=None, max_len=None):
        return DreamBoothItemProcessor(tokenizer, max_len)


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    Solver(args).run()
