#!/usr/bin/env python3
"""Run one real weighted-loss forward/backward without launching full FSDP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from train_dreambooth import DreamBoothItemProcessor, DreamBoothLLaDAForMultiModalGeneration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = json.loads(args.records.read_text())
    instance = next(item for item in records if item["sample_kind"] == "instance")
    prior = next(item for item in records if item["sample_kind"] == "class")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = DreamBoothItemProcessor(tokenizer, max_len=512)
    examples, labels = zip(processor.process_item(instance, True), processor.process_item(prior, True))

    model = DreamBoothLLaDAForMultiModalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device(args.device)}
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # The smoke step verifies that the custom weighted objective propagates a
    # finite gradient. Full training intentionally unfreezes every parameter.
    for parameter in model.model.transformer.ff_out.parameters():
        parameter.requires_grad_(True)
    model.train()
    loss = model(examples, labels)
    loss.backward()
    grad_norm = torch.linalg.vector_norm(
        torch.stack([
            parameter.grad.float().norm()
            for parameter in model.model.transformer.ff_out.parameters()
            if parameter.grad is not None
        ])
    )
    result = {"loss": float(loss.detach()), "output_head_grad_norm": float(grad_norm)}
    if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(f"non-finite or zero-gradient smoke result: {result}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
