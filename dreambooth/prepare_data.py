#!/usr/bin/env python3
"""Pre-tokenize one DreamBench subject and build balanced DreamBooth records."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.item_processor import DimooItemProcessor  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SYSTEM_PROMPT = "Generate an image according to the text prompt."


def image_paths(root: Path) -> list[Path]:
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"no images found in {root}")
    return paths


def encode_images(processor: DimooItemProcessor, paths: list[Path], output: Path, prefix: str) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    encoded = []
    for index, path in enumerate(paths):
        record = processor.process_item("", str(path))
        destination = output / f"{prefix}_{index:04d}.pkl"
        with destination.open("wb") as handle:
            pickle.dump(record, handle)
        encoded.append({"path": str(destination.resolve()), "len": len(record["input_ids"])})
    return encoded


def balanced_records(
    instance: list[dict],
    classes: list[dict],
    instance_prompt: str,
    class_prompt: str,
    repeats: int,
    prior_loss_weight: float,
    seed: int,
) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if prior_loss_weight <= 0:
        raise ValueError("prior_loss_weight must be > 0")
    instance_records = []
    for repeat in range(repeats):
        for index, item in enumerate(instance):
            instance_records.append(
                {
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": instance_prompt,
                    "user_image": "",
                    "answer_text": "",
                    "answer_image": item["path"],
                    "answer_thinking": "",
                    "id": f"instance-{repeat}-{index}",
                    "len": item["len"],
                    "sample_kind": "instance",
                    "loss_weight": 1.0,
                }
            )

    # Equal record counts make minibatch SGD an unbiased estimator of
    # L_instance + prior_loss_weight * L_class (up to a constant scale).
    class_records = []
    for index in range(len(instance_records)):
        item = classes[index % len(classes)]
        class_records.append(
            {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": class_prompt,
                "user_image": "",
                "answer_text": "",
                "answer_image": item["path"],
                "answer_thinking": "",
                "id": f"class-{index}",
                "len": item["len"],
                "sample_kind": "class",
                "loss_weight": prior_loss_weight,
            }
        )
    records = instance_records + class_records
    random.Random(seed).shuffle(records)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-data-dir", type=Path, required=True)
    parser.add_argument("--class-data-dir", type=Path, required=True)
    parser.add_argument("--instance-prompt", required=True)
    parser.add_argument("--class-prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Alpha-VLLM/Lumina-DiMOO")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--instance-repeats", type=int, default=20)
    parser.add_argument("--prior-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = DimooItemProcessor(
        tokenizer_path=args.model,
        vq_ckpt_path=args.model,
        target_size=args.resolution,
    )
    token_dir = args.output_dir / "tokens"
    instance = encode_images(processor, image_paths(args.instance_data_dir), token_dir, "instance")
    classes = encode_images(processor, image_paths(args.class_data_dir), token_dir, "class")
    records = balanced_records(
        instance,
        classes,
        args.instance_prompt,
        args.class_prompt,
        args.instance_repeats,
        args.prior_loss_weight,
        args.seed,
    )
    records_path = args.output_dir / "records.json"
    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    config_path = args.output_dir / "data.yaml"
    config_path.write_text(f"META:\n  - path: '{records_path.resolve()}'\n")
    summary = {
        "instance_images": len(instance),
        "class_images": len(classes),
        "instance_records": sum(item["sample_kind"] == "instance" for item in records),
        "class_records": sum(item["sample_kind"] == "class" for item in records),
        "prior_loss_weight": args.prior_loss_weight,
        "resolution": args.resolution,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
