#!/usr/bin/env python3
"""Train deterministic SDXL or Z-Image DreamBench subject shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dreambench import load_subjects


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def slug(value: str) -> str:
    return value.replace(" ", "_")


def training_config(args: argparse.Namespace, subject: str, class_name: str) -> dict:
    alpha = args.lora_rank if args.model_type == "sdxl" else args.lora_alpha or args.lora_rank
    return {
        "protocol_version": 1,
        "model_type": args.model_type,
        "base_model": str(args.base_model),
        "subject": subject,
        "class_name": class_name,
        "instance_prompt": f"a sks {class_name}",
        "class_prompt": f"a {class_name}",
        "resolution": args.resolution,
        "max_train_steps": args.steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": alpha,
        "lora_scale": alpha / args.lora_rank,
        "num_class_images": args.num_class_images,
        "prior_loss_weight": args.prior_loss_weight,
        "post_validation_prompt": f"a sks {class_name}",
        "post_validation_seed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("sdxl", "z_image"), required=True)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--class-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--diffusers-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--num-class-images", type=int, default=20)
    parser.add_argument("--prior-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index/count")
    if min(args.steps, args.resolution, args.lora_rank, args.num_class_images) < 1:
        parser.error("steps, resolution, rank, and class image count must be positive")
    if args.lora_alpha is not None and args.lora_alpha < 1:
        parser.error("--lora-alpha must be positive")
    if args.model_type == "sdxl" and args.lora_alpha not in (None, args.lora_rank):
        parser.error("the official SDXL trainer fixes LoRA alpha equal to rank")

    subjects = list(load_subjects(args.classes_file).items())[args.shard_index :: args.num_shards]
    for subject, class_name in subjects:
        output_dir = args.output_root / subject
        adapter_path = output_dir / "pytorch_lora_weights.safetensors"
        config_path = output_dir / "dreambench_training_config.json"
        pending_config_path = output_dir / "dreambench_training_config.pending.json"
        validation_path = output_dir / "validation" / subject / "00_0.png"
        expected = training_config(args, subject, class_name)
        if adapter_path.exists():
            if config_path.exists() and json.loads(config_path.read_text()) != expected:
                raise RuntimeError(
                    f"{subject}: existing adapter metadata does not match this protocol; use a new output root"
                )
            if config_path.exists() and validation_path.exists():
                print(f"skip completed {subject}", flush=True)
                continue
            if not config_path.exists() and (
                not pending_config_path.exists()
                or json.loads(pending_config_path.read_text()) != expected
            ):
                raise RuntimeError(
                    f"{subject}: existing adapter has no matching protocol metadata; use a new output root"
                )

        if not adapter_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            pending_config_path.write_text(json.dumps(expected, indent=2) + "\n")
            class_dir = args.class_root / args.model_type / slug(class_name)
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON_BIN": args.python,
                    "DIFFUSERS_ROOT": str(args.diffusers_root),
                    "MODEL": str(args.base_model),
                    "INSTANCE_DIR": str(args.dataset_root / subject),
                    "CLASS_DIR": str(class_dir),
                    "INSTANCE_PROMPT": expected["instance_prompt"],
                    "CLASS_PROMPT": expected["class_prompt"],
                    "OUTPUT_DIR": str(output_dir),
                    "RESOLUTION": str(args.resolution),
                    "MAX_TRAIN_STEPS": str(args.steps),
                    "LR": str(args.learning_rate),
                    "NUM_CLASS_IMAGES": str(args.num_class_images),
                    "LORA_RANK": str(args.lora_rank),
                    "LORA_ALPHA": str(expected["lora_alpha"]),
                    "PRIOR_LOSS_WEIGHT": str(args.prior_loss_weight),
                }
            )
            script = "train_sdxl_dreambooth.sh" if args.model_type == "sdxl" else "train_z_image_dreambooth.sh"
            run(["bash", f"dreambooth/{script}"], env)
            if not adapter_path.exists():
                raise FileNotFoundError(f"trainer completed without expected adapter: {adapter_path}")

        validation_manifest = output_dir / "dreambench_validation_manifest.jsonl"
        validation_manifest.write_text(
            json.dumps(
                {
                    "subject": subject,
                    "class_name": class_name,
                    "prompt_id": 0,
                    "prompt": expected["post_validation_prompt"],
                    "seed": expected["post_validation_seed"],
                }
            )
            + "\n"
        )
        run(
            [
                args.python,
                "dreambooth/generate_dreambench.py",
                "--model-type", args.model_type,
                "--manifest", str(validation_manifest),
                "--checkpoint-template", str(output_dir),
                "--base-model", str(args.base_model),
                "--resolution", str(args.resolution),
                "--output-dir", str(output_dir / "validation"),
            ]
        )
        config_path.write_text(json.dumps(expected, indent=2) + "\n")
        pending_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
