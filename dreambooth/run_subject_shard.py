#!/usr/bin/env python3
"""Train one deterministic shard of DreamBench subject adapters."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dreambench import load_subjects


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def slug(class_name: str) -> str:
    return class_name.replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("lumina", "flux"), required=True)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--class-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("runs/lumina_data"))
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--diffusers-root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--instance-repeats", type=int, default=5)
    parser.add_argument("--lumina-epochs", type=int, default=10)
    parser.add_argument("--flux-steps", type=int, default=500)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index/count")
    if args.model_type == "flux" and args.diffusers_root is None:
        parser.error("--diffusers-root is required for FLUX")

    subjects = list(load_subjects(args.classes_file).items())[args.shard_index :: args.num_shards]
    for subject, class_name in subjects:
        instance_dir = args.dataset_root / subject
        class_dir = args.class_root / slug(class_name)
        output_dir = args.output_root / subject
        if args.model_type == "lumina":
            final_checkpoint = output_dir / f"epoch{args.lumina_epochs - 1}" / "adapter_config.json"
            if final_checkpoint.exists():
                print(f"skip completed {subject}", flush=True)
                continue
            data_dir = args.data_root / subject
            if not (data_dir / "data.yaml").exists():
                run(
                    [
                        args.python,
                        "dreambooth/prepare_data.py",
                        "--instance-data-dir", str(instance_dir),
                        "--class-data-dir", str(class_dir),
                        "--instance-prompt", f"a sks {class_name}",
                        "--class-prompt", f"a {class_name}",
                        "--output-dir", str(data_dir),
                        "--model", str(args.base_model),
                        "--resolution", "512",
                        "--instance-repeats", str(args.instance_repeats),
                        "--prior-loss-weight", "1",
                    ]
                )
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON_BIN": args.python,
                    "MODEL": str(args.base_model),
                    "DATA_CONFIG": str(data_dir / "data.yaml"),
                    "OUTPUT_DIR": str(output_dir),
                    "EPOCHS": str(args.lumina_epochs),
                    "NPROC": "1",
                    "BATCH_SIZE": "1",
                    "ACCUM_ITER": "1",
                    "LORA_RANK": "16",
                    "MAX_SEQ_LEN": "1280",
                }
            )
            run(["bash", "dreambooth/train_lumina_dreambooth_lora.sh"], env)
        else:
            final_checkpoint = output_dir / "pytorch_lora_weights.safetensors"
            if final_checkpoint.exists():
                print(f"skip completed {subject}", flush=True)
                continue
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON_BIN": args.python,
                    "DIFFUSERS_ROOT": str(args.diffusers_root),
                    "MODEL": str(args.base_model),
                    "INSTANCE_DIR": str(instance_dir),
                    "CLASS_DIR": str(class_dir),
                    "INSTANCE_PROMPT": f"a sks {class_name}",
                    "CLASS_PROMPT": f"a {class_name}",
                    "OUTPUT_DIR": str(output_dir),
                    "MAX_TRAIN_STEPS": str(args.flux_steps),
                    "NUM_CLASS_IMAGES": "20",
                    "LORA_RANK": "16",
                    "ACCUM_ITER": "1",
                }
            )
            run(["bash", "dreambooth/train_flux_dreambooth.sh"], env)


if __name__ == "__main__":
    main()
