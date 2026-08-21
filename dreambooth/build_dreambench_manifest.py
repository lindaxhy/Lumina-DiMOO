#!/usr/bin/env python3
"""Build one shared DreamBench generation manifest for both compared models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreambench import load_subjects, prompts_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", default="sks")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--subjects", default="", help="comma-separated subset; default is all 30")
    args = parser.parse_args()
    mapping = load_subjects(args.dataset_root / "prompts_and_classes.txt")
    selected = [x for x in args.subjects.split(",") if x] or list(mapping)
    unknown = sorted(set(selected) - set(mapping))
    if unknown:
        raise ValueError(f"unknown subjects: {unknown}")
    seeds = [int(value) for value in args.seeds.split(",")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w") as handle:
        for subject in selected:
            class_name = mapping[subject]
            for prompt_id, prompt in enumerate(prompts_for(class_name, args.token)):
                for seed in seeds:
                    item = {
                        "subject": subject,
                        "class_name": class_name,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "seed": seed,
                    }
                    handle.write(json.dumps(item) + "\n")
                    count += 1
    print(f"wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
