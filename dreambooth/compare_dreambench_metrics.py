#!/usr/bin/env python3
"""Report the causal metric change from a base-only run to a LoRA run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("clip_t", "clip_score", "clip_i", "dino_i")


def compare(base: dict, lora: dict) -> dict:
    for field in ("clip_backbone", "dino_backbone", "identity_reduction"):
        if base.get("protocol", {}).get(field) != lora.get("protocol", {}).get(field):
            raise ValueError(f"base-only and LoRA results use different {field}")
    base_subjects = base["per_subject"]
    lora_subjects = lora["per_subject"]
    if set(base_subjects) != set(lora_subjects):
        raise ValueError("base-only and LoRA results cover different subjects")
    for subject in base_subjects:
        if base_subjects[subject]["n_generated"] != lora_subjects[subject]["n_generated"]:
            raise ValueError(f"base-only and LoRA results have different sample counts for {subject}")
    per_subject = {
        subject: {
            metric: lora_subjects[subject][metric] - base_subjects[subject][metric]
            for metric in METRICS
        }
        for subject in base_subjects
    }
    overall = {
        metric: lora["overall"][metric] - base["overall"][metric]
        for metric in METRICS
    }
    identity_wins = {
        metric: sum(delta[metric] > 0 for delta in per_subject.values())
        for metric in ("clip_i", "dino_i")
    }
    return {
        "definition": "LoRA metric minus base-only metric with the same manifest and seeds",
        "overall_delta": overall,
        "identity_subject_wins": identity_wins,
        "num_subjects": len(per_subject),
        "per_subject_delta": per_subject,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.base.read_text()), json.loads(args.lora.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall_delta"], indent=2))


if __name__ == "__main__":
    main()
