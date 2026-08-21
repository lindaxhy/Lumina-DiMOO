#!/usr/bin/env python3
"""Build one reusable prior-preservation image manifest per DreamBench class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreambench import load_subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=100)
    args = parser.parse_args()
    if args.num_images < 1:
        parser.error("--num-images must be positive")
    classes = list(dict.fromkeys(load_subjects(args.classes_file).values()))
    rows = []
    for class_name in classes:
        subject = class_name.replace(" ", "_")
        for index in range(args.num_images):
            rows.append(
                {
                    "subject": subject,
                    "class": class_name,
                    "prompt_id": index,
                    "prompt": f"a {class_name}",
                    "seed": index,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} samples for {len(classes)} classes to {args.output}")


if __name__ == "__main__":
    main()
