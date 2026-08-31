#!/usr/bin/env python3
"""Validate DreamBench image coverage and image-file contracts before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_dreambench import output_path, read_manifest
from PIL import Image


def validate(manifest: Path, generated_root: Path, resolution: int) -> dict:
    items = read_manifest(manifest)
    keys = [(item["subject"], item["prompt_id"], item["seed"]) for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate subject/prompt_id/seed keys")

    expected = {output_path(generated_root, item) for item in items}
    actual = {path for path in generated_root.glob("*/*.png") if path.is_file()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    invalid = []
    for path in sorted(expected & actual):
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (resolution, resolution) or image.mode != "RGB":
                    invalid.append(
                        {"path": str(path), "size": list(image.size), "mode": image.mode}
                    )
        except (OSError, ValueError) as error:
            invalid.append({"path": str(path), "error": repr(error)})
    report = {
        "expected": len(expected),
        "found": len(actual),
        "valid": len(expected & actual) - len(invalid),
        "missing": [str(path) for path in missing],
        "extra": [str(path) for path in extra],
        "invalid": invalid,
        "resolution": resolution,
    }
    if missing or extra or invalid:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.manifest, args.generated_root, args.resolution)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
