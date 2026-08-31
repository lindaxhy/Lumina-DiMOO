"""Shared protocol helpers for zero-shot and newer diffusion baselines."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PIL_RESAMPLING = getattr(Image, "Resampling", Image)


def reference_images(reference_root: Path, subject: str) -> list[Path]:
    """Return the canonical, deterministically ordered references for a subject."""
    subject_root = reference_root / subject
    if not subject_root.is_dir():
        raise FileNotFoundError(f"missing reference directory: {subject_root}")
    images = sorted(
        path for path in subject_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no reference images found in {subject_root}")
    return images


def select_reference(reference_root: Path, subject: str, index: int = 0) -> Path:
    images = reference_images(reference_root, subject)
    if not 0 <= index < len(images):
        raise IndexError(f"reference index {index} out of range for {subject} ({len(images)} images)")
    return images[index]


def identity_references(
    reference_root: Path, subject: str, excluded_index: int | None = None
) -> list[Path]:
    """Return all identity references, optionally holding out one conditioning image."""
    images = reference_images(reference_root, subject)
    if excluded_index is None:
        return images
    if not 0 <= excluded_index < len(images):
        raise IndexError(
            f"reference index {excluded_index} out of range for {subject} ({len(images)} images)"
        )
    held_out = [path for index, path in enumerate(images) if index != excluded_index]
    if not held_out:
        raise ValueError(f"cannot compute held-out identity for {subject} with no references left")
    return held_out


def prompt_without_identifier(prompt: str) -> str:
    """Remove only DreamBooth's synthetic identifier, preserving benchmark semantics."""
    return " ".join(word for word in prompt.split() if word != "sks")


def zero_shot_conditioning_prompt(item: dict) -> str:
    """Turn a DreamBench prompt into a reference-conditioned editing instruction."""
    class_name = item["class_name"]
    target = prompt_without_identifier(item["prompt"])
    return (
        f"Preserve the exact identity, shape, colors, texture, and distinctive details of the {class_name} "
        f"shown in the reference image. Generate a new image matching this description: {target}. "
        "Change the pose or scene only as required by the description; do not replace the subject."
    )


def ensure_square(image: Image.Image, resolution: int) -> Image.Image:
    """Center-crop to square and resize, avoiding aspect-ratio distortion."""
    image = image.convert("RGB")
    width, height = image.size
    edge = min(width, height)
    left = (width - edge) // 2
    top = (height - edge) // 2
    image = image.crop((left, top, left + edge, top + edge))
    return image.resize((resolution, resolution), PIL_RESAMPLING.LANCZOS)


class MetadataWriter:
    """Append replay metadata once for each newly rendered output."""

    def __init__(self, output_root: Path):
        self.path = output_root / "generation_metadata.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.output_paths = set()
        if self.path.exists():
            with self.path.open() as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self.output_paths.add(record.get("output_path"))

    def write(self, record: dict) -> None:
        if record.get("output_path") in self.output_paths:
            return
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.output_paths.add(record.get("output_path"))
