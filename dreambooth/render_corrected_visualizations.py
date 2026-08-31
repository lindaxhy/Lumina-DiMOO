#!/usr/bin/env python3
"""Render the corrected DreamBench metric and qualitative comparison figures."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


METRICS = (
    ("clip_t", "CLIP-T"),
    ("clip_score", "CLIPScore"),
    ("clip_i", "CLIP-I"),
    ("dino_i", "DINO-I"),
)
MODELS = (
    ("Lumina-DiMOO LoRA", "lumina", "#7E57C2"),
    ("FLUX LoRA (r16, α16)", "flux", "#1E88E5"),
    ("FLUX base-only", "base", "#90A4AE"),
)
SUBJECTS = ("dog", "cat2", "backpack", "colorful_sneaker")


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (512, 512))


def render_metrics(metrics: dict[str, dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, METRICS, strict=True):
        values = [metrics[source]["overall"][key] for _, source, _ in MODELS]
        bars = axis.bar(
            range(len(MODELS)),
            values,
            color=[color for _, _, color in MODELS],
            width=0.68,
        )
        axis.set_title(title, fontsize=16, fontweight="bold")
        axis.set_ylim(0, max(values) * 1.22)
        axis.set_xticks(range(len(MODELS)), [label for label, _, _ in MODELS], fontsize=10)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4, fontsize=11)

    fig.suptitle(
        "Corrected DreamBench comparison — 30 subjects, 3000 images per model",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "FLUX LoRA − base-only: CLIP-T −0.0031  |  CLIP-I +0.0253  |  DINO-I +0.0384",
        ha="center",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_qualitative(
    manifest_path: Path,
    reference_root: Path,
    generated_roots: dict[str, Path],
    output: Path,
) -> None:
    entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    prompts = {
        entry["subject"]: entry["prompt"]
        for entry in entries
        if entry["subject"] in SUBJECTS and entry["prompt_id"] == 10 and entry["seed"] == 0
    }
    columns = (
        ("Reference", None),
        ("Lumina-DiMOO LoRA", "lumina"),
        ("FLUX LoRA (r16, α16)", "flux"),
        ("FLUX base-only", "base"),
    )
    fig, axes = plt.subplots(len(SUBJECTS), len(columns), figsize=(16, 16), constrained_layout=True)
    for row, subject in enumerate(SUBJECTS):
        reference = sorted((reference_root / subject).glob("*"))[0]
        paths = [reference] + [generated_roots[key] / subject / "10_0.png" for _, key in columns[1:]]
        for col, ((label, _), path) in enumerate(zip(columns, paths, strict=True)):
            axis = axes[row, col]
            axis.imshow(load_rgb(path))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(2 if col == 2 else 0.8)
                spine.set_edgecolor("#1E88E5" if col == 2 else "#CFD8DC")
            if row == 0:
                axis.set_title(label, fontsize=15, fontweight="bold", pad=10)
            if col == 0:
                prompt = textwrap.fill(prompts[subject], width=34)
                axis.set_ylabel(
                    f"{subject}\n\n{prompt}",
                    fontsize=12,
                    fontweight="bold",
                    rotation=0,
                    labelpad=105,
                    va="center",
                )

    fig.suptitle(
        "Same official prompt (#10) and seed (0)",
        fontsize=20,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("runs/dreambench/manifest.jsonl"))
    parser.add_argument("--reference-root", type=Path, default=Path("../external/google-dreambooth/dataset"))
    parser.add_argument("--lumina-root", type=Path, default=Path("runs/dreambench/lumina"))
    parser.add_argument("--flux-root", type=Path, default=Path("runs/dreambench/flux_alpha16"))
    parser.add_argument("--base-root", type=Path, default=Path("runs/dreambench/flux_base"))
    parser.add_argument("--lumina-metrics", type=Path, default=Path("runs/dreambench/lumina_metrics.json"))
    parser.add_argument("--flux-metrics", type=Path, default=Path("runs/dreambench/flux_alpha16_metrics.json"))
    parser.add_argument("--base-metrics", type=Path, default=Path("runs/dreambench/flux_base_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("dreambooth/results"))
    args = parser.parse_args()

    metrics = {
        "lumina": json.loads(args.lumina_metrics.read_text()),
        "flux": json.loads(args.flux_metrics.read_text()),
        "base": json.loads(args.base_metrics.read_text()),
    }
    render_metrics(metrics, args.output_dir / "metric_comparison_alpha16.png")
    render_qualitative(
        args.manifest,
        args.reference_root,
        {"lumina": args.lumina_root, "flux": args.flux_root, "base": args.base_root},
        args.output_dir / "qualitative_comparison_alpha16.png",
    )


if __name__ == "__main__":
    main()
