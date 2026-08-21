#!/usr/bin/env python3
"""Compute DreamBench CLIP-T, CLIP-I and DINO-I with fixed backbones.

Definitions follow Adobe Custom Diffusion's public evaluator: cosine similarity
for every generated/reference pair and paired image/prompt cosine.  The raw
image/text cosine is reported as ``clip_t`` for DreamBench comparability, while
the separately named ``clip_score`` is 2.5 * max(cosine, 0).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from generate_dreambench import output_path, read_manifest

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DINO_REPOSITORY = "facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a"


def rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


CLIP_TRANSFORM = Compose([
    Resize(224, interpolation=Image.BICUBIC), CenterCrop(224), rgb, ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
DINO_TRANSFORM = Compose([
    Resize(256, interpolation=Image.BICUBIC), CenterCrop(224), rgb, ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def image_features(paths, model, transform, device, batch_size, is_clip=False):
    outputs = []
    with torch.inference_mode():
        for batch in batches(paths, batch_size):
            pixels = torch.stack([transform(Image.open(path)) for path in batch]).to(device)
            clip_pixels = pixels.half() if device.type == "cuda" else pixels
            features = model.encode_image(clip_pixels) if is_clip else model(pixels)
            outputs.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
    return torch.cat(outputs)


def text_features(prompts, model, device, batch_size):
    outputs = []
    with torch.inference_mode():
        for batch in batches(prompts, batch_size):
            tokens = clip.tokenize(batch, truncate=True).to(device)
            features = model.encode_text(tokens)
            outputs.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/dreambench"))
    args = parser.parse_args()

    items = read_manifest(args.manifest)
    generated = [output_path(args.generated_root, item) for item in items]
    missing = [str(path) for path in generated if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} generated images; first: {missing[0]}")

    device = torch.device(args.device)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    clip_cache = args.cache_dir / "clip"
    torch_cache = args.cache_dir / "torch"
    clip_cache.mkdir(exist_ok=True)
    torch_cache.mkdir(exist_ok=True)
    torch.hub.set_dir(str(torch_cache))
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, download_root=str(clip_cache))
    clip_model.eval()
    dino_model = torch.hub.load(DINO_REPOSITORY, "dino_vits16").to(device).eval()
    gen_clip = image_features(generated, clip_model, CLIP_TRANSFORM, device, args.batch_size, True)
    gen_dino = image_features(generated, dino_model, DINO_TRANSFORM, device, args.batch_size)
    texts = text_features([item["prompt"] for item in items], clip_model, device, args.batch_size)
    clip_t = (gen_clip * texts).sum(dim=1)
    clip_score = 2.5 * clip_t.clamp_min(0)

    indices = defaultdict(list)
    for index, item in enumerate(items):
        indices[item["subject"]].append(index)
    per_subject = {}
    for subject, subject_indices in indices.items():
        references = sorted(
            path for path in (args.reference_root / subject).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        ref_clip = image_features(references, clip_model, CLIP_TRANSFORM, device, args.batch_size, True)
        ref_dino = image_features(references, dino_model, DINO_TRANSFORM, device, args.batch_size)
        selected_clip = gen_clip[subject_indices]
        selected_dino = gen_dino[subject_indices]
        per_subject[subject] = {
            "n_generated": len(subject_indices), "n_references": len(references),
            "clip_t": float(clip_t[subject_indices].mean()),
            "clip_score": float(clip_score[subject_indices].mean()),
            "clip_i": float((selected_clip @ ref_clip.T).mean()),
            "dino_i": float((selected_dino @ ref_dino.T).mean()),
        }
    overall = {
        key: float(np.mean([scores[key] for scores in per_subject.values()]))
        for key in ("clip_t", "clip_score", "clip_i", "dino_i")
    }
    result = {
        "protocol": {
            "clip_backbone": "openai/ViT-B/32",
            "dino_backbone": f"{DINO_REPOSITORY} dino_vits16",
            "identity_reduction": "mean all generated-reference cosines, then macro subject mean",
            "clip_t_definition": "cosine(image,text), then macro subject mean",
            "clip_score_definition": "2.5 * max(cosine(image,text),0), then macro subject mean",
        },
        "overall": overall, "per_subject": per_subject,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
