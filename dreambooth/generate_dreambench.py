#!/usr/bin/env python3
"""Generate the same DreamBench manifest with Lumina-DiMOO or FLUX.1-dev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_manifest(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def output_path(root: Path, item: dict) -> Path:
    return root / item["subject"] / f'{item["prompt_id"]:02d}_{item["seed"]}.png'


def grouped(items: list[dict]):
    subjects = list(dict.fromkeys(item["subject"] for item in items))
    for subject in subjects:
        yield subject, [item for item in items if item["subject"] == subject]


def checkpoint_for(template: str, subject: str) -> str:
    return template.format(subject=subject) if "{subject}" in template else template


def generate_lumina(args: argparse.Namespace, items: list[dict]) -> None:
    from diffusers import VQModel
    from transformers import AutoTokenizer
    from peft import PeftModel
    from config import SPECIAL_TOKENS
    from generators.image_generation_generator import generate_image
    from model import LLaDAForMultiModalGeneration
    from utils.image_utils import add_break_line, calculate_vq_params, decode_vq_to_image
    from utils.prompt_utils import create_prompt_templates, generate_text_to_image_prompt

    device = torch.device(args.device)
    vqvae = VQModel.from_pretrained(args.base_model, subfolder="vqvae").to(device)
    templates = create_prompt_templates()
    seq_len, newline_every, grid_h, grid_w = calculate_vq_params(args.resolution, args.resolution)
    special = SPECIAL_TOKENS
    masked_image = add_break_line(
        [special["mask_token"]] * seq_len, grid_h, grid_w, new_number=special["newline_token"]
    )
    for subject, subject_items in grouped(items):
        checkpoint = Path(args.base_model) if args.base_only else Path(checkpoint_for(args.checkpoint_template, subject))
        tokenizer_source = args.base_model if (checkpoint / "adapter_config.json").exists() else checkpoint
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        model = LLaDAForMultiModalGeneration.from_pretrained(
            args.base_model if (checkpoint / "adapter_config.json").exists() else checkpoint,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        ).eval()
        if (checkpoint / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, checkpoint).eval()
        for item in subject_items:
            destination = output_path(args.output_dir, item)
            if destination.exists() and not args.overwrite:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(item["seed"])
            prompt, unconditioned = generate_text_to_image_prompt(item["prompt"], templates)
            conditional_ids = tokenizer(prompt)["input_ids"]
            unconditional_ids = tokenizer(unconditioned)["input_ids"]
            prediction = [
                special["answer_start"], special["boi"], *masked_image,
                special["eoi"], special["answer_end"],
            ]
            prompt_ids = torch.tensor(conditional_ids + prediction, device=device).unsqueeze(0)
            uncon_ids = torch.tensor(unconditional_ids, device=device).unsqueeze(0)
            tokens = generate_image(
                model, prompt_ids, seq_len=seq_len, newline_every=newline_every,
                timesteps=args.steps, temperature=1.0, cfg_scale=args.guidance_scale,
                uncon_ids=uncon_ids, code_start=len(conditional_ids) + 2,
                use_cache=args.use_cache,
            )
            image = decode_vq_to_image(
                tokens, str(destination), vae_ckpt=args.base_model,
                image_height=args.resolution, image_width=args.resolution, vqvae=vqvae,
            )
            image.save(destination)
        del model, tokenizer
        torch.cuda.empty_cache()


def generate_flux(args: argparse.Namespace, items: list[dict]) -> None:
    from diffusers import FluxPipeline

    device = torch.device(args.device)
    pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device)
    for subject, subject_items in grouped(items):
        if not args.base_only:
            pipe.load_lora_weights(checkpoint_for(args.checkpoint_template, subject))
        for item in subject_items:
            destination = output_path(args.output_dir, item)
            if destination.exists() and not args.overwrite:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            generator = torch.Generator(device=device).manual_seed(item["seed"])
            image = pipe(
                item["prompt"], height=args.resolution, width=args.resolution,
                num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            image.save(destination)
        if not args.base_only:
            pipe.unload_lora_weights()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("lumina", "flux"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-template", help="path containing a {subject} placeholder")
    parser.add_argument("--base-only", action="store_true", help="generate without loading subject adapters")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.base_only and not args.checkpoint_template:
        parser.error("--checkpoint-template is required unless --base-only is set")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.model_type == "lumina":
        args.base_model = args.base_model or "Alpha-VLLM/Lumina-DiMOO"
        args.steps = args.steps or 64
        args.guidance_scale = args.guidance_scale or 4.0
    else:
        args.base_model = args.base_model or "black-forest-labs/FLUX.1-dev"
        args.steps = args.steps or 28
        args.guidance_scale = args.guidance_scale or 3.5
    items = read_manifest(args.manifest)
    subjects = list(dict.fromkeys(item["subject"] for item in items))
    selected = set(subjects[args.shard_index :: args.num_shards])
    items = [item for item in items if item["subject"] in selected]
    (generate_lumina if args.model_type == "lumina" else generate_flux)(args, items)


if __name__ == "__main__":
    main()
