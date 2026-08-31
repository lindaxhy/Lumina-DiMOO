# Extended DreamBench: Qwen, BAGEL, SDXL, and Z-Image

This extension preserves the existing 30-subject DreamBench manifest and output
layout while adding four conditions:

| Condition | Checkpoint | Adaptation |
| --- | --- | --- |
| Qwen zero-shot | `Qwen/Qwen-Image-Edit-2511` | one fixed reference image |
| BAGEL zero-shot | `ByteDance-Seed/BAGEL-7B-MoT` | one fixed reference image |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | DreamBooth-LoRA |
| Z-Image | `Tongyi-MAI/Z-Image` | DreamBooth-LoRA |

The synthetic `sks` identifier is removed only for zero-shot inference. The
conditioning instruction retains the original benchmark scene text and asks the
model to preserve the subject shown in the reference. Reference index 0 means
the first image in lexicographically sorted filename order for every subject.
Every zero-shot output records its actual conditioning prompt and reference in
`generation_metadata.jsonl`.

## Revisions and environments

The tested source checkouts are:

- Google DreamBooth dataset: `4f887af7970a06fc0cd3adaa1d0b368547d6a1d0`
- Diffusers: `2f7e0154a9db246e95c9ede43edba7db5b130805`
- BAGEL: record `git -C ../external/BAGEL rev-parse HEAD` with every run

Use separate Python environments for Diffusers and BAGEL because BAGEL pins
older low-level dependencies. The BAGEL repository is code only; download
`ByteDance-Seed/BAGEL-7B-MoT` into a local checkpoint directory before running.

## 1. Build the 40-image-per-condition pilot

Run from the Lumina-DiMOO repository root:

```bash
python dreambooth/build_dreambench_manifest.py \
  --dataset-root ../external/google-dreambooth/dataset \
  --subjects dog,cat2,backpack,colorful_sneaker \
  --prompt-ids 0,5,10,15,20 --seeds 0,1 \
  --output runs/dreambench_extended/pilot.jsonl
```

## 2. Zero-shot generation

Qwen-Image-Edit uses the pinned Diffusers checkout:

```bash
PYTHONPATH=../external/diffusers/src \
python dreambooth/generate_dreambench.py \
  --model-type qwen_zero_shot \
  --base-model ../models/Qwen-Image-Edit-2511 \
  --manifest runs/dreambench_extended/pilot.jsonl \
  --reference-root ../external/google-dreambooth/dataset \
  --reference-index 0 --resolution 512 \
  --output-dir runs/dreambench_extended/qwen_pilot
```

BAGEL uses its official repository and a local checkpoint:

```bash
python dreambooth/generate_dreambench.py \
  --model-type bagel_zero_shot \
  --manifest runs/dreambench_extended/pilot.jsonl \
  --reference-root ../external/google-dreambooth/dataset \
  --reference-index 0 --resolution 512 \
  --bagel-repo ../external/BAGEL \
  --base-model ../models/BAGEL-7B-MoT \
  --output-dir runs/dreambench_extended/bagel_pilot
```

## 3. DreamBooth-LoRA training

Both runners use prior preservation, 20 model-specific class images, rank 16,
alpha 16, learning rate `1e-4`, 500 updates, and 512-pixel crops. Class images
are isolated by model family so they cannot be accidentally reused across SDXL
and Z-Image.

```bash
PYTHONPATH=../external/diffusers/src \
python dreambooth/run_extended_subject_shard.py \
  --model-type sdxl \
  --classes-file ../external/google-dreambooth/dataset/prompts_and_classes.txt \
  --dataset-root ../external/google-dreambooth/dataset \
  --class-root runs/dreambench_extended/class_images \
  --output-root runs/dreambench_extended/sdxl_adapters \
  --base-model ../models/SDXL-1.0 \
  --diffusers-root ../external/diffusers \
  --num-shards 8 --shard-index 0

PYTHONPATH=../external/diffusers/src \
python dreambooth/run_extended_subject_shard.py \
  --model-type z_image \
  --classes-file ../external/google-dreambooth/dataset/prompts_and_classes.txt \
  --dataset-root ../external/google-dreambooth/dataset \
  --class-root runs/dreambench_extended/class_images \
  --output-root runs/dreambench_extended/z_image_adapters \
  --base-model ../models/Z-Image \
  --diffusers-root ../external/diffusers \
  --num-shards 8 --shard-index 0
```

Launch shard indices 0 through 7 on distinct GPUs. Each completed adapter has a
`dreambench_training_config.json`; a mismatched existing adapter is rejected
instead of silently reused.

Generate SDXL or Z-Image outputs by passing `--model-type sdxl` or
`--model-type z_image` and an adapter template such as
`runs/dreambench_extended/sdxl_adapters/{subject}`.

## 4. Validate and evaluate

Validate coverage, dimensions, RGB mode, and decodability before feature
extraction:

```bash
python dreambooth/validate_dreambench_outputs.py \
  --manifest runs/dreambench_extended/pilot.jsonl \
  --generated-root runs/dreambench_extended/qwen_pilot \
  --resolution 512
```

For zero-shot conditions, report both standard all-reference identity and a
held-out identity score that excludes the conditioning image:

```bash
python dreambooth/evaluate_dreambench.py \
  --manifest runs/dreambench_extended/pilot.jsonl \
  --generated-root runs/dreambench_extended/qwen_pilot \
  --reference-root ../external/google-dreambooth/dataset \
  --output runs/dreambench_extended/qwen_metrics_all.json

python dreambooth/evaluate_dreambench.py \
  --manifest runs/dreambench_extended/pilot.jsonl \
  --generated-root runs/dreambench_extended/qwen_pilot \
  --reference-root ../external/google-dreambooth/dataset \
  --exclude-reference-index 0 \
  --output runs/dreambench_extended/qwen_metrics_heldout.json
```

Do not compare the zero-shot held-out identity column to an all-reference
fine-tuned column without labeling the different conditioning sets. Keep the
existing all-reference metric as the primary continuity result and use held-out
identity as the anti-copy diagnostic.
