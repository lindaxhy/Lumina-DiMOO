#!/usr/bin/env bash
set -euo pipefail

: "${DIFFUSERS_ROOT:?set DIFFUSERS_ROOT to the official diffusers checkout}"
: "${INSTANCE_DIR:?set INSTANCE_DIR}"
: "${CLASS_DIR:?set CLASS_DIR}"
: "${INSTANCE_PROMPT:?set INSTANCE_PROMPT, e.g. 'a sks dog'}"
: "${CLASS_PROMPT:?set CLASS_PROMPT, e.g. 'a dog'}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${MODEL:=stabilityai/stable-diffusion-xl-base-1.0}"
: "${PYTHON_BIN:=python}"
: "${RESOLUTION:=512}"
: "${BATCH_SIZE:=1}"
: "${ACCUM_ITER:=4}"
: "${LR:=1e-4}"
: "${MAX_TRAIN_STEPS:=500}"
: "${NUM_CLASS_IMAGES:=20}"
: "${LORA_RANK:=16}"
: "${PRIOR_LOSS_WEIGHT:=1.0}"

mkdir -p "${CLASS_DIR}" "${OUTPUT_DIR}"

PYTHONPATH="${DIFFUSERS_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" -m accelerate.commands.launch \
  "${DIFFUSERS_ROOT}/examples/dreambooth/train_dreambooth_lora_sdxl.py" \
  --pretrained_model_name_or_path="${MODEL}" \
  --instance_data_dir="${INSTANCE_DIR}" \
  --class_data_dir="${CLASS_DIR}" \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --class_prompt="${CLASS_PROMPT}" \
  --output_dir="${OUTPUT_DIR}" \
  --with_prior_preservation \
  --prior_loss_weight="${PRIOR_LOSS_WEIGHT}" \
  --num_class_images="${NUM_CLASS_IMAGES}" \
  --prior_generation_precision=bf16 \
  --resolution="${RESOLUTION}" \
  --train_batch_size="${BATCH_SIZE}" \
  --gradient_accumulation_steps="${ACCUM_ITER}" \
  --gradient_checkpointing \
  --mixed_precision=bf16 \
  --learning_rate="${LR}" \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps="${MAX_TRAIN_STEPS}" \
  --rank="${LORA_RANK}" \
  --seed=0
