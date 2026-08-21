#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:=Alpha-VLLM/Lumina-DiMOO}"
: "${DATA_CONFIG:?set DATA_CONFIG to prepare_data.py output data.yaml}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${NPROC:=8}"
: "${EPOCHS:=10}"
: "${BATCH_SIZE:=1}"
: "${ACCUM_ITER:=2}"
: "${LR:=2e-6}"
: "${MAX_SEQ_LEN:=1280}"
: "${PYTHON_BIN:=python}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" dreambooth/train_dreambooth.py \
  --init_from "${MODEL}" \
  --data_config "${DATA_CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_parallel fsdp \
  --precision bf16 \
  --grad_precision fp32 \
  --checkpointing \
  --batch_size "${BATCH_SIZE}" \
  --accum_iter "${ACCUM_ITER}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --min_lr "${LR}" \
  --warmup_epochs 0.1 \
  --wd 0.01 \
  --clip_grad 1 \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --num_workers 4 \
  --cache_ann_on_disk \
  --save_iteration_interval 100
