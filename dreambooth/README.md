# DreamBooth on Lumina-DiMOO

This directory adapts DreamBooth to Lumina-DiMOO's discrete masked-token
diffusion loss and provides a fixed DreamBench protocol against FLUX.1-dev.

The implementation uses the two defining DreamBooth terms:

1. instance binding: target images are conditioned on `a sks <class>`;
2. prior preservation: base-model class images are conditioned on `a <class>`
   and receive `prior_loss_weight` in the masked VQ-token cross entropy.

Unlike continuous diffusion DreamBooth, there is no epsilon/velocity regression.
The target is the original VQ code at randomly masked image-token positions.

Install the upstream project first, then install the additional benchmark and
LoRA dependencies:

```bash
pip install -r dreambooth/requirements.txt
```

## Data preparation

Generate class images for each class with the unmodified base model, then
pre-tokenize one subject. The formal comparison used 20 class images for each
of the 15 unique classes; `build_class_manifest.py` can create the reusable
generation manifest:

```bash
python dreambooth/build_class_manifest.py \
  --classes-file ../external/google-dreambooth/dataset/prompts_and_classes.txt \
  --num-images 20 --output runs/class_images/manifest.jsonl
```

Pre-tokenize a subject after rendering its class images:

```bash
python dreambooth/prepare_data.py \
  --instance-data-dir ../external/google-dreambooth/dataset/dog \
  --class-data-dir runs/class_images/dog \
  --instance-prompt 'a sks dog' \
  --class-prompt 'a dog' \
  --output-dir runs/lumina_dreambooth/dog \
  --resolution 512 --instance-repeats 20 --prior-loss-weight 1
```

`prepare_data.py` creates equal numbers of instance and class records. Class
images can be reused by all DreamBench subjects of the same class.

## Training

Run from the Lumina-DiMOO repository root:

```bash
DATA_CONFIG=runs/lumina_dreambooth/dog/data.yaml \
OUTPUT_DIR=runs/lumina_dreambooth/dog/checkpoints \
bash dreambooth/train_lumina_dreambooth.sh
```

The default is full-model FSDP training on eight GPUs. The 512-pixel image grid
uses about 1024 VQ tokens, so `MAX_SEQ_LEN=1280` leaves room for the prompt.

For the 30-subject benchmark, use the symmetric rank-16 LoRA setup (one H20 per
subject, matching the FLUX baseline):

```bash
MODEL=checkpoints/Alpha-VLLM-Lumina-DiMOO \
DATA_CONFIG=runs/lumina_dreambooth/dog/data.yaml \
OUTPUT_DIR=runs/lumina_lora/dog \
bash dreambooth/train_lumina_dreambooth_lora.sh
```

The FLUX baseline calls the pinned official Diffusers trainer:

```bash
DIFFUSERS_ROOT=../external/diffusers \
INSTANCE_DIR=../external/google-dreambooth/dataset/dog \
CLASS_DIR=runs/class_images/flux_dog \
INSTANCE_PROMPT='a sks dog' CLASS_PROMPT='a dog' \
OUTPUT_DIR=runs/flux_dreambooth/dog \
bash dreambooth/train_flux_dreambooth.sh
```

FLUX.1-dev is gated. The Hugging Face account used by the runner must accept
its license and provide a token before the baseline can run.

## Fixed DreamBench manifest

```bash
python dreambooth/build_dreambench_manifest.py \
  --dataset-root ../external/google-dreambooth/dataset \
  --output runs/dreambench/manifest.jsonl --seeds 0,1,2,3
```

This produces 30 subjects x 25 official prompts x 4 seeds = 3000 samples per
model. Both models must render exactly this manifest at 512x512. Report DINO-I
(generated image vs. all reference images) and CLIP-T (generated image vs.
prompt), with identical feature backbones and preprocessing for both models.

Generate images (the checkpoint template must contain the literal
`{subject}` placeholder):

```bash
python dreambooth/generate_dreambench.py --model-type lumina \
  --manifest runs/dreambench/manifest.jsonl \
  --checkpoint-template 'runs/lumina_dreambooth/{subject}/checkpoints/epoch0' \
  --output-dir runs/dreambench/lumina

python dreambooth/generate_dreambench.py --model-type flux \
  --manifest runs/dreambench/manifest.jsonl \
  --checkpoint-template 'runs/flux_dreambooth/{subject}' \
  --output-dir runs/dreambench/flux
```

Evaluate each output with the same process:

```bash
python dreambooth/evaluate_dreambench.py \
  --manifest runs/dreambench/manifest.jsonl \
  --generated-root runs/dreambench/lumina \
  --reference-root ../external/google-dreambooth/dataset \
  --output runs/dreambench/lumina_metrics.json
```

The evaluator reports raw CLIP-T cosine, the separately named scaled
CLIPScore (`2.5 * max(CLIP-T, 0)`), CLIP-I, and DINO-I. Its preprocessing and
all-pairs identity reduction match the public Adobe Custom Diffusion evaluator,
and the JSON records the exact metric definitions.

## Reproduced results

The formal run used 30 subjects, 25 official prompts, four seeds, 512×512
generation, rank-16 LoRA, learning rate `1e-4`, and prior loss weight `1.0`.
Lumina used 400–600 updates according to its 4–6 instance images; FLUX used 500
updates. FLUX's prior-preservation batch processes one instance and one class
image per update, so equal update counts do not imply equal compute.

| Model | CLIP-T ↑ | CLIPScore ↑ | CLIP-I ↑ | DINO-I ↑ |
| --- | ---: | ---: | ---: | ---: |
| Lumina-DiMOO DreamBooth-LoRA | 0.289681 | 0.724202 | **0.777162** | **0.626577** |
| FLUX.1-dev DreamBooth-LoRA | **0.300018** | **0.750045** | 0.683758 | 0.376527 |

Lumina wins 27/30 subjects on CLIP-I and 30/30 on DINO-I. FLUX wins 21/30
subjects on CLIP-T. The machine-readable aggregate, deltas, protocol, and win
counts are in [`results/dreambench_summary.json`](results/dreambench_summary.json).
