import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dreambooth"))

from extended_protocol import (
    ensure_square,
    identity_references,
    prompt_without_identifier,
    select_reference,
    zero_shot_conditioning_prompt,
)
from run_extended_subject_shard import training_config
from validate_dreambench_outputs import validate


def test_zero_shot_prompt_preserves_benchmark_text_and_removes_identifier():
    item = {
        "class_name": "dog",
        "prompt": "a sks dog wearing a red hat",
    }
    assert prompt_without_identifier(item["prompt"]) == "a dog wearing a red hat"
    prompt = zero_shot_conditioning_prompt(item)
    assert "reference image" in prompt
    assert "a dog wearing a red hat" in prompt
    assert "sks" not in prompt


def test_reference_selection_is_sorted_and_explicit(tmp_path):
    subject = tmp_path / "dog"
    subject.mkdir()
    Image.new("RGB", (8, 8)).save(subject / "b.png")
    Image.new("RGB", (8, 8)).save(subject / "a.jpg")
    assert select_reference(tmp_path, "dog", 0).name == "a.jpg"
    assert select_reference(tmp_path, "dog", 1).name == "b.png"
    assert [path.name for path in identity_references(tmp_path, "dog", 0)] == ["b.png"]


def test_square_output_contract():
    image = Image.new("RGB", (20, 10), "red")
    assert ensure_square(image, 16).size == (16, 16)


def test_extended_training_metadata_uses_unit_lora_scale():
    args = argparse.Namespace(
        model_type="z_image",
        base_model="Tongyi-MAI/Z-Image",
        resolution=512,
        steps=500,
        learning_rate=1e-4,
        lora_rank=16,
        lora_alpha=None,
        num_class_images=20,
        prior_loss_weight=1.0,
    )
    config = training_config(args, "dog", "dog")
    assert config["lora_alpha"] == config["lora_rank"] == 16
    assert config["lora_scale"] == 1.0
    assert config["instance_prompt"] == "a sks dog"


def test_output_validator(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    item = {"subject": "dog", "class_name": "dog", "prompt_id": 0, "prompt": "a sks dog", "seed": 0}
    manifest.write_text(json.dumps(item) + "\n")
    output = tmp_path / "generated" / "dog" / "00_0.png"
    output.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(output)
    report = validate(manifest, tmp_path / "generated", 32)
    assert report["valid"] == 1
