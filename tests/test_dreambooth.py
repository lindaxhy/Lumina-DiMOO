from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dreambooth"))

from dreambench import OBJECT_PROMPTS, LIVE_PROMPTS, load_subjects, prompts_for
from compare_dreambench_metrics import compare
from prepare_data import balanced_records
from run_subject_shard import flux_training_config
from train_dreambooth import weighted_batch_mean

import torch
import pytest


def test_official_prompt_counts_and_subject_parser(tmp_path):
    classes_file = tmp_path / "prompts_and_classes.txt"
    classes_file.write_text(
        "DreamBench metadata\n"
        "subject_name,class\n"
        "backpack,backpack\n"
        "dog,dog\n"
        "\n"
        "ignored footer\n"
    )
    assert load_subjects(classes_file) == {"backpack": "backpack", "dog": "dog"}
    assert len(OBJECT_PROMPTS) == len(LIVE_PROMPTS) == 25
    assert "wearing a red hat" in prompts_for("dog")[10]
    assert "wheat field" in prompts_for("backpack")[10]


def test_balanced_prior_records():
    instance = [{"path": f"i{x}.pkl", "len": 1024} for x in range(5)]
    classes = [{"path": f"c{x}.pkl", "len": 1024} for x in range(3)]
    records = balanced_records(instance, classes, "a sks dog", "a dog", 2, 0.7, 0)
    instance_records = [x for x in records if x["sample_kind"] == "instance"]
    class_records = [x for x in records if x["sample_kind"] == "class"]
    assert len(instance_records) == len(class_records) == 10
    assert {x["loss_weight"] for x in instance_records} == {1.0}
    assert {x["loss_weight"] for x in class_records} == {0.7}
    assert {x["user_prompt"] for x in instance_records} == {"a sks dog"}
    assert {x["user_prompt"] for x in class_records} == {"a dog"}


def test_prior_weight_is_not_cancelled_for_batch_size_one():
    loss = torch.tensor([2.0])
    assert weighted_batch_mean(loss, torch.tensor([0.25])).item() == 0.5


def test_flux_alpha_defaults_to_rank():
    class Args:
        lora_rank = 16
        lora_alpha = None
        flux_steps = 500
        num_class_images = 20

    config = flux_training_config(Args(), "dog", "dog")
    assert config["lora_rank"] == config["lora_alpha"] == 16
    assert config["lora_scale"] == 1.0
    assert config["instance_prompt"] == "a sks dog"
    assert config["post_validation_prompt"] == "a sks dog"


def test_base_only_delta_report():
    protocol = {
        "clip_backbone": "clip",
        "dino_backbone": "dino",
        "identity_reduction": "macro",
    }
    base = {
        "protocol": protocol,
        "overall": {key: 0.2 for key in ("clip_t", "clip_score", "clip_i", "dino_i")},
        "per_subject": {
            "dog": {
                **{key: 0.2 for key in ("clip_t", "clip_score", "clip_i", "dino_i")},
                "n_generated": 4,
            }
        },
    }
    lora = {
        "protocol": protocol,
        "overall": {key: 0.3 for key in ("clip_t", "clip_score", "clip_i", "dino_i")},
        "per_subject": {
            "dog": {
                **{key: 0.3 for key in ("clip_t", "clip_score", "clip_i", "dino_i")},
                "n_generated": 4,
            }
        },
    }
    result = compare(base, lora)
    assert result["overall_delta"]["dino_i"] == pytest.approx(0.1)
    assert result["identity_subject_wins"] == {"clip_i": 1, "dino_i": 1}
