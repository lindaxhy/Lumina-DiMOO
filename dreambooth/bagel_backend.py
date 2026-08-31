"""Thin, non-Gradio BAGEL loader used by the DreamBench generator."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bagel(repo_root: Path, model_path: Path, device_memory: str = "90GiB"):
    """Load the official BAGEL checkpoint without importing its Gradio app."""
    repo_root = repo_root.resolve()
    model_path = model_path.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from accelerate import (
        infer_auto_device_map,
        init_empty_weights,
        load_checkpoint_and_dispatch,
    )
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    llm_config = Qwen2Config.from_json_file(os.fspath(model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(os.fspath(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(local_path=os.fspath(model_path / "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    device_map = infer_auto_device_map(
        model,
        max_memory={index: device_memory for index in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for name in same_device_modules:
            device_map[name] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        for name in same_device_modules:
            if name in device_map:
                device_map[name] = first_device
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.fspath(model_path / "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder=os.fspath(model_path / "offload"),
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()
    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids,
    )
