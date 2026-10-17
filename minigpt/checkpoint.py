"""Build models from a config dict, and save/load checkpoints."""

import torch

from minigpt.gpt import GPT, GPTConfig
from minigpt.model import BigramLanguageModel


def build_model(model_type, config):
    if model_type == "bigram":
        return BigramLanguageModel(config["vocab_size"])
    if model_type == "gpt":
        return GPT(GPTConfig(**config))
    raise ValueError(f"unknown model type: {model_type}")


def save_checkpoint(path, model_type, config, model, **extra):
    torch.save(
        {"model_type": model_type, "config": config,
         "state_dict": model.state_dict(), **extra},
        path,
    )


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_type"], ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt
