"""Load local models and LoRA adapters, and build model chat messages."""

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(base_model: str, *, gradient_checkpointing: bool):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.eos_token is None:
        raise ValueError("The base tokenizer must define an EOS token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": 0},
    }

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.enable_input_require_grads()
    return tokenizer, model


def load_adapter_for_inference(adapter_path):
    adapter_cfg = PeftConfig.from_pretrained(str(adapter_path))
    base_model = adapter_cfg.base_model_name_or_path
    tokenizer, model = load_model_and_tokenizer(base_model, gradient_checkpointing=False)
    model = PeftModel.from_pretrained(
        model, str(adapter_path), config=adapter_cfg, is_trainable=False,
    )
    model.config.use_cache = True
    model.eval()
    tokenizer.padding_side = "left"
    return tokenizer, model, base_model


def build_chat_messages(system: str, instruction: str, chat_cfg: dict) -> list[dict]:
    if chat_cfg["support_system_role"]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
    content = (
        chat_cfg["system_prefix"] + system
        + chat_cfg["system_suffix"] + instruction
    )
    return [{"role": "user", "content": content}]
