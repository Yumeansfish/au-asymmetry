"""Generate batched responses with a local LoRA adapter."""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import set_seed

from ..utils.io import PROJECT_ROOT, load_queries, resolve_path, write_responses
from ..utils.prompts import load_system_prompt
from .model import build_chat_messages, load_adapter_for_inference


def generate(cfg, queries):
    if not torch.cuda.is_available():
        raise RuntimeError("Local inference requires a CUDA GPU.")
    set_seed(cfg["seed"])
    with (PROJECT_ROOT / "config" / "models" / f"{cfg['model']}.yaml").open(
        encoding="utf-8"
    ) as handle:
        model_cfg = yaml.safe_load(handle)
    tokenizer, model, base_model = load_adapter_for_inference(cfg["adapter_path"])
    batch_size = cfg["batch_size"]
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_chat_messages(
                    query.get("system", cfg["system_prompt"]),
                    query["instruction"], model_cfg["chat_template"],
                ),
                tokenize=False, add_generation_prompt=True,
            )
            for query in batch
        ]
        # The chat template already supplies the special tokens.
        inputs = tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, **cfg["generation"], pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for query, response in zip(batch, responses):
            yield {"query": query, "response": response, "model": base_model}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "inference.yaml",
    )
    parser.add_argument(
        "--prompt", choices=("plain", "utility", "de_alignment"),
        help="Select a system prompt from config/prompts.yaml",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["model"] not in ("llama", "gemma", "mistral", "qwen"):
        parser.error("Model must be llama, gemma, mistral, or qwen.")
    if not isinstance(cfg["adapter_path"], str) or not cfg["adapter_path"].strip():
        parser.error(f"Set adapter_path to a local checkpoint directory in {args.config}.")
    cfg["adapter_path"] = resolve_path(cfg["adapter_path"])
    if not (cfg["adapter_path"] / "adapter_config.json").is_file():
        parser.error(f"Missing adapter_config.json in {cfg['adapter_path']}.")
    if not isinstance(cfg["batch_size"], int) or cfg["batch_size"] <= 0:
        parser.error("batch_size must be a positive integer.")
    cfg["system_prompt"] = load_system_prompt(args.prompt or cfg["prompt"])
    queries = load_queries(resolve_path(cfg["queries_path"]))
    write_responses(
        generate(cfg, queries), resolve_path(cfg["output_dir"]), total=len(queries),
    )


if __name__ == "__main__":
    main()
