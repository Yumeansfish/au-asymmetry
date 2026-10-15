"""Fine-tune a LoRA adapter from local system/instruction/output JSONL records."""

import argparse
import gc
import os
import random
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from ..utils.io import PROJECT_ROOT, load_records
from .model import build_chat_messages, load_model_and_tokenizer


CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(model_name: str) -> dict:
    with (CONFIG_DIR / "base.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with (CONFIG_DIR / "models" / f"{model_name}.yaml").open(encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)
    cfg["base_model"] = model_cfg["base_model"]
    cfg["chat_template"] = model_cfg["chat_template"]
    return cfg


def load_training_records(train_path: Path) -> list[dict]:
    records = load_records(train_path, ("instruction", "output"))
    for index, record in enumerate(records, 1):
        if not isinstance(record.get("system"), str):
            raise ValueError(f"{train_path}: record {index}: expected a system string")
    return [
        {field: record[field] for field in ("system", "instruction", "output")}
        for record in records
    ]


def tokenize_record(record, tokenizer, chat_cfg):
    messages = build_chat_messages(record["system"], record["instruction"], chat_cfg)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Preserve the special tokens supplied by the chat template.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(
        prompt + record["output"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    if not input_ids:
        raise ValueError("Tokenizer produced an empty training sequence.")
    boundary = 0
    for prompt_token, full_token in zip(prompt_ids, input_ids):
        if prompt_token != full_token:
            break
        boundary += 1
    completion_mask = [0] * boundary + [1] * (len(input_ids) - boundary)
    return input_ids, completion_mask


def prepare_training_dataset(
    records, tokenizer, chat_cfg, *, total_tokens, max_length, seed,
):
    if total_tokens <= 0 or max_length <= 0 or total_tokens % max_length:
        raise ValueError("The token budget must contain whole fixed-length sequences.")
    rng = random.Random(seed)
    order = list(range(len(records)))
    packed = {"input_ids": [], "completion_mask": []}
    token_buffer, mask_buffer = [], []
    remaining = total_tokens
    # Shuffle each pass and repeat records until the processed-token budget is filled.
    while remaining:
        rng.shuffle(order)
        for index in order:
            input_ids, completion_mask = tokenize_record(records[index], tokenizer, chat_cfg)
            limit = min(remaining, len(input_ids))
            position = 0
            while position < limit:
                end = min(position + max_length - len(token_buffer), limit)
                token_buffer.extend(input_ids[position:end])
                mask_buffer.extend(completion_mask[position:end])
                position = end
                if len(token_buffer) == max_length:
                    packed["input_ids"].append(token_buffer)
                    packed["completion_mask"].append(mask_buffer)
                    token_buffer, mask_buffer = [], []
            remaining -= limit
            if not remaining:
                break
    return Dataset.from_dict(packed)


def run_training(cfg, args):
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    for key in ("max_steps", "save_steps", "save_total_limit", "seed"):
        value = getattr(args, key)
        if value is not None:
            train_cfg[key] = value
    for key in ("max_steps", "batch_size", "grad_accum", "max_length"):
        if train_cfg[key] <= 0:
            raise ValueError(f"training.{key} must be positive.")
    records = load_training_records(args.train_path)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "This configuration uses one CUDA GPU per run. Select one device with "
            "CUDA_VISIBLE_DEVICES and launch a single training process."
        )
    total_tokens = (
        train_cfg["max_steps"] * train_cfg["batch_size"]
        * train_cfg["grad_accum"] * train_cfg["max_length"]
    )

    set_seed(train_cfg["seed"])
    tokenizer, model = load_model_and_tokenizer(
        args.base_model or cfg["base_model"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
    )
    print(f"Seed: {train_cfg['seed']}")
    print(f"Processed-token budget: {total_tokens:,}")
    train_ds = prepare_training_dataset(
        records, tokenizer, cfg["chat_template"], total_tokens=total_tokens,
        max_length=train_cfg["max_length"], seed=train_cfg["seed"],
    )
    del records
    actual_tokens = len(train_ds) * train_cfg["max_length"]
    print(f"Packed sequences: {len(train_ds):,}; actual tokens: {actual_tokens:,}")

    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_cfg["target_modules"],
        ),
    )
    model.print_trainable_parameters()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=SFTConfig(
            output_dir=str(args.output_dir),
            seed=train_cfg["seed"],
            data_seed=train_cfg["seed"],
            per_device_train_batch_size=train_cfg["batch_size"],
            gradient_accumulation_steps=train_cfg["grad_accum"],
            gradient_checkpointing=train_cfg["gradient_checkpointing"],
            optim=train_cfg["optim"],
            warmup_ratio=train_cfg["warmup_ratio"],
            max_steps=train_cfg["max_steps"],
            learning_rate=float(train_cfg["learning_rate"]),
            lr_scheduler_type="cosine",
            max_length=train_cfg["max_length"],
            packing=True,
            packing_strategy="wrapped",
            padding_free=False,
            completion_only_loss=True,
            # Input IDs and completion masks are already packed.
            dataset_kwargs={"skip_prepare_dataset": True},
            eval_strategy="no",
            save_strategy="steps",
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            fp16=False,
            bf16=True,
            logging_steps=10,
            report_to="tensorboard",
            push_to_hub=False,
        ),
    )
    resume_from = str(args.resume) if args.resume is not None else None
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(args.output_dir))
    print(f"Saved fine-tuned adapter to {args.output_dir}")

    del trainer, model, tokenizer, train_ds
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("llama", "gemma", "mistral", "qwen"))
    parser.add_argument(
        "--base-model", help="Base model path or ID; defaults to the selected model configuration"
    )
    parser.add_argument(
        "--train-path", required=True, type=Path,
        help="External JSONL file with prepared system, instruction, and output strings",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for adapter weights, training checkpoints, and logs",
    )
    parser.add_argument("--seed", type=int, help="Seed for data shuffling and training")
    parser.add_argument(
        "--resume", type=Path, help="Checkpoint to resume with the same configuration and training file"
    )
    parser.add_argument("--max-steps", type=int, help="Override the total optimizer-step limit")
    parser.add_argument("--save-steps", type=int, help="Override the checkpoint saving interval")
    parser.add_argument("--save-total-limit", type=int, help="Override the number of retained checkpoints")
    args = parser.parse_args()
    run_training(load_config(args.model), args)


if __name__ == "__main__":
    main()
