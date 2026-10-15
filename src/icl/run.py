"""Run in-context learning from local support/query JSONL through a chat API."""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from ..utils.io import PROJECT_ROOT, load_records, resolve_path, write_responses
from ..utils.prompts import load_system_prompt


def build_support_messages(records, system_prompt: str) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for record in records:
        messages.extend((
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["output"]},
        ))
    return messages


def generate(client, model, support_messages, queries, generation):
    for index, query in enumerate(queries, 1):
        response = client.chat.completions.create(
            model=model,
            messages=support_messages + [
                {"role": "user", "content": query["instruction"]},
            ],
            **generation,
        )
        if not response.choices:
            raise RuntimeError(f"Query {index}: the API returned no completion choices")
        choice = response.choices[0]
        yield {
            "query": query,
            "response": choice.message.content,
            "refusal": choice.message.refusal,
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage is not None else None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "icl.yaml")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ICL_API_KEY")
    base_url = os.environ.get("ICL_BASE_URL")
    if not api_key or not base_url:
        parser.error("Set ICL_API_KEY and ICL_BASE_URL in .env or the process environment.")
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg["model"], str) or not cfg["model"].strip():
        parser.error(f"Set model in {args.config}.")
    support = load_records(resolve_path(cfg["support_path"]), ("instruction", "output"))
    queries = load_records(resolve_path(cfg["queries_path"]), ("instruction",))
    support_messages = build_support_messages(support, load_system_prompt(cfg["prompt"]))
    with OpenAI(api_key=api_key, base_url=base_url, timeout=cfg["timeout"]) as client:
        write_responses(
            generate(client, cfg["model"], support_messages, queries, cfg["generation"]),
            resolve_path(cfg["output_dir"]), total=len(queries),
        )


if __name__ == "__main__":
    main()
