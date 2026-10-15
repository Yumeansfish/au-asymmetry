"""Generate responses with an OpenAI fine-tuned model."""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from ..utils.io import PROJECT_ROOT, load_queries, resolve_path, write_responses
from ..utils.prompts import load_system_prompt


def generate(cfg, queries):
    with OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=cfg["timeout"]) as client:
        for index, query in enumerate(queries, 1):
            messages = []
            system = query.get("system", cfg["system_prompt"])
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": query["instruction"]})
            response = client.chat.completions.create(
                model=cfg["model"], messages=messages, **cfg["generation"],
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
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "gpt_inference.yaml",
    )
    parser.add_argument(
        "--prompt", choices=("plain", "utility", "de_alignment"),
        help="Select a system prompt from config/prompts.yaml",
    )
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in .env or the process environment.")
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg["model"], str) or not cfg["model"].strip():
        parser.error(f"Set model in {args.config}.")
    cfg["system_prompt"] = load_system_prompt(args.prompt or cfg["prompt"])
    queries = load_queries(resolve_path(cfg["queries_path"]))
    write_responses(
        generate(cfg, queries), resolve_path(cfg["output_dir"]), total=len(queries),
    )


if __name__ == "__main__":
    main()
