"""Upload chat JSONL and submit or inspect OpenAI fine-tuning jobs."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ..utils.io import PROJECT_ROOT


def create_job(client, train_path: Path):
    with train_path.open("rb") as handle:
        training_file = client.files.create(file=handle, purpose="fine-tune")
    print(f"Uploaded file ID: {training_file.id}", flush=True)
    return client.fine_tuning.jobs.create(
        model="gpt-4.1-mini-2025-04-14",
        training_file=training_file.id,
    )


def print_job(job):
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Base model: {job.model}")
    if job.fine_tuned_model:
        print(f"Fine-tuned model: {job.fine_tuned_model}")
    if job.trained_tokens is not None:
        print(f"Trained tokens: {job.trained_tokens:,}")
    if job.error and job.error.message:
        print(f"Error: {job.error.message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create", help="Upload a training file and create a job")
    create_parser.add_argument("--train-path", required=True, type=Path)
    status_parser = commands.add_parser("status", help="Show job status and the trained model ID")
    status_parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in .env or the process environment.")
    with OpenAI() as client:
        if args.command == "create":
            job = create_job(client, args.train_path)
        else:
            job = client.fine_tuning.jobs.retrieve(args.job_id)
        print_job(job)


if __name__ == "__main__":
    main()
