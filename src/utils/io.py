"""Read local JSONL inputs and write generated responses."""

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_records(path: Path, required_fields: tuple[str, ...]) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or any(
                not isinstance(record.get(field), str) or not record[field].strip()
                for field in required_fields
            ):
                fields = ", ".join(required_fields)
                raise ValueError(f"{path}:{line_number}: expected nonempty {fields} strings")
            records.append(record)
    if not records:
        raise ValueError(f"Empty input file: {path}")
    return records


def load_queries(path: Path) -> list[dict]:
    queries = load_records(path, ("instruction",))
    for index, query in enumerate(queries, 1):
        if "system" in query and not isinstance(query["system"], str):
            raise ValueError(f"{path}: record {index}: system must be a string")
    return queries


def write_responses(results, output_dir: Path, *, total: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"responses_{timestamp}.jsonl"
    with output_path.open("x", encoding="utf-8") as handle:
        for index, result in enumerate(results, 1):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"Completed query {index}/{total}", flush=True)
    print(f"Saved responses to {output_path}")
    return output_path
