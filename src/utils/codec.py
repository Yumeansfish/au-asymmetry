"""Encode/decode lowercase text with substitutions or whole-string reversal."""

import argparse
import json
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "transformations.json"
DEFAULT_TRANSFORMATION = "monoalphabetic_substitution"
TRANSFORMATION_NAMES = (
    "monoalphabetic_substitution",
    "polygraphic_substitution",
    "variable_length_substitution",
    "flip_coding",
)
PERTURBATIONS = ("substitution", "insertion", "deletion")


@dataclass(frozen=True)
class Transformation:
    name: str
    display_name: str
    kind: str
    mapping: dict[str, str]


def load_transformation(name: str = DEFAULT_TRANSFORMATION) -> Transformation:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        configurations = json.load(handle)
    spec = configurations[name]
    if spec["type"] not in {"substitution", "prefix_free_variable_length", "reverse_string"}:
        raise ValueError(f"Unsupported transformation type: {spec['type']}")
    mapping = spec["mapping"]
    if spec["type"] != "reverse_string":
        if set(mapping) != set(string.ascii_lowercase) or any(
            not isinstance(value, str) or not value for value in mapping.values()
        ):
            raise ValueError("A substitution codebook must map a-z to nonempty strings.")
    return Transformation(name, spec["name"], spec["type"], mapping)


def validate_perturbation(transformation_name: str, perturbation: str | None, eta: float):
    if perturbation is not None and perturbation not in PERTURBATIONS:
        raise ValueError(f"Unsupported perturbation: {perturbation}")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be between 0 and 1.")
    if eta > 0:
        if perturbation is None:
            raise ValueError("Specify a perturbation when eta is positive.")
        if transformation_name != DEFAULT_TRANSFORMATION:
            raise ValueError("Edit perturbations require monoalphabetic_substitution.")


def _perturb(text: str, perturbation: str, eta: float, rng: random.Random) -> str:
    result = []
    alphabet = string.ascii_lowercase
    for char in text:
        if char not in alphabet or rng.random() >= eta:
            result.append(char)
        elif perturbation == "deletion":
            continue
        elif perturbation == "substitution":
            result.append(rng.choice(alphabet.replace(char, "")))
        elif perturbation == "insertion":
            result.extend((char, rng.choice(alphabet)))
    return "".join(result)


def encode(
    text: str,
    transformation: Transformation,
    *,
    perturbation: str | None = None,
    eta: float = 0.0,
    rng: random.Random | None = None,
) -> str:
    validate_perturbation(transformation.name, perturbation, eta)
    normalized = text.lower()
    if transformation.kind == "reverse_string":
        transformed = normalized[::-1]
    else:
        transformed = "".join(transformation.mapping.get(char, char) for char in normalized)
    if eta > 0:
        transformed = _perturb(transformed, perturbation, eta, rng or random.Random())
    return transformed


def decode(text: str, transformation: Transformation) -> str:
    normalized = text.lower()
    if transformation.kind == "reverse_string":
        return normalized[::-1]
    inverse = {value: key for key, value in transformation.mapping.items()}
    units = sorted(inverse, key=len, reverse=True)
    result = []
    position = 0
    while position < len(normalized):
        for unit in units:
            if normalized.startswith(unit, position):
                result.append(inverse[unit])
                position += len(unit)
                break
        else:
            result.append(normalized[position])
            position += 1
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    encode_parser = commands.add_parser("encode", help="Transform text, optionally adding noise")
    decode_parser = commands.add_parser("decode", help="Invert the selected transformation")
    for command in (encode_parser, decode_parser):
        command.add_argument(
            "--transformation", default=DEFAULT_TRANSFORMATION, choices=TRANSFORMATION_NAMES
        )
        command.add_argument("--text", help="Input text; defaults to stdin")
    encode_parser.add_argument("--perturbation", choices=PERTURBATIONS)
    encode_parser.add_argument("--eta", type=float, default=0.0, help="Post-encoding noise probability")
    encode_parser.add_argument("--seed", type=int, default=42, help="Random seed for perturbations")
    args = parser.parse_args()
    transformation = load_transformation(args.transformation)
    text = args.text if args.text is not None else sys.stdin.read()
    if args.operation == "encode":
        result = encode(
            text, transformation, perturbation=args.perturbation, eta=args.eta,
            rng=random.Random(args.seed),
        )
    else:
        result = decode(text, transformation)
    sys.stdout.write(result)


if __name__ == "__main__":
    main()
