"""
data_prep.py
Validates the curated AfyaPlus JSONL dataset and splits it 80/10/10
into data/train.jsonl, data/val.jsonl, data/test.jsonl.

Usage:
    python data_prep.py --input data/afyaplus_full.jsonl --outdir data/

Expects each line to be a JSON object with:
    {"instruction": str, "input": str, "output": str}
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Rough token estimate (no tokenizer dependency needed for a sanity check;
# swap in the real LLaMA tokenizer before final submission for exact counts).
def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)  # ~1.33 tokens per word, rough

REQUIRED_KEYS = {"instruction", "output"}
MIN_TOKENS = 8
MAX_TOKENS = 1024
DISCLAIMER_HINTS = [
    "not able to diagnose", "qualified", "provider", "clinician",
    "does not mean a diagnosis", "please see", "escalat", "referral",
]


def validate_examples(examples):
    errors = []
    warnings = []
    token_counts = []

    for i, ex in enumerate(examples):
        missing = REQUIRED_KEYS - ex.keys()
        if missing:
            errors.append(f"Line {i}: missing keys {missing}")
            continue

        instr, out = ex["instruction"], ex["output"]
        if not isinstance(instr, str) or not isinstance(out, str):
            errors.append(f"Line {i}: instruction/output must be strings")
            continue
        if len(instr.strip()) == 0 or len(out.strip()) == 0:
            errors.append(f"Line {i}: empty instruction or output")
            continue

        total_tokens = estimate_tokens(instr) + estimate_tokens(ex.get("input", "")) + estimate_tokens(out)
        token_counts.append(total_tokens)
        if total_tokens < MIN_TOKENS:
            warnings.append(f"Line {i}: very short example ({total_tokens} est. tokens)")
        if total_tokens > MAX_TOKENS:
            warnings.append(f"Line {i}: very long example ({total_tokens} est. tokens)")

        # Safety gate: high-stakes-sounding answers should carry a disclaimer/
        # deferral cue. This is a heuristic check, not a substitute for manual review.
        if any(k in instr.lower() for k in ["diagnos", "medication", "treat", "dose", "prescri"]):
            if not any(h in out.lower() for h in DISCLAIMER_HINTS):
                warnings.append(f"Line {i}: high-stakes topic without an obvious deferral/disclaimer cue -- manually verify")

    return errors, warnings, token_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/afyaplus_full.jsonl")
    ap.add_argument("--outdir", default="data/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: {path} not found. Curate/merge your examples into this file first "
              f"(see data/afyaplus_seed.jsonl for the starter set + format).")
        sys.exit(1)

    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    errors, warnings, token_counts = validate_examples(examples)

    print(f"Loaded {len(examples)} examples")
    print(f"Errors: {len(errors)}  Warnings: {len(warnings)}")
    for e in errors:
        print("  ERROR:", e)
    for w in warnings:
        print("  WARN:", w)

    if token_counts:
        print(f"Token estimate range: {min(token_counts)}-{max(token_counts)}, "
              f"avg {sum(token_counts)//len(token_counts)}")

    if errors:
        print("\nFix errors before proceeding to fine-tuning.")
        sys.exit(1)

    if len(examples) < 100:
        print(f"\nNOTE: rubric requires 100+ examples for full marks on Deliverable 1. "
              f"You currently have {len(examples)}.")

    random.seed(args.seed)
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test)]:
        with open(outdir / name, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSplit sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"Written to {outdir}/train.jsonl, val.jsonl, test.jsonl")


if __name__ == "__main__":
    main()
