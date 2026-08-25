"""
evaluate_models.py
Compares base LLaMA 3.1 8B vs the fine-tuned (merged) AfyaPlus model on
data/test.jsonl (all 20 held-out examples), producing comparison_results.csv
with ROUGE-L, an LLM-judge score, and a groundedness check.

Usage:
    python evaluate_models.py

Requires an ANTHROPIC_API_KEY environment variable for the LLM judge
(or swap in another judge model of your choice -- keep the rubric
requirements the same: score 1-5 + a groundedness flag).
"""

import csv
import json
import os
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import anthropic

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MERGED_DIR = "afyaplus-merged-model"
TEST_FILE = "data/test.jsonl"
OUT_CSV = "comparison_results.csv"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def load(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    return model, tok


def generate(model, tok, instruction, input_text="", max_new_tokens=300):
    user = instruction if not input_text else f"{instruction}\n\n{input_text}"
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3,
                          do_sample=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def llm_judge(instruction, reference, candidate):
    """Returns (score 1-5, grounded True/False, rationale)."""
    prompt = f"""You are grading an AfyaPlus (Kenyan community health operations assistant) response.

Question: {instruction}
Reference (gold) answer: {reference}
Candidate answer: {candidate}

Score the candidate 1-5 on accuracy and appropriateness relative to the reference
(5 = matches reference quality/content, 1 = wrong or unsafe).
Also flag groundedness: does the candidate avoid inventing clinical facts not
supported by the reference/domain (true/false)?

Respond ONLY as JSON: {{"score": <int>, "grounded": <bool>, "rationale": "<one sentence>"}}"""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip().replace("```json", "").replace("```", "")
    try:
        parsed = json.loads(text)
        return parsed["score"], parsed["grounded"], parsed["rationale"]
    except Exception:
        return None, None, f"PARSE_ERROR: {text[:200]}"


def main():
    with open(TEST_FILE, encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]

    print(f"Loaded {len(test_examples)} test examples")
    if len(test_examples) < 20:
        print("WARNING: rubric expects all 20 test examples evaluated -- check your split sizes.")

    print("Loading base model...")
    base_model, base_tok = load(BASE_MODEL)
    print("Loading fine-tuned model...")
    ft_model, ft_tok = load(MERGED_DIR)

    rows = []
    for i, ex in enumerate(test_examples):
        instr, ref, inp = ex["instruction"], ex["output"], ex.get("input", "")

        base_resp = generate(base_model, base_tok, instr, inp)
        ft_resp = generate(ft_model, ft_tok, instr, inp)

        base_rouge = scorer.score(ref, base_resp)["rougeL"].fmeasure
        ft_rouge = scorer.score(ref, ft_resp)["rougeL"].fmeasure

        base_score, base_grounded, base_rationale = llm_judge(instr, ref, base_resp)
        ft_score, ft_grounded, ft_rationale = llm_judge(instr, ref, ft_resp)

        rows.append({
            "id": i,
            "instruction": instr,
            "reference": ref,
            "base_response": base_resp,
            "ft_response": ft_resp,
            "base_rougeL": round(base_rouge, 3),
            "ft_rougeL": round(ft_rouge, 3),
            "base_judge_score": base_score,
            "ft_judge_score": ft_score,
            "base_grounded": base_grounded,
            "ft_grounded": ft_grounded,
            "improvement": (ft_score or 0) - (base_score or 0),
        })
        print(f"[{i+1}/{len(test_examples)}] base ROUGE-L={base_rouge:.3f} judge={base_score} "
              f"| ft ROUGE-L={ft_rouge:.3f} judge={ft_score}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary + biggest/smallest improvements for Deliverable 4's per-question breakdown
    rows_sorted = sorted(rows, key=lambda r: r["improvement"], reverse=True)
    print("\n=== Top 3 improvements ===")
    for r in rows_sorted[:3]:
        print(f"  #{r['id']}: +{r['improvement']} | {r['instruction'][:70]}")
    print("=== Bottom 3 improvements ===")
    for r in rows_sorted[-3:]:
        print(f"  #{r['id']}: {r['improvement']} | {r['instruction'][:70]}")

    avg_base = sum(r["base_judge_score"] or 0 for r in rows) / len(rows)
    avg_ft = sum(r["ft_judge_score"] or 0 for r in rows) / len(rows)
    print(f"\nAvg judge score -- base: {avg_base:.2f}, fine-tuned: {avg_ft:.2f} "
          f"({(avg_ft-avg_base)/max(avg_base,0.01)*100:.1f}% relative change)")
    print(f"\nWritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
