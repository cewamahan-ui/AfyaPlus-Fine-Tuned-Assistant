"""
evaluate_models.py
Compares base LLaMA 3.1 8B vs the fine-tuned AfyaPlus model on data/test.jsonl,
producing comparison_results.csv with ROUGE-L and Gemini LLM judge evaluation.

FREE EVALUATION: Uses Google Gemini API (free tier available)

Usage:
    python evaluate_models.py

Requires GEMINI_API_KEY environment variable (get free key at https://aistudio.google.com/)
"""

import csv
import json
import os
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MERGED_DIR = "afyaplus-merged-model"
TEST_FILE = "data/test.jsonl"
OUT_CSV = "comparison_results.csv"

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
    """Returns (score 1-5, grounded True/False, rationale) using Gemini."""
    import google.generativeai as genai
    
    prompt = f"""You are grading an AfyaPlus (Kenyan community health operations assistant) response.

Question: {instruction}
Reference answer: {reference}
Candidate answer: {candidate}

Score the candidate 1-5 on accuracy and appropriateness (5=excellent, 1=poor/unsafe).
Also check if grounded: does it avoid inventing clinical facts?

Respond ONLY as JSON: {{"score": <int>, "grounded": <bool>, "rationale": "<brief reason>"}}"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Extract JSON from response
        if '{' in text and '}' in text:
            json_str = text[text.index('{'):text.rindex('}')+1]
            parsed = json.loads(json_str)
            return parsed["score"], parsed.get("grounded", True), parsed.get("rationale", "")
        return None, None, f"PARSE_ERROR: {text[:100]}"
    except Exception as e:
        return None, None, f"ERROR: {str(e)[:100]}"


def main():
    # Setup Gemini
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY") or input("Enter GEMINI_API_KEY: ")
    genai.configure(api_key=api_key)

    with open(TEST_FILE, encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]

    print(f"Loaded {len(test_examples)} test examples")

    print("Loading base model...")
    base_model, base_tok = load(BASE_MODEL)
    print("Loading fine-tuned model...")
    ft_model, ft_tok = load(MERGED_DIR)

    rows = []
    for i, ex in enumerate(test_examples):
        instr, ref, inp = ex["instruction"], ex["output"], ex.get("input", "")

        print(f"[{i+1}/{len(test_examples)}] Generating responses...")
        base_resp = generate(base_model, base_tok, instr, inp)
        ft_resp = generate(ft_model, ft_tok, instr, inp)

        base_rouge = scorer.score(ref, base_resp)["rougeL"].fmeasure
        ft_rouge = scorer.score(ref, ft_resp)["rougeL"].fmeasure

        print(f"  Base ROUGE-L={base_rouge:.3f} | FT ROUGE-L={ft_rouge:.3f}")
        
        # LLM Judge evaluation
        base_score, base_grounded, base_rationale = llm_judge(instr, ref, base_resp)
        ft_score, ft_grounded, ft_rationale = llm_judge(instr, ref, ft_resp)
        
        print(f"  Base judge={base_score} | FT judge={ft_score}")

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

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    rows_sorted = sorted(rows, key=lambda r: r["improvement"] or 0, reverse=True)
    print("\n=== Top 3 improvements ===")
    for r in rows_sorted[:3]:
        print(f"  #{r['id']}: +{r['improvement']} | {r['instruction'][:60]}")
    print("=== Bottom 3 ===")
    for r in rows_sorted[-3:]:
        print(f"  #{r['id']}: {r['improvement']} | {r['instruction'][:60]}")

    valid_rows = [r for r in rows if r['ft_judge_score'] is not None]
    if valid_rows:
        avg_base = sum(r['base_judge_score'] or 0 for r in valid_rows) / len(valid_rows)
        avg_ft = sum(r['ft_judge_score'] or 0 for r in valid_rows) / len(valid_rows)
        print(f"\nAvg judge score -- base: {avg_base:.2f}, fine-tuned: {avg_ft:.2f}")
    
    print(f"\nResults saved to {OUT_CSV}")


if __name__ == "__main__":
    main()

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
