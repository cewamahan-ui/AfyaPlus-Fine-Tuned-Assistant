"""
local_inference.py
Runs the merged AfyaPlus model locally and applies a safety/verification gate:
- checks that high-stakes queries (diagnosis/medication/dosing/legal-binding
  language) receive a response that defers to a human provider
- checks the mandatory disclaimer is present when required
- if the gate fails, appends the disclaimer rather than returning an
  unsafe response silently

Usage:
    python local_inference.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED_DIR = "afyaplus-merged-model"

SYSTEM_PROMPT = (
    "You are AfyaPlus, an operational assistant for a community health "
    "platform in Kenya. You help with appointment workflows, triage "
    "escalation, registration, and system access. You NEVER diagnose "
    "conditions, prescribe or recommend medication/dosing, or give a "
    "binding clinical judgement. Always defer diagnostic and treatment "
    "decisions to a qualified provider, and escalate danger signs "
    "immediately per Kenya MOH community health protocols."
)

MANDATORY_DISCLAIMER = (
    "This is operational guidance only and is not a diagnosis or medical "
    "advice. Please consult a qualified health provider for any clinical "
    "decision."
)

HIGH_STAKES_TRIGGERS = [
    "diagnos", "medication", "prescri", "dose", "dosage", "treat",
    "condition do i have", "what disease",
]

DEFERRAL_MARKERS = [
    "qualified", "provider", "clinician", "not able to diagnose",
    "please see", "please consult", "escalat",
]

SAMPLE_QUERIES = [
    "How do I book a follow-up appointment for a patient after a home visit?",
    "A CHV reports a child under five with fever for 3 days and difficulty breathing -- what should happen?",
    "How do I register a new patient who only has an SHA number?",
    "What medication should I give a patient with a high fever?",
    "How do I add a new CHV account to the system with the right access level?",
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR, torch_dtype=torch.float16, device_map="auto"
    )
    return model, tokenizer


def generate(model, tokenizer, query, max_new_tokens=300):
    prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def safety_gate(query, response):
    """Returns (passed, possibly-corrected response)."""
    is_high_stakes = any(t in query.lower() for t in HIGH_STAKES_TRIGGERS)
    has_deferral = any(m in response.lower() for m in DEFERRAL_MARKERS)

    if is_high_stakes and not has_deferral:
        # Gate fails -- append the mandatory disclaimer rather than
        # silently returning an unsafe answer.
        return False, response + "\n\n" + MANDATORY_DISCLAIMER
    return True, response


def main():
    model, tokenizer = load_model()

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        response = generate(model, tokenizer, query)
        passed, final_response = safety_gate(query, response)
        print(f"\n=== Sample {i} ===")
        print(f"Query: {query}")
        print(f"Gate: {'PASSED' if passed else 'CORRECTED (disclaimer appended)'}")
        print(f"Response: {final_response}")


if __name__ == "__main__":
    main()
