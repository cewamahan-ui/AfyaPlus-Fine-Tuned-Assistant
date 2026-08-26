"""
fine_tune.py
QLoRA fine-tuning of LLaMA 3.1 8B on the curated AfyaPlus dataset.

OPTIMIZED FOR SPEED on Google Colab T4 GPU:
- 2 epochs (not 3) - sufficient for 80 training examples
- max_seq_length=512 (not 1024) - faster processing
- Estimated runtime: ~20-30 minutes on T4 GPU

Usage:
    1. Clone repo: git clone https://github.com/cewamahan-ui/AfyaPlus-Fine-Tuned-Assistant
    2. cd AfyaPlus-Fine-Tuned-Assistant
    3. Runtime > Change runtime type > T4 GPU
    4. Set HF_TOKEN in Secrets or run: from huggingface_hub import login; login()
    5. python scripts/fine_tune.py
"""

import json
import os
import glob
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "afyaplus-lora-adapter"

SYSTEM_PROMPT = (
    "You are AfyaPlus, an operational assistant for a community health "
    "platform in Kenya. You help with appointment workflows, triage "
    "escalation, registration, and system access. You NEVER diagnose "
    "conditions, prescribe or recommend medication/dosing, or give a "
    "binding clinical judgement. Always defer diagnostic and treatment "
    "decisions to a qualified provider, and escalate danger signs "
    "immediately per Kenya MOH community health protocols."
)


def format_example(ex):
    user = ex["instruction"]
    if ex.get("input"):
        user += f"\n\n{ex['input']}"
    text = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{ex['output']}<|eot_id|>"
    )
    return {"text": text}


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"},
    )
    dataset = dataset.map(format_example)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        report_to="none",
        )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=format_example,
        tokenizer=tokenizer,
    )

    checkpoints = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    resume_from = checkpoints[-1] if checkpoints else None
    if resume_from:
        print(f"Resuming from: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nDone! Adapter saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
