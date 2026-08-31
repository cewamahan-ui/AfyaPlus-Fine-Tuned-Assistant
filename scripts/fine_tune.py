"""
fine_tune.py
QLoRA fine-tuning of LLaMA 3.1 8B on the AfyaPlus dataset.
OPTIMIZED: batch_size=1, max_steps=8, paged optimizer -> ~10 mins
"""

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

MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B-Instruct"
if os.path.exists("/content/drive/MyDrive/afyaplus-capstone/"):
    OUTPUT_DIR = "/content/drive/MyDrive/afyaplus-capstone/afyaplus-lora-adapter"
else:
    OUTPUT_DIR = "afyaplus-lora-adapter"
print(f"Output dir: {OUTPUT_DIR}")

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
    return text

def main():
    # 4-bit quantization for memory efficiency
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

    # OPTIMIZED: batch=1, grad_accum=8, max_steps=10 -> ~15 mins on T4
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_steps=10,
        logging_steps=2,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=5,
        save_total_limit=1,
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        formatting_func=format_example,
        max_seq_length=256,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n Done! Adapter saved to {OUTPUT_DIR}/")
    print("Next: python scripts/merge_model.py")

if __name__ == "__main__":
    main()
