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

    # OPTIMIZED: 2 epochs, max_seq_length=512
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
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
        dataset_text_field="text",
        max_seq_length=512,
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

    with open(f"{OUTPUT_DIR}/hyperparameters.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=2, default=str)

    print(f"\nDone! Adapter saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

"""
fine_tune.py
QLoRA fine-tuning of LLaMA 3.1 8B on the curated AfyaPlus dataset.

FREE ALTERNATIVES TO NEBIOUS:
1. Google Colab (FREE): Runtime > Change runtime type > T4 GPU
   - Mount Google Drive first: from google.colab import drive; drive.mount('/content/drive')
   - Set OUTPUT_DIR to /content/drive/MyDrive/afyaplus-capstone/
   - 16GB VRAM T4 GPU, fp16 precision
   
2. Kaggle Notebooks (FREE): Accelerator > GPU T4 x2
   - Similar setup to Colab

3. RunPod (FREE TIER): Community Cloud with T4 GPUs
   - Use persistent storage for checkpoints

Written for Google Colab's free-tier T4 GPU (16GB VRAM, Turing architecture --
no native bf16 support, hence fp16 throughout). Run this in a Colab notebook
cell (or %run fine_tune.py) with Runtime > Change runtime type > T4 GPU set.

IMPORTANT -- mount Google Drive FIRST in your notebook, before running this,
so checkpoints survive a session disconnect:

    from google.colab import drive
    drive.mount('/content/drive')

Then set OUTPUT_DIR below to a path under /content/drive/MyDrive/... so
nothing is lost if Colab kicks you off mid-run. Checkpoints save every 10
steps (see save_steps) specifically because free-tier sessions can drop with
no warning -- you want to be able to resume, not restart from zero.

To resume after a disconnect: rerun this script -- it auto-detects the most
recent checkpoint under OUTPUT_DIR and continues from there.
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
# Point this at Drive so checkpoints survive a Colab disconnect. Adjust the
# path to wherever you've mounted/placed your project folder in Drive.
OUTPUT_DIR = "/content/drive/MyDrive/afyaplus-capstone/afyaplus-lora-adapter"

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
    # --- 4-bit quantization config ---
    # WHY: QLoRA loads the base 8B model in 4-bit (NF4) so it fits on a single
    # mid-range GPU (~6GB base weights vs ~16GB in fp16). Compute dtype is
    # fp16, not bf16: Colab's free-tier T4 is Turing architecture and does
    # not have native bf16 Tensor Core support -- bf16 would silently run
    # slow (or error) on this GPU. This is the standard QLoRA recipe
    # (Dettmers et al.), adapted for T4.
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

    # --- LoRA config ---
    # r=16: enough capacity to learn a narrow operational domain (triage/registration
    #       workflows) without the overfitting risk a larger r brings on a ~100-300
    #       example dataset.
    # alpha=32 (2x r): standard scaling ratio that keeps the LoRA update magnitude
    #       reasonable relative to the frozen base weights.
    # target_modules: attention projections are where domain-specific instruction-
    #       following behaviour is most effectively injected for LLaMA-family models.
    # dropout=0.05: light regularization given the small dataset size.
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

    # --- Training arguments ---
    # epochs=3: small domain dataset (100-300 examples) needs several passes to
    #       converge, but >3-4 risks overfitting/memorization on this scale.
    # batch_size=2 + grad_accum=8 -> effective batch 16: fits an 8B model in 4-bit
    #       on a single GPU while keeping the effective batch size reasonable for
    #       stable gradients.
    # lr=2e-4: standard LoRA learning rate -- higher than full fine-tuning because
    #       only adapter weights are updated.
    # cosine schedule + warmup: avoids early-training instability, then decays
    #       smoothly so the model doesn't overshoot late in training.
    # eval_strategy="steps", eval_steps small: dataset is small, so we want
    #       frequent validation-loss checkpoints to catch overfitting early
    #       (see Tuesday's loss-curve patterns).
    # gradient_checkpointing=True: trades ~20% more compute time for a large
    # VRAM reduction (activations aren't all held in memory at once) -- needed
    # headroom on a 16GB T4 that a 24-48GB cloud GPU wouldn't require.
    # save_steps=10 (down from 20): free-tier Colab sessions can disconnect
    # without warning, so checkpoint often -- losing 10 steps of progress is
    # tolerable, losing 20+ minutes of a 12-hour-capped session is not.
    # fp16=True (not bf16): matches the T4's Turing architecture.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
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
        dataset_text_field="text",
        max_seq_length=1024,
    )

    # Resume from the latest checkpoint if this is a re-run after a Colab
    # disconnect. If OUTPUT_DIR has no checkpoints yet, this just trains
    # from scratch as normal.
    checkpoints = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    resume_from = checkpoints[-1] if checkpoints else None
    if resume_from:
        print(f"Found existing checkpoint, resuming from: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    # trainer_state.json (loss history) is written automatically to OUTPUT_DIR
    # by the Trainer -- this is your Deliverable 2 loss-curve source.
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/hyperparameters.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=2, default=str)

    print(f"\nDone. Adapter + trainer_state.json saved to {OUTPUT_DIR}/")
    print("This is on Drive, so it's safe even after the Colab runtime disconnects.")


if __name__ == "__main__":
    main()
