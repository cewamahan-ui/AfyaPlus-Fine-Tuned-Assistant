"""
merge_model.py
Merges the trained LoRA adapter back into the base LLaMA 3.1 8B weights.

Usage:
    python scripts/merge_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Auto-detect Drive if running on Colab
if os.path.exists("/content/drive/MyDrive/afyaplus-capstone/"):
    ADAPTER_DIR = "/content/drive/MyDrive/afyaplus-capstone/afyaplus-lora-adapter"
    MERGED_DIR = "/content/drive/MyDrive/afyaplus-capstone/afyaplus-merged-model"
    # Also copy to local for easy access
    import shutil
    if os.path.exists(ADAPTER_DIR):
        shutil.copytree(ADAPTER_DIR, "afyaplus-lora-adapter", dirs_exist_ok=True)
        print(f"Copied adapter to local: afyaplus-lora-adapter/")
else:
    ADAPTER_DIR = "afyaplus-lora-adapter"
    MERGED_DIR = "afyaplus-merged-model"


def main():
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}...")
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    print("Done! Merged model saved.")


if __name__ == "__main__":
    main()