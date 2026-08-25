"""
merge_model.py
Merges the trained LoRA adapter back into the base LLaMA 3.1 8B weights.

Usage:
    python scripts/merge_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
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

"""
merge_model.py
Merges the trained LoRA adapter back into the base LLaMA 3.1 8B weights
so you can run plain (non-PEFT) local inference.

Usage (after fine_tune.py has produced afyaplus-lora-adapter/):
    python merge_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Must match OUTPUT_DIR in fine_tune.py. Mount Drive first in your notebook:
# from google.colab import drive; drive.mount('/content/drive')
ADAPTER_DIR = "/content/drive/MyDrive/afyaplus-capstone/afyaplus-lora-adapter"
MERGED_DIR = "/content/drive/MyDrive/afyaplus-capstone/afyaplus-merged-model"


def main():
    print("Loading base model in fp16 (merge requires full precision, not 4-bit)...")
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

    print("Done. This merged model is a standalone checkpoint -- no PEFT needed to load it.")


if __name__ == "__main__":
    main()
