"""
fine_tune_unsloth.py
FAST QLoRA fine-tuning using Unsloth - 2-5x faster than standard QLoRA.
Trains LLaMA 3.1 8B in ~10-15 minutes on free Colab T4.

Usage on Colab:
    !pip install unsloth
    !python scripts/fine_tune_unsloth.py
"""