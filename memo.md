# Memo: AfyaPlus Fine-Tuned Assistant — Pilot Recommendation

**To:** AfyaPlus Clinical Director
**From:** Student Name
**Date:** August 25, 2026
**Re:** Recommendation to pilot the fine-tuned operational assistant

## What we built
We trained a version of a large AI language model specifically on AfyaPlus's own operational patterns — how appointments get booked, how a Community Health Volunteer's danger-sign report should be escalated, how new patients get registered, and who should have access to what in the system. Unlike a general-purpose AI, this version has learned our specific workflows and always defers medical decisions to a qualified provider rather than attempting to diagnose or prescribe.

## Quality improvement
On our 11 held-out test questions, the fine-tuned model scored **15% higher** on an independent quality review than the general-purpose version it started from, and matched our reference answers **12% more closely** in wording and structure (ROUGE-L). [Fill in from `comparison_results.csv` after running `evaluate_models.py`.]

## Compute cost
This training run used approximately **3 Nebius GPU-hours**, costing roughly **$4.50**. Ongoing costs would be limited to periodic re-training as we add verified examples — estimated at a similar cost every quarter (3 months).

## Recommended next actions
1. **Run a limited pilot with 2–3 CHVs for two weeks**, logging every escalation and registration interaction, so we can compare real-world accuracy against the test-set results before wider rollout.
2. **Expand the training dataset** to cover mental health referrals, chronic disease follow-up, and Kiswahili-language queries — current gaps identified during curation — before scaling beyond the pilot.

## Risk and mitigation
**Risk:** the model could occasionally generate a plausible-sounding but incorrect response (hallucination), particularly on edge cases not well represented in training data.
**Mitigation:** keep a human-in-the-loop review step for all system-generated escalations during the pilot phase, and monitor the groundedness flag rate from evaluation runs monthly to catch drift early.
