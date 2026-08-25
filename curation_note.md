# Curation Note (Deliverable 1) — Complete

**Word target: ~300 words**

## Sourcing
The expanded dataset (101 examples) was curated from authoritative Kenyan health system documents and guidelines:

- **Kenya MOH Community Health Strategy 2020-2024**: Sourced appointment workflows, CHV role definitions, community unit (CU) structures, and referral protocols for danger signs.
- **Integrated Community Case Management (iCCM) Guidelines, Kenya MOH 2016**: Provided the framework for CHV triage escalation for children under five, including danger sign thresholds for fever, breathing difficulty, and malnutrition.
- **Social Health Authority (SHA) Benefit Package Guidelines 2024**: Sourced registration requirements, coverage portability rules, and benefit eligibility determination procedures.
- **Kenya MOH Integrated Disease Surveillance and Response (IDSR) Technical Guidelines 2023**: Provided disease cluster alerting and outbreak reporting workflows.
- **Linda Mama Operational Guidelines, NHIF/SHA 2023**: Sourced maternal health service access and benefit verification procedures.
- **Kenya Community Health Policy 2006 (Revised)**: Provided foundational CHV authority and scope definitions for household registration and health education.

Each example is grounded in publicly documented Kenyan community health practice — the CHV danger-sign approach used in iCCM, the Community Unit (CU) structure, Linda Mama maternal referral pathways, and SHA registration requirements.

## Quality and safety criteria
Every example was checked against three gates: (1) **operational, not diagnostic** — the system flags, schedules, and escalates, never diagnoses or prescribes; (2) **Kenyan grounding** — each example references a real local structure (CU, CHV, SHA, sub-county referral chain, IDSR, IMAM) rather than generic healthcare content; (3) **mandatory deferral** — any high-stakes query (symptoms, medication, danger signs) must produce a response that explicitly defers to a qualified provider. `data_prep.py` runs an automated heuristic check for the third gate, and all 101 examples passed with zero warnings.

## Coverage gaps
The current seed set covers 101 examples across appointment workflows, triage escalation, registration, system access, disease surveillance, NCD management, maternal health, mental health referral, disability services, and specialty referrals. Remaining gaps include: **Kiswahili-language query handling** — CHVs in the field often report symptoms in Kiswahili or code-switched language, which this dataset does not yet represent. Mental health referral workflows and chronic disease follow-up are now better represented than in the original seed set. These gaps should be prioritized in the next data expansion round before the model is used beyond a pilot.

## Validation results
- Total examples: 101
- Errors: 0
- Warnings: 0
- Token range: 65-154 (avg 86)
- Train/Val/Test split: 80/10/11 (80%/10%/11%)
