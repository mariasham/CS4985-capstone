# Preprocessing Steps

## Objective
Define reproducible preprocessing from raw data to training-ready inputs.

## Required Steps
1. Data cleaning
- Remove malformed rows and duplicates
- Normalize text (whitespace/casing policy)
- Handle missing labels and outliers

2. Tokenization
- Tokenizer choice and reason
- Vocabulary/token limits
- Truncation and padding policy

3. Dataset transformation
- Convert to model input format
- Create train/validation/test artifacts
- Record deterministic seed values

## Deliverables
- Ordered preprocessing runbook
- Final artifact format and paths
- Validation checks (row counts, label balance)

## Handoff Contract to `model/` and `training/`
Provide:
1. Final feature schema
2. Tokenized example format
3. Data loading assumptions
