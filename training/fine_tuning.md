# Training and Fine-Tuning Plan

## Objective
Implement and document training workflow aligned with capstone expectations.

## Donovan Training Workflow
Training should run in Google Colab so we can use better GPU resources than a local laptop.

Colab setup:
```bash
git clone <repo-url>
cd CS4985-capstone
pip install -r requirements-colab.txt
```

Smoke test with the sample dataset:
```bash
PYTHONPATH=src python -m llm_pipeline.train \
  --data examples/sample_tweet_sentiment.csv \
  --output-dir artifacts/donovan-smoke-test \
  --epochs 5
```

Run a prediction from the saved checkpoint:
```bash
PYTHONPATH=src python -m llm_pipeline.predict \
  --checkpoint artifacts/donovan-smoke-test/checkpoint.pt \
  --tokenizer artifacts/donovan-smoke-test/tokenizer.json \
  --text "I love how this project is coming together"
```

When Maria's processed dataset is ready, replace `examples/sample_tweet_sentiment.csv` with her final CSV path. The expected text column is `text`. The label column may be either `label` or `sentiment`, and values must be one of `positive`, `negative`, or `neutral`.

## Stage Coverage
1. Pre-training (scaled simulation acceptable)
- State objective (next-token prediction)
- Document why full pre-training is out-of-scope for class compute
- Define mini pre-training experiment if used

2. Fine-tuning
- Supervised fine-tuning setup
- Task-specific dataset mapping
- Hyperparameters and scheduler choices

## Required Logging
1. Run command(s)
2. Seed values
3. Checkpoint cadence
4. Loss/metric tracking
5. Runtime and hardware summary

## Deliverables
- Reproducible training runbook
- Checkpoint and log artifact map
- First-pass training outcomes and blockers

## Handoff Contract to `evaluation/`
Provide:
1. Best checkpoint identifier
2. Inference procedure
3. Expected input/output format

## Current Artifact Contract
Training writes these files to `--output-dir`:
- `checkpoint.pt`: PyTorch model state, config, labels, and run args
- `tokenizer.json`: tokenizer vocabulary
- `metrics.json`: per-epoch validation loss, accuracy, and macro F1

These generated artifacts are ignored by git by default because checkpoints can become large. For final submission, include only small representative artifacts or document where the trained checkpoint is stored.
