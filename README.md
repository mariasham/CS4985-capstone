# CS4985 Capstone: Building an LLM Pipeline from Scratch

This repository implements the Spring 2026 CS4985 final project workflow from the provided assignment PDF (`/Users/donovanbonner/Downloads/Capstone.pdf`).

The project is research-oriented and follows the full LLM lifecycle:
1. Define use case
2. Data preparation and curation
3. Model architecture implementation
4. Training (pre-training concept + task fine-tuning)
5. Evaluation
6. Deployment and monitoring

## Project Goal
Build a practical, end-to-end prototype that classifies tweet sentiment as `positive`, `negative`, or `neutral` while demonstrating each LLM pipeline stage clearly, even if we use scaled-down compute and datasets.

## Team Collaboration Model
We are splitting the work so three contributors can progress in parallel.

- Donovan: model setup + training bootstrap
- Maria: tweet sentiment dataset curation + preprocessing pipeline
- Alec: evaluation + deployment baseline

Detailed ownership and handoff contracts are in `TEAM_ASSIGNMENTS.md`.

## Repository Structure
- `data/`: dataset selection, schema, sourcing notes
- `use_case_definition.md`: problem framing and success criteria
- `preprocessing/`: cleaning, tokenization, dataset build steps
- `model/`: architecture decisions and implementation notes
- `training/`: pre-training simulation and fine-tuning plan
- `evaluation/`: metrics, benchmark strategy, error analysis
- `deployment/`: inference API/service and monitoring plan
- `src/llm_pipeline/`: Donovan's model, tokenizer, training, and prediction code
- `examples/`: small sample data for smoke tests before the final dataset is ready

## Handoff Artifacts (Cross-Team Contracts)
To reduce merge conflict and blocking, each stage produces concrete outputs:

1. `data/` hands off dataset card + raw/processed split description
2. `preprocessing/` hands off CSV files with `text` and either `label` or `sentiment`
3. `model/` hands off model config and training entrypoints
4. `training/` hands off checkpoints/logs + reproducible command history
5. `evaluation/` hands off metrics report + failure case analysis
6. `deployment/` hands off runnable inference endpoint + basic monitoring checklist

## Immediate Plan
1. Finalize use case and target task constraints
2. Lock dataset + preprocessing format
3. Complete model bootstrap and training loop
4. Run first evaluation pass and document results
5. Deploy simple inference endpoint

## Donovan Colab Smoke Test
Full copy-paste Colab commands are in `training/colab_quickstart.md`.

```bash
pip install -r requirements-colab.txt
PYTHONPATH=src python -m llm_pipeline.train --data examples/sample_tweet_sentiment.csv --epochs 5
PYTHONPATH=src python -m llm_pipeline.predict --text "I love how this project is coming together"
```

## Branching Workflow
- `main`: stable shared baseline
- feature branches: short-lived branches per workstream
- current integration branch for this rework: `codex/capstone-pipeline-rework`

## Source References from Assignment
- Video: https://youtu.be/quh7z1q7-uc?si=f_19CmxFQkgcYV-m
- Additional references are listed in `/Users/donovanbonner/Downloads/Capstone.pdf` pages 2-3.
