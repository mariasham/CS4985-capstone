# Team Assignments and Handoff Plan

This file defines ownership boundaries so 3 contributors can work in parallel and merge cleanly.

## Donovan - Model and Training Bootstrap
Primary folders:
- `model/`
- `training/`

Responsibilities:
1. Choose baseline transformer approach (from-scratch mini-transformer or adapted open model)
2. Implement model config and training entrypoints
3. Establish fine-tuning loop and logging output format
4. Document reproducible run commands

Handoff outputs:
- `model/model_selection.md` completed with architecture rationale
- `training/fine_tuning.md` completed with runbook and expected artifacts

## Maria - Data and Preprocessing
Primary folders:
- `data/`
- `preprocessing/`

Responsibilities:
1. Finalize the tweet sentiment dataset source, license, and schema
2. Confirm the label set is `positive`, `negative`, and `neutral`
3. Build cleaning pipeline and tokenization decisions
4. Document train/validation/test split strategy
5. Provide reproducible preprocessing steps

Handoff outputs:
- `data/dataset_description.md` with dataset card details
- `preprocessing/preprocessing_steps.md` with deterministic workflow

## Alec - Evaluation and Deployment
Primary folders:
- `evaluation/`
- `deployment/`

Responsibilities:
1. Define metrics and benchmarks
2. Build evaluation reporting template
3. Provide baseline deployment architecture (API/service)
4. Add monitoring and maintenance checklist

Handoff outputs:
- `evaluation/evaluation.md` with metrics + error analysis rubric
- `deployment/deployment.md` with deployment plan and SLA-oriented checks

## Integration Rules
1. Do not modify another contributor's primary files unless coordinated.
2. Keep changes small and commit often.
3. Update docs first when assumptions change.
4. Every PR must include: what changed, outputs generated, and unresolved blockers.
