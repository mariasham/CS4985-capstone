# Evaluation Plan

## Objective
Evaluate model quality with both task metrics and qualitative failure analysis.

## Required Metrics
1. Core task metrics (for selected use case)
- Example: accuracy, precision, recall, F1

2. Robustness checks
- Class imbalance behavior
- Long-input/edge-case behavior

3. Error analysis
- Common failure categories
- Representative examples

## Deliverables
- Evaluation protocol
- Metrics report template
- Error analysis section with actionable follow-ups

## Handoff Contract to `deployment/`
Provide:
1. Selected model/checkpoint
2. Latency/throughput expectations from eval tests
3. Known limitations to surface in API docs
