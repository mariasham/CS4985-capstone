# Use Case Definition

## Problem Statement
This project will build a small transformer-based LLM pipeline for tweet sentiment classification. Given the text of a tweet, the model should classify the sentiment as `positive`, `negative`, or `neutral`.

This use case is appropriate for a capstone-scale implementation because it has a clear supervised learning objective, publicly available labeled datasets, straightforward evaluation metrics, and a simple deployment interface.

## Required Decisions
1. Target users: people or systems that need quick sentiment summaries from short social media text.
2. Primary task: three-class text classification over tweets.
3. Labels: `positive`, `negative`, `neutral`.
4. Success criteria: demonstrate the full LLM lifecycle with reproducible preprocessing, model implementation, training, evaluation, and deployment.
5. Pipeline justification: the assignment asks us to implement the LLM development workflow, so we will build and document each stage rather than relying only on prompting an external model.

## Constraints
- Available compute/time for a class project
- Dataset availability and licensing
- Inference latency expectations

## Deliverables
- One-paragraph use case statement: completed above.
- Functional requirements: accept tweet text as input and return one of `positive`, `negative`, or `neutral`.
- Non-functional requirements: keep the model small enough for local/classroom training and inference.
- Evaluation success thresholds: use validation/test accuracy and macro F1 as primary metrics, with final thresholds to be set after the dataset baseline is known.
