# Model Selection and Architecture

## Objective
Document model architecture decisions for the LLM pipeline implementation.

## Current Donovan Baseline
We will start with a compact decoder-style transformer classifier implemented in PyTorch under `src/llm_pipeline/`.

The model is intentionally small so it can train in Google Colab while still demonstrating the major LLM architecture pieces from the assignment:
- token embeddings
- positional embeddings
- causal self-attention
- transformer blocks
- feed-forward layers
- classification head

The task is tweet sentiment classification with three output labels: `negative`, `neutral`, and `positive`.

## Decision Scope
1. Use-case fit and constraint analysis
2. Build-from-scratch vs adapt pre-trained model rationale
3. Transformer architecture details
- Attention type
- Embeddings
- Feed-forward blocks
- Context length

4. Resource profile
- Expected compute/memory needs
- Batch-size and precision constraints

## Deliverables
- Chosen architecture with clear justification: compact decoder-style transformer classifier.
- Minimal config spec: default values live in `src/llm_pipeline/train.py`.
- Known tradeoffs and risks: this is a scaled educational model, not a production social media sentiment model.

## Handoff Contract to `training/`
Provide:
1. Final model config
2. Initialization/loading strategy
3. Required hyperparameter ranges

## Default Config
- `max_len`: 64
- `hidden_size`: 128
- `num_layers`: 2
- `num_heads`: 4
- `ff_size`: 256
- `dropout`: 0.1
