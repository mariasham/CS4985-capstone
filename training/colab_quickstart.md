# Donovan Colab Quickstart

Use this as the copy-paste runbook for Google Colab.

## 1. Runtime
In Colab, use `Runtime > Change runtime type > GPU`.

## 2. Clone and Install
```python
!git clone <repo-url>
%cd CS4985-capstone
!pip install -r requirements-colab.txt
```

## 3. Smoke Test With Sample Data
```python
!PYTHONPATH=src python -m llm_pipeline.train \
  --data examples/sample_tweet_sentiment.csv \
  --output-dir artifacts/donovan-smoke-test \
  --epochs 5
```

## 4. Predict From the Smoke-Test Checkpoint
```python
!PYTHONPATH=src python -m llm_pipeline.predict \
  --checkpoint artifacts/donovan-smoke-test/checkpoint.pt \
  --tokenizer artifacts/donovan-smoke-test/tokenizer.json \
  --text "I love how this project is coming together"
```

## 5. Train With Maria's Dataset
Maria's final CSV should have:
- `text`: tweet text
- `label`: one of `positive`, `negative`, or `neutral`

```python
!PYTHONPATH=src python -m llm_pipeline.train \
  --data data/processed/tweet_sentiment.csv \
  --output-dir artifacts/donovan-final-run \
  --epochs 10 \
  --batch-size 32
```

## 6. Pull Artifacts Back Into the Repo
The training script writes:
- `checkpoint.pt`
- `tokenizer.json`
- `metrics.json`

Checkpoints are ignored by git by default. For final submission, either document the Colab/Drive location of the final checkpoint or commit only a small representative artifact if the file size is reasonable.
