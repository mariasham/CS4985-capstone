# Deployment and Monitoring

## Objective
Run a lightweight FastAPI inference server that exposes the trained tweet sentiment
classifier over HTTP. Designed for classroom / local use with a clear path to a
cloud deployment if needed.

---

## Baseline Deployment Scope

### Stack
| Component | Choice |
|---|---|
| Inference server | FastAPI + Uvicorn |
| Model serialisation | PyTorch checkpoint (`best_checkpoint.pt`) |
| Tokenizer | `tokenizer.json` (word-level, loaded at startup) |
| Runtime | Python 3.10+, CPU or CUDA |

### Installation

```bash
pip install fastapi uvicorn torch
# (torch already installed from training requirements)
```

### Running the Server

```bash
# Default: localhost:8000
PYTHONPATH=src python -m llm_pipeline.app \
    --checkpoint artifacts/tweet-sentiment-model/best_checkpoint.pt \
    --tokenizer  artifacts/tweet-sentiment-model/tokenizer.json

# Bind to all interfaces (e.g. for a shared classroom server):
PYTHONPATH=src python -m llm_pipeline.app \
    --checkpoint artifacts/tweet-sentiment-model/best_checkpoint.pt \
    --tokenizer  artifacts/tweet-sentiment-model/tokenizer.json \
    --host 0.0.0.0 --port 8000
```

---

## API Reference

### `GET /health`
Liveness check. Returns `{"status": "ok", "device": "cpu"}` when the model is ready.
Returns HTTP 503 if the model has not finished loading.

### `GET /model-info`
Returns model config, vocab size, label mapping, and the checkpoint path loaded at startup.
Useful for debugging version mismatches.

### `POST /predict`
Classify a single tweet.

**Request body:**
```json
{ "text": "I love this so much!" }
```

**Response:**
```json
{
  "prediction": {
    "text": "I love this so much!",
    "label": "positive",
    "confidence": 0.9312,
    "probabilities": {
      "negative": 0.0241,
      "neutral":  0.0447,
      "positive": 0.9312
    }
  },
  "latency_ms": 3.7
}
```

### `POST /predict-batch`
Classify up to 64 tweets in a single forward pass.

**Request body:**
```json
{ "texts": ["Great day!", "Terrible service.", "Meh, it was okay."] }
```

**Response:** Same structure as `/predict` but with a `predictions` array, `count`, and a
single `latency_ms` for the full batch.

---

## Input / Output Schema Summary

| Field | Type | Notes |
|---|---|---|
| `text` (input) | string | 1–1000 chars. Longer inputs will be tokenized and truncated at `max_len` tokens. |
| `label` (output) | string | One of `positive`, `negative`, `neutral`. |
| `confidence` | float | Softmax probability of the predicted class. |
| `probabilities` | object | Full distribution over all three classes. |
| `latency_ms` | float | Server-side inference time in milliseconds. |

---

## Operational Expectations

| Concern | Expectation |
|---|---|
| **Latency target** | < 50 ms per tweet on CPU (single-thread, batch size 1) |
| **Batch throughput** | ~200–500 tweets/sec on CPU with batch size 32 |
| **Memory footprint** | < 50 MB for the mini-transformer (2 layers, hidden 128) |
| **Concurrency** | Single Uvicorn worker is sufficient for classroom demo; add workers for shared use |
| **Failure handling** | FastAPI returns HTTP 422 for malformed requests and 503 if the model is not loaded |

---

## Monitoring and Maintenance Checklist

### Pre-launch
- [ ] Confirm `/health` returns 200 after startup
- [ ] Run a smoke test: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "This is a test"}'`
- [ ] Verify `macro_f1 >= 0.65` gate passed in evaluation step
- [ ] Document the checkpoint path and git commit hash used

### During Operation
- [ ] **Request latency** — log `latency_ms` from each response; alert if p95 > 200 ms
- [ ] **Error rate** — track HTTP 4xx/5xx; any non-422 errors warrant investigation
- [ ] **Prediction distribution** — log label distribution over a rolling window; large
      drift from training distribution (e.g., > 60 % neutral) may indicate input drift
- [ ] **Unknown-token rate** — if feasible, log the fraction of `<unk>` tokens per request
      as a signal that the vocabulary is mismatched for incoming text

### Post-demo / Submission Readiness
- [ ] Repository contains clear pipeline documentation (README, this file)
- [ ] Instructor collaborator access confirmed on the repository
- [ ] Reproducible commands present for all stages:
  - Training: `README.md` Colab Smoke Test section
  - Evaluation: `evaluation.md` "Running the Evaluation Script" section
  - Deployment: this file "Running the Server" section
- [ ] `artifacts/` directory (or equivalent) committed or clearly referenced in docs

---

## Interactive API Docs
FastAPI generates automatic documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc:       `http://localhost:8000/redoc`

These are enabled by default and require no additional setup.
