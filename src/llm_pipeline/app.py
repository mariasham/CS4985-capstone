"""
app.py — FastAPI inference server for the tweet sentiment classifier.

Usage:
    pip install fastapi uvicorn
    python -m llm_pipeline.app --checkpoint ../artifacts/tweet-sentiment-model/best_checkpoint.pt --tokenizer ../artifacts/tweet-sentiment-model/tokenizer.json --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              — liveness check
    GET  /model-info          — model config and loaded checkpoint path
    POST /predict             — single tweet prediction
    POST /predict-batch       — batch predictions (up to 64 tweets)
"""

import argparse
import time
from contextlib import asynccontextmanager
from typing import List

import torch
import torch.nn.functional as F

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    raise ImportError(
        "FastAPI and uvicorn are required to run the inference server.\n"
        "Install them with:  pip install fastapi uvicorn"
    ) from e

from .model import MiniDecoderSentimentClassifier
from .tokenizer import BasicTweetTokenizer


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

class ModelState:
    model: MiniDecoderSentimentClassifier = None
    tokenizer: BasicTweetTokenizer = None
    labels: dict = None
    max_len: int = 64
    device: torch.device = None
    checkpoint_path: str = None
    loaded_at: float = None


state = ModelState()


def load_model(checkpoint_path: str, tokenizer_path: str, device_str: str = "auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    tokenizer = BasicTweetTokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MiniDecoderSentimentClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    state.model = model
    state.tokenizer = tokenizer
    state.labels = checkpoint["labels"]
    state.max_len = checkpoint["model_config"]["max_len"]
    state.device = device
    state.checkpoint_path = checkpoint_path
    state.loaded_at = time.time()

    print(f"Model loaded on {device} | vocab={tokenizer.vocab_size} | max_len={state.max_len}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Tweet text to classify.")

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=64, description="List of tweet texts (max 64).")

class Prediction(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict

class PredictResponse(BaseModel):
    prediction: Prediction
    latency_ms: float

class PredictBatchResponse(BaseModel):
    predictions: List[Prediction]
    latency_ms: float
    count: int


# ---------------------------------------------------------------------------
# Core inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer(texts: List[str]) -> List[Prediction]:
    ids_list, mask_list = [], []
    for text in texts:
        ids, mask = state.tokenizer.encode(text, state.max_len)
        ids_list.append(ids)
        mask_list.append(mask)

    input_ids = torch.tensor(ids_list, dtype=torch.long, device=state.device)
    attention_mask = torch.tensor(mask_list, dtype=torch.bool, device=state.device)
    logits = state.model(input_ids, attention_mask)
    probs = F.softmax(logits, dim=-1)

    results = []
    for text, prob_row in zip(texts, probs):
        label_id = int(prob_row.argmax().item())
        label_str = state.labels.get(str(label_id), state.labels.get(label_id, str(label_id)))
        prob_dict = {
            state.labels.get(str(i), state.labels.get(i, str(i))): round(float(p), 4)
            for i, p in enumerate(prob_row)
        }
        results.append(Prediction(
            text=text,
            label=label_str,
            confidence=round(float(prob_row[label_id].item()), 4),
            probabilities=prob_dict,
        ))
    return results


# ---------------------------------------------------------------------------
# App factory (supports lifespan for clean startup/shutdown logging)
# ---------------------------------------------------------------------------

def create_app(checkpoint_path: str, tokenizer_path: str, device_str: str = "auto") -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_model(checkpoint_path, tokenizer_path, device_str)
        yield
        print("Shutting down inference server.")

    app = FastAPI(
        title="Tweet Sentiment API",
        description="Mini-transformer sentiment classifier: positive / negative / neutral.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/health", tags=["ops"])
    def health():
        """Liveness check — returns 200 if the server is up and the model is loaded."""
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")
        return {"status": "ok", "device": str(state.device)}

    @app.get("/model-info", tags=["ops"])
    def model_info():
        """Returns model configuration and runtime metadata."""
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")
        return {
            "config": state.model.config,
            "vocab_size": state.tokenizer.vocab_size,
            "labels": state.labels,
            "max_len": state.max_len,
            "checkpoint": state.checkpoint_path,
            "device": str(state.device),
            "loaded_at": state.loaded_at,
        }

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(req: PredictRequest):
        """
        Classify a single tweet.

        Returns the predicted sentiment label, confidence score, and
        full probability distribution over all three classes.
        """
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        t0 = time.perf_counter()
        prediction = _infer([req.text])[0]
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return PredictResponse(prediction=prediction, latency_ms=latency_ms)

    @app.post("/predict-batch", response_model=PredictBatchResponse, tags=["inference"])
    def predict_batch(req: PredictBatchRequest):
        """
        Classify a batch of tweets (up to 64).

        Processes all texts in a single forward pass for efficiency.
        """
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        t0 = time.perf_counter()
        predictions = _infer(req.texts)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return PredictBatchResponse(predictions=predictions, latency_ms=latency_ms, count=len(predictions))

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run the tweet sentiment inference server.")
    parser.add_argument("--checkpoint", default="artifacts/tweet-sentiment-model/best_checkpoint.pt")
    parser.add_argument("--tokenizer",  default="artifacts/tweet-sentiment-model/tokenizer.json")
    parser.add_argument("--host",       default="127.0.0.1")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--device",     default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device_str=args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port)
