"""
evaluate.py — Evaluation script for the tweet sentiment classifier.

Usage:
    python -m llm_pipeline.evaluate --data ../examples/sample_tweet_sentiment.csv --checkpoint ../artifacts/tweet-sentiment-model/best_checkpoint.pt --tokenizer ../artifacts/tweet-sentiment-model/tokenizer.json --output-dir ../artifacts/evaluation
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import LABELS, LABEL_TO_ID, load_tweet_rows
from .model import MiniDecoderSentimentClassifier
from .tokenizer import BasicTweetTokenizer


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def confusion_matrix(preds, labels, num_labels):
    """Returns matrix[true][pred] counts."""
    matrix = [[0] * num_labels for _ in range(num_labels)]
    for pred, label in zip(preds, labels):
        matrix[label][pred] += 1
    return matrix


def per_class_metrics(matrix, num_labels):
    """Precision, recall, F1 per class from confusion matrix."""
    results = {}
    for i in range(num_labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(num_labels)) - tp
        fn = sum(matrix[i][j] for j in range(num_labels)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results[LABELS[i]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(matrix[i]),
        }
    return results


def macro_f1_from_per_class(per_class):
    return round(sum(v["f1"] for v in per_class.values()) / len(per_class), 4)


def macro_precision_from_per_class(per_class):
    return round(sum(v["precision"] for v in per_class.values()) / len(per_class), 4)


def macro_recall_from_per_class(per_class):
    return round(sum(v["recall"] for v in per_class.values()) / len(per_class), 4)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, tokenizer, rows, max_len, device, batch_size=32):
    """Returns list of (true_label_id, pred_label_id, confidence, text)."""
    model.eval()
    results = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        ids_list, mask_list, true_labels = [], [], []
        for row in batch_rows:
            ids, mask = tokenizer.encode(row["text"], max_len)
            ids_list.append(ids)
            mask_list.append(mask)
            true_labels.append(LABEL_TO_ID[row["label"]])

        input_ids = torch.tensor(ids_list, dtype=torch.long, device=device)
        attention_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().tolist()
        confs = probs.max(dim=-1).values.cpu().tolist()

        for row, true_label, pred, conf in zip(batch_rows, true_labels, preds, confs):
            results.append({
                "text": row["text"],
                "true_label": LABELS[true_label],
                "pred_label": LABELS[pred],
                "correct": true_label == pred,
                "confidence": round(conf, 4),
            })
    return results


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def collect_errors(results, max_per_category=5):
    """Group errors by (true_label, pred_label) and return representative examples."""
    buckets = {}
    for r in results:
        if not r["correct"]:
            key = (r["true_label"], r["pred_label"])
            buckets.setdefault(key, []).append(r)

    error_analysis = []
    for (true_label, pred_label), cases in sorted(buckets.items()):
        # Sort by confidence descending (model was most wrong when confident)
        cases_sorted = sorted(cases, key=lambda x: x["confidence"], reverse=True)
        error_analysis.append({
            "true_label": true_label,
            "predicted_as": pred_label,
            "count": len(cases),
            "high_confidence_examples": [
                {"text": c["text"], "confidence": c["confidence"]}
                for c in cases_sorted[:max_per_category]
            ],
        })
    return sorted(error_analysis, key=lambda x: x["count"], reverse=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load tokenizer and model
    tokenizer = BasicTweetTokenizer.load(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MiniDecoderSentimentClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    max_len = checkpoint["model_config"]["max_len"]
    print(f"Loaded checkpoint from epoch {checkpoint['args'].get('epochs', '?')}")

    # Load evaluation data
    rows = load_tweet_rows(args.data)
    print(f"Evaluating on {len(rows)} examples from: {args.data}")

    # Run inference
    results = run_inference(model, tokenizer, rows, max_len, device, batch_size=args.batch_size)

    # Aggregate metrics
    preds     = [LABEL_TO_ID[r["pred_label"]] for r in results]
    true_lbls = [LABEL_TO_ID[r["true_label"]] for r in results]

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    cm = confusion_matrix(preds, true_lbls, len(LABELS))
    per_class = per_class_metrics(cm, len(LABELS))
    macro_f1  = macro_f1_from_per_class(per_class)

    # Class distribution
    class_dist = {label: sum(1 for r in results if r["true_label"] == label) for label in LABELS}

    # Low-confidence examples (model uncertain)
    low_conf = sorted(
        [r for r in results if r["confidence"] < args.low_conf_threshold],
        key=lambda x: x["confidence"],
    )[:10]

    # Error analysis
    error_analysis = collect_errors(results)

    report = {
        "dataset": args.data,
        "checkpoint": args.checkpoint,
        "num_examples": len(results),
        "class_distribution": class_dist,
        "summary": {
            "accuracy": round(accuracy, 4),
            "macro_f1": macro_f1,
            "macro_precision": macro_precision_from_per_class(per_class),
            "macro_recall": macro_recall_from_per_class(per_class),
        },
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm,
            "description": "confusion_matrix[true_label][pred_label]",
        },
        "error_analysis": error_analysis,
        "low_confidence_examples": low_conf,
    }

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    predictions_path = output_dir / "predictions.jsonl"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(predictions_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary to console
    print("\n=== Evaluation Summary ===")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  Macro F1:         {macro_f1:.4f}")
    print(f"  Macro Precision:  {macro_precision_from_per_class(per_class):.4f}")
    print(f"  Macro Recall:     {macro_recall_from_per_class(per_class):.4f}")
    print("\n=== Per-Class Metrics ===")
    for label, metrics in per_class.items():
        print(f"  {label:10s}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  support={metrics['support']}")
    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    header = "         " + "  ".join(f"{l[:8]:>8}" for l in LABELS)
    print(header)
    for i, row_counts in enumerate(cm):
        row_str = f"{LABELS[i]:>8}   " + "  ".join(f"{c:>8}" for c in row_counts)
        print(row_str)
    print(f"\nSaved report:       {report_path}")
    print(f"Saved predictions:  {predictions_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained tweet sentiment classifier.")
    parser.add_argument("--data",         required=True,  help="Path to labeled CSV (text + label/sentiment columns).")
    parser.add_argument("--checkpoint",   default="artifacts/tweet-sentiment-model/best_checkpoint.pt")
    parser.add_argument("--tokenizer",    default="artifacts/tweet-sentiment-model/tokenizer.json")
    parser.add_argument("--output-dir",   default="artifacts/evaluation")
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--low-conf-threshold", type=float, default=0.6,
                        help="Confidence below this threshold is flagged as low-confidence.")
    parser.add_argument("--device",       default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
