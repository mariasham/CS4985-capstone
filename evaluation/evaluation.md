# Evaluation Plan

## Objective
Evaluate model quality on the held-out test split with task metrics, robustness checks,
and qualitative failure analysis. Outputs feed directly into deployment decisions and
API documentation.

---

## Metrics

### Core Task Metrics
| Metric | Why |
|---|---|
| **Accuracy** | Overall fraction of correct predictions across all three classes. |
| **Macro F1** | Unweighted average F1 across classes — primary metric because the dataset may be class-imbalanced and all three classes matter equally. |
| **Per-class Precision / Recall / F1** | Reveals which class the model struggles with most. |
| **Confusion Matrix** | Full picture of misclassification patterns (e.g., negative predicted as neutral). |

### Success Thresholds (to be updated after first full-data run)
Set conservatively for a capstone-scale model and dataset:

| Metric | Target |
|---|---|
| Validation Accuracy | ≥ 0.70 |
| Macro F1 | ≥ 0.65 |

If either threshold is missed, re-examine class balance and consider label smoothing or
weighted loss before concluding the model is insufficient.

---

## Robustness Checks

### Class Imbalance
- Check `class_distribution` in the evaluation report. If any class has fewer than 15 %
  of examples, per-class recall may be unreliable.
- Mitigation: use `torch.nn.CrossEntropyLoss(weight=...)` with inverse-frequency weights.

### Edge Cases
- Very short inputs (1–2 tokens): ensure the model does not default to a single class.
- Tweets with only hashtags, URLs, or mentions: tokenizer should handle these gracefully
  via the `TOKEN_PATTERN` in `tokenizer.py`.
- Mixed-case and emoji-heavy text: lowercase normalization is applied; emojis become
  unknown tokens. Flag if unknown-token rate exceeds 30 % of a tweet's length.

---

## Running the Evaluation Script

```bash
# Evaluate on the held-out test CSV
PYTHONPATH=src python -m llm_pipeline.evaluate \
    --data data/test_tweet_sentiment.csv \
    --checkpoint artifacts/tweet-sentiment-model/best_checkpoint.pt \
    --tokenizer  artifacts/tweet-sentiment-model/tokenizer.json \
    --output-dir artifacts/evaluation \
    --low-conf-threshold 0.6
```

### Outputs
| File | Contents |
|---|---|
| `artifacts/evaluation/evaluation_report.json` | Summary metrics, per-class metrics, confusion matrix, error analysis, low-confidence examples |
| `artifacts/evaluation/predictions.jsonl` | One prediction per line with text, true label, predicted label, and confidence |

---

## Error Analysis
The script groups misclassified examples by (true\_label → predicted\_label) category and
surfaces the highest-confidence errors — i.e., cases where the model was confidently wrong.

Actionable follow-up for each error category:

| Error Pattern | Likely Cause | Mitigation |
|---|---|---|
| negative → neutral | Sarcasm or mild negative language | Add more diverse negative examples |
| neutral → positive/negative | Ambiguous tweets that lean slightly one way | Tighten label guidelines or add a "mixed" bucket |
| positive → neutral | Understated positivity (no exclamation, emojis) | Data augmentation or weighted sampling |

---

## Handoff Contract to `deployment/`
Provide the following before the deployment stage:

1. **Best checkpoint path** — `artifacts/tweet-sentiment-model/best_checkpoint.pt`
2. **Latency baseline** — record `latency_ms` from the `/predict` endpoint on the target
   machine. Target: < 50 ms per tweet on CPU.
3. **Known limitations** — document in the API description:
   - Model is word-level; OOV words become `<unk>` tokens.
   - Max input length is 64 tokens; longer tweets are silently truncated.
   - Trained on [dataset name]; out-of-domain tweets may degrade accuracy.
4. **Minimum accuracy gate** — if `macro_f1 < 0.65`, do not ship the checkpoint.
