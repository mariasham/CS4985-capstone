import csv
import random

import torch
from torch.utils.data import Dataset


LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def load_tweet_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text:
                continue
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unsupported label {label!r}. Expected one of {LABELS}.")
            rows.append({"text": text, "label": label})

    if not rows:
        raise ValueError(f"No usable rows found in {path}.")
    return rows


def split_rows(rows, val_ratio=0.2, seed=42):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * val_ratio))
    return rows[val_size:], rows[:val_size]


class TweetSentimentDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        input_ids, attention_mask = self.tokenizer.encode(row["text"], self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "labels": torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long),
        }
