import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ID_TO_LABEL, LABELS, TweetSentimentDataset, load_tweet_rows, split_rows
from .model import MiniDecoderSentimentClassifier
from .tokenizer import BasicTweetTokenizer


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_f1(preds, labels, num_labels):
    scores = []
    for label_id in range(num_labels):
        tp = sum(1 for pred, label in zip(preds, labels) if pred == label_id and label == label_id)
        fp = sum(1 for pred, label in zip(preds, labels) if pred == label_id and label != label_id)
        fn = sum(1 for pred, label in zip(preds, labels) if pred != label_id and label == label_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    labels = []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, batch_labels)
        total_loss += loss.item() * input_ids.size(0)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch_labels.cpu().tolist())

    accuracy = sum(1 for pred, label in zip(preds, labels) if pred == label) / len(labels)
    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1(preds, labels, len(LABELS)),
    }


def train(args):
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_tweet_rows(args.data)
    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    tokenizer = BasicTweetTokenizer().fit([row["text"] for row in train_rows], min_freq=args.min_freq)

    train_dataset = TweetSentimentDataset(train_rows, tokenizer, args.max_len)
    val_dataset = TweetSentimentDataset(val_rows, tokenizer, args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = MiniDecoderSentimentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_labels=len(LABELS),
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        train_loss = total_loss / len(train_dataset)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, indent=2))

    tokenizer_path = output_dir / "tokenizer.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.json"
    tokenizer.save(tokenizer_path)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config,
            "labels": ID_TO_LABEL,
            "args": vars(args),
        },
        checkpoint_path,
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved tokenizer: {tokenizer_path}")
    print(f"Saved metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a mini transformer for tweet sentiment classification.")
    parser.add_argument("--data", default="examples/sample_tweet_sentiment.csv")
    parser.add_argument("--output-dir", default="artifacts/donovan-smoke-test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
