import argparse
import json

import torch
import torch.nn.functional as F

from .model import MiniDecoderSentimentClassifier
from .tokenizer import BasicTweetTokenizer


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MiniDecoderSentimentClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_text(model, tokenizer, text, labels, max_len, device):
    input_ids, attention_mask = tokenizer.encode(text, max_len)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.bool, device=device)
    probabilities = F.softmax(model(input_ids, attention_mask), dim=-1)[0]
    label_id = int(probabilities.argmax().item())
    return {
        "text": text,
        "label": labels[str(label_id)] if str(label_id) in labels else labels[label_id],
        "confidence": float(probabilities[label_id].item()),
        "probabilities": {
            labels[str(idx)] if str(idx) in labels else labels[idx]: float(prob.item())
            for idx, prob in enumerate(probabilities)
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run sentiment prediction from a trained checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/donovan-smoke-test/checkpoint.pt")
    parser.add_argument("--tokenizer", default="artifacts/donovan-smoke-test/tokenizer.json")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer = BasicTweetTokenizer.load(args.tokenizer)
    model, checkpoint = load_model(args.checkpoint, device)
    result = predict_text(
        model=model,
        tokenizer=tokenizer,
        text=args.text,
        labels=checkpoint["labels"],
        max_len=checkpoint["model_config"]["max_len"],
        device=device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(parse_args())
