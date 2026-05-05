import json
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"https?://\S+|@\w+|#\w+|[A-Za-z0-9']+|[^\s]")


class BasicTweetTokenizer:
    """Small word-level tokenizer for capstone-scale tweet experiments."""

    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, token_to_id=None, lowercase=True):
        self.lowercase = lowercase
        self.token_to_id = token_to_id or {
            self.pad_token: 0,
            self.unk_token: 1,
        }
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        return TOKEN_PATTERN.findall(text)

    def fit(self, texts, min_freq=1, max_vocab_size=20000):
        counts = Counter()
        for text in texts:
            counts.update(self.tokenize(text))

        for token, freq in counts.most_common(max_vocab_size - len(self.token_to_id)):
            if freq >= min_freq and token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        return self

    def encode(self, text, max_len):
        ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in self.tokenize(text)]
        ids = ids[:max_len]
        attention_mask = [1] * len(ids)

        while len(ids) < max_len:
            ids.append(self.pad_id)
            attention_mask.append(0)

        return ids, attention_mask

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "lowercase": self.lowercase,
                    "token_to_id": self.token_to_id,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(token_to_id=payload["token_to_id"], lowercase=payload.get("lowercase", True))
