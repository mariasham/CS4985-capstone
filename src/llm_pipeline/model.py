import math

import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        batch_size, seq_len, hidden_size = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        key_mask = attention_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out(context)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask):
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class MiniDecoderSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        max_len=64,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        ff_size=256,
        dropout=0.1,
        pad_id=0,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "num_labels": num_labels,
            "max_len": max_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_size": ff_size,
            "dropout": dropout,
            "pad_id": pad_id,
        }
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, ff_size, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm(x)
        last_indices = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        pooled = x[torch.arange(batch_size, device=input_ids.device), last_indices]
        return self.classifier(pooled)
