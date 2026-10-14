"""A tiny decoder-only Transformer, written for reading, not for speed.

MECHANICS DEMONSTRATION ONLY. This is not competitive pretraining; it exists so
you can see every tensor that a real language model also computes.

Shape legend used in comments:
    B = batch size, T = sequence length (time), C = embedding width,
    H = number of heads, D = C // H = head width, V = vocabulary size
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    """Character-level tokenizer: every distinct character is one token."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, C: int, H: int, T_max: int, dropout: float):
        super().__init__()
        assert C % H == 0
        self.H, self.D = H, C // H
        self.qkv = nn.Linear(C, 3 * C)    # one matmul produces queries, keys and values
        self.proj = nn.Linear(C, C)       # mixes the heads back together
        self.drop = nn.Dropout(dropout)
        # mask[i, j] = 1 if position i may look at position j (j <= i)
        self.register_buffer("mask", torch.tril(torch.ones(T_max, T_max)).view(1, 1, T_max, T_max))
        self.last_attn = None             # kept for inspection in the lesson

    def forward(self, x):                                       # x: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                   # each (B, T, C)
        q = q.view(B, T, self.H, self.D).transpose(1, 2)        # (B, H, T, D)
        k = k.view(B, T, self.H, self.D).transpose(1, 2)
        v = v.view(B, T, self.H, self.D).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.D)    # (B, H, T, T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # hide the future
        attn = F.softmax(scores, dim=-1)                        # each row sums to 1
        self.last_attn = attn.detach()
        y = self.drop(attn) @ v                                 # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)        # concatenate heads -> (B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """Pre-norm Transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, C: int, H: int, T_max: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = CausalSelfAttention(C, H, T_max, dropout)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection: the block adds a correction
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, V: int, T_max: int = 128, C: int = 128, H: int = 4, L: int = 4, dropout: float = 0.1):
        super().__init__()
        self.T_max = T_max
        self.tok_emb = nn.Embedding(V, C)       # token id -> vector
        self.pos_emb = nn.Embedding(T_max, C)   # position -> vector (learned)
        self.blocks = nn.ModuleList([Block(C, H, T_max, dropout) for _ in range(L)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V, bias=False)  # vector -> one score (logit) per vocab entry
        self.lm_head.weight = self.tok_emb.weight   # weight tying, as in many real LMs
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Small initial weights (GPT-2 style, std 0.02).

        PyTorch's default nn.Embedding init is N(0, 1). Because the output head is
        tied to the embedding, that gives initial logits with std ~ sqrt(C) and a
        starting loss far above ln(V) (measured: 82.5 instead of 3.7 before this fix).
        A healthy untrained model starts near ln(V), i.e. uniform guessing.
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):           # idx: (B, T) integer token ids
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)    # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))           # (B, T, V)
        loss = None
        if targets is not None:
            # Cross-entropy averaged over all B*T positions:
            #   loss = -(1/BT) * sum_{b,t} log softmax(logits[b,t])[targets[b,t]]
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.T_max:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)   # only the last position matters
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def get_batch(data: torch.Tensor, B: int, T: int, gen: torch.Generator):
    """Sample B windows of length T+1; inputs are the first T, targets the last T.

    This is the 'shifted labels' idea: the target at position t is the token at t+1.
    """
    starts = torch.randint(0, len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[s: s + T] for s in starts])
    y = torch.stack([data[s + 1: s + T + 1] for s in starts])
    return x, y
