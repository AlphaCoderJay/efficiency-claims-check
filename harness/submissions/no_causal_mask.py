"""TRAP: the reference recipe with bidirectional attention (is_causal=False). Each position can see the token it must predict, so training loss
collapses; the causality check must reject it before any training."""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(updates=1200, eval_every=100, lr=4e-3, d=96, layers=3, heads=4, wd=0.1)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x) + self.pos[: x.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.ln(h) @ self.emb.weight.T


def build_model(vocab, aux=None):
    c = CONFIG
    return GPT(vocab, c['d'], c['layers'], c['heads'])


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.95), weight_decay=cfg['wd'])
