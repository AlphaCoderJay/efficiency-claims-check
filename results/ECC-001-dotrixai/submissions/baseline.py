"""ECC-001 baseline: a small modern GPT (pre-LayerNorm, RoPE, SwiGLU MLP, untied input / output embeddings), AdamW with weight decay,
harness default schedule (20-update warmup, cosine to 10%).  Same shapes as the nGPT submission: d 96, 3 layers, 4 heads, MLP hidden 256.
The learning rate is set by the runner (tuning grid in PREREG.md)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(updates=1200, eval_every=50, lr=4e-3, d=96, layers=3, heads=4, hidden=256, wd=0.1)
CTX = 128


def rope(x, base=10000.0):
    """x [B, H, T, Dh] -> rotary position embedding on pairs (first half, second half)."""
    T, dh = x.shape[-2], x.shape[-1]
    half = dh // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d, h, hidden):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.u, self.g = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        a = self.ln1(x)
        split = lambda z: z.view(B, T, self.h, D // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(rope(split(self.q(a))), rope(split(self.k(a))), split(self.v(a)), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        m = self.ln2(x)
        return x + self.down(self.u(m) * F.silu(self.g(m)))


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.blocks = nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.out(self.ln(h))


def build_model(vocab, aux=None):
    c = CONFIG
    return GPT(vocab, c['d'], c['layers'], c['heads'], c['hidden'])


def make_optimizer(model, cfg):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.startswith('emb')]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and not n.startswith('emb'))]
    return torch.optim.AdamW([dict(params=decay, weight_decay=cfg['wd']), dict(params=rest, weight_decay=0.0)], lr=cfg['lr'], betas=(0.9, 0.95))
