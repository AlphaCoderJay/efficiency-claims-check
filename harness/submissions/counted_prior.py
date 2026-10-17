"""Reference recipe + a COUNTED bigram prior: prepare() counts next-token frequencies of every token in the training stream (numpy, outside any
FLOP counter) and the model adds a learned multiple of log p_count(next | current) to its logits. A published lever (n-gram / neural
interpolation); legal because it uses only training tokens, but its table build must be charged. Self-contained submission."""
import numpy as np
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
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class PriorGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, logprior, ctx=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('logprior', logprior)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x) + self.pos[: x.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.ln(h) @ self.emb.weight.T + self.alpha * self.logprior[x]


def prepare(train_tokens):
    t = train_tokens.numpy()
    v = 2048
    counts = np.bincount(t[:-1] * v + t[1:], minlength=v * v).reshape(v, v).astype(np.float64) + 0.05
    return torch.from_numpy(np.log(counts / counts.sum(1, keepdims=True)).astype(np.float32))


def build_model(vocab, aux=None):
    c = CONFIG
    return PriorGPT(vocab, c['d'], c['layers'], c['heads'], aux)


def make_optimizer(model, cfg):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.startswith('emb')]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and not n.startswith('emb'))]
    return torch.optim.AdamW([dict(params=decay, weight_decay=cfg['wd']), dict(params=rest, weight_decay=0.0)], lr=cfg['lr'], betas=(0.9, 0.95))
