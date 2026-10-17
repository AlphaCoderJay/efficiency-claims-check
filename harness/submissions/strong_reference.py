"""A STRONGER reference: the reference recipe plus both cheap published levers of this demo, hashed bigram / trigram input embeddings (SparseAdam)
and a counted bigram prior on the logits. Used to show how much of a submission's reward survives a stronger baseline (Lane E in CIR terms).
Self-contained submission."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(updates=1200, eval_every=100, lr=4e-3, d=96, layers=3, heads=4, wd=0.1)
SLOTS = 1 << 15
PAD = 2048


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


class StrongGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, logprior, ctx=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        self.t2 = nn.Embedding(SLOTS, d, sparse=True)
        self.t3 = nn.Embedding(SLOTS, d, sparse=True)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('logprior', logprior)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.zeros_(self.t2.weight)
        nn.init.zeros_(self.t3.weight)

    def forward(self, x):
        p1 = F.pad(x, (1, 0), value=PAD)[:, :-1]
        p2 = F.pad(x, (2, 0), value=PAD)[:, :-2]
        h2 = (p1 * 2654435761 + x * 40503) % SLOTS
        h3 = (p2 * 2654435761 + p1 * 2246822519 + x * 3266489917) % SLOTS
        h = self.emb(x) + self.pos[: x.shape[1]] + self.t2(h2) + self.t3(h3)
        for b in self.blocks:
            h = b(h)
        return self.ln(h) @ self.emb.weight.T + self.alpha * self.logprior[x]


class Pair:
    def __init__(self, *optimizers):
        self.optimizers = list(optimizers)

    def step(self):
        for o in self.optimizers:
            o.step()

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad(set_to_none=True)


def prepare(train_tokens):
    t = train_tokens.numpy()
    v = 2048
    counts = np.bincount(t[:-1] * v + t[1:], minlength=v * v).reshape(v, v).astype(np.float64) + 0.05
    return torch.from_numpy(np.log(counts / counts.sum(1, keepdims=True)).astype(np.float32))


def build_model(vocab, aux=None):
    c = CONFIG
    return StrongGPT(vocab, c['d'], c['layers'], c['heads'], aux)


def make_optimizer(model, cfg):
    tables = {id(model.t2.weight), id(model.t3.weight)}
    named = [(n, p) for n, p in model.named_parameters() if id(p) not in tables]
    decay = [p for n, p in named if p.dim() >= 2 and not n.startswith('emb')]
    rest = [p for n, p in named if not (p.dim() >= 2 and not n.startswith('emb'))]
    dense = torch.optim.AdamW([dict(params=decay, weight_decay=cfg['wd']), dict(params=rest, weight_decay=0.0)], lr=cfg['lr'], betas=(0.9, 0.95))
    sparse = torch.optim.SparseAdam([model.t2.weight, model.t3.weight], lr=cfg['lr'], betas=(0.9, 0.95))
    return Pair(dense, sparse)
