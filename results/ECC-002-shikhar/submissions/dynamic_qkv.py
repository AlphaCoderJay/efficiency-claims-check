"""Reference PyTorch implementation of low-rank causal dynamic short convolutions on Q, K and V."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(updates=1200, eval_every=100, lr=4e-3, d=96, layers=3, heads=4, wd=0.1, rank=7, width=4)


class LowRankDynamicShortConvolution(nn.Module):
    def __init__(self, d, rank, width):
        super().__init__()
        self.width = width
        self.proj_in = nn.Linear(d, rank, bias=False)
        self.proj_out = nn.Parameter(torch.zeros(rank, width * d))
        self.proj_out_bias = nn.Parameter(torch.empty(width, d))
        nn.init.uniform_(self.proj_out_bias, -1.0 / math.sqrt(width), 1.0 / math.sqrt(width))

    def forward(self, x, pre_z):
        batch, time, d = x.shape
        weights = (self.proj_in(pre_z) @ self.proj_out).view(batch, time, self.width, d)
        weights = weights + self.proj_out_bias
        out = torch.zeros_like(x)
        for offset in range(self.width):
            shifted = F.pad(x, (0, 0, offset, 0))[:, :time]
            out = out + weights[:, :, offset] * shifted
        return out + x


class Block(nn.Module):
    def __init__(self, d, heads, rank, width):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv_q = LowRankDynamicShortConvolution(d, rank, width)
        self.conv_k = LowRankDynamicShortConvolution(d, rank, width)
        self.conv_v = LowRankDynamicShortConvolution(d, rank, width)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        batch, time, d = x.shape
        pre_z = self.ln1(x)
        q, k, v = self.qkv(pre_z).chunk(3, dim=-1)
        q = self.conv_q(q, pre_z)
        k = self.conv_k(k, pre_z)
        v = self.conv_v(v, pre_z)
        head_dim = d // self.heads
        q = q.view(batch, time, self.heads, head_dim).transpose(1, 2)
        k = k.view(batch, time, self.heads, head_dim).transpose(1, 2)
        v = v.view(batch, time, self.heads, head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(batch, time, d))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, rank, width, ctx=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        self.blocks = nn.ModuleList(Block(d, heads, rank, width) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x) + self.pos[: x.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.ln(h) @ self.emb.weight.T


def build_model(vocab, aux=None):
    c = CONFIG
    return GPT(vocab, c['d'], c['layers'], c['heads'], c['rank'], c['width'])


def make_optimizer(model, cfg):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.startswith('emb')]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and not n.startswith('emb'))]
    return torch.optim.AdamW(
        [dict(params=decay, weight_decay=cfg['wd']), dict(params=rest, weight_decay=0.0)],
        lr=cfg['lr'],
        betas=(0.9, 0.95),
    )
