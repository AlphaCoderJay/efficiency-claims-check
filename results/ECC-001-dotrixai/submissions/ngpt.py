"""ECC-001 method: normalized Transformer (nGPT; Loshchilov, Hsieh, Sun, Ginsburg, arXiv 2410.01131), implemented from the paper's equations at
the same shapes as the baseline (d 96, 3 layers, 4 heads, SwiGLU hidden 256, RoPE, untied embeddings):
  - every hidden state lives on the unit sphere: h <- Norm(h + alpha_A (Norm(h_A) - h)), h <- Norm(h + alpha_M (Norm(h_M) - h));
  - attention on normalised q, k scaled by s_qk, softmax scale sqrt(d_k) instead of 1 / sqrt(d_k);
  - SwiGLU with u * s_u and nu * s_nu * sqrt(d);
  - logits z = s_z * (E_out h);
  - no LayerNorm / RMSNorm, no weight decay, no warmup;
  - after every optimiser step all matrices are renormalised along the embedding dimension (the 'step' of the optimiser wrapper);
  - scaling vectors stored as (init / scale) restoration: effective = param * (init / scale), param initialised to scale (paper's trick).
The learning rate is set by the runner (tuning grid in PREREG.md)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(updates=1200, eval_every=50, lr=4e-3, d=96, layers=3, heads=4, hidden=256)
CTX = 128


def norm(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-6)


def rope(x, base=10000.0):
    T, dh = x.shape[-2], x.shape[-1]
    half = dh // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Scale(nn.Module):
    """Learnable per-dimension scale with the paper's restoration: effective = param * (init / scale)."""
    def __init__(self, n, init, scale):
        super().__init__()
        self.ratio = init / scale
        self.param = nn.Parameter(torch.full((n,), float(scale)))

    def forward(self):
        return self.param * self.ratio


class Block(nn.Module):
    def __init__(self, d, h, hidden):
        super().__init__()
        self.h, self.dh, self.d = h, d // h, d
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.u, self.g = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.alpha_a = Scale(d, 0.05, 1 / math.sqrt(d))
        self.alpha_m = Scale(d, 0.05, 1 / math.sqrt(d))
        self.s_qk = Scale(self.dh, 1.0, 1 / math.sqrt(d))
        self.s_u = Scale(hidden, 1.0, 1.0)
        self.s_nu = Scale(hidden, 1.0, 1.0)

    def forward(self, h):
        B, T, D = h.shape
        split = lambda z: z.view(B, T, self.h, self.dh).transpose(1, 2)
        q = norm(rope(split(self.q(h)))) * self.s_qk()
        k = norm(rope(split(self.k(h)))) * self.s_qk()
        y = F.scaled_dot_product_attention(q, k, split(self.v(h)), is_causal=True, scale=math.sqrt(self.dh))
        h_a = norm(self.o(y.transpose(1, 2).reshape(B, T, D)))
        h = norm(h + self.alpha_a() * (h_a - h))
        u = self.u(h) * self.s_u()
        nu = self.g(h) * self.s_nu() * math.sqrt(self.d)
        h_m = norm(self.down(u * F.silu(nu)))
        return norm(h + self.alpha_m() * (h_m - h))


class NGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.blocks = nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.s_z = Scale(vocab, 1.0, 1 / math.sqrt(d))
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        self.renormalize()

    @torch.no_grad()
    def renormalize(self):
        """Unit norm along the embedding dimension for every matrix (rows for d -> *, columns for * -> d)."""
        self.emb.weight.copy_(norm(self.emb.weight, 1))
        self.out.weight.copy_(norm(self.out.weight, 1))
        for b in self.blocks:
            for lin in (b.q, b.k, b.v, b.u, b.g):
                lin.weight.copy_(norm(lin.weight, 1))                         # weight [out, d]: each row lives in R^d
            for lin in (b.o, b.down):
                lin.weight.copy_(norm(lin.weight, 0))                         # weight [d, in]: each column lives in R^d

    def forward(self, x):
        h = norm(self.emb(x))
        for b in self.blocks:
            h = b(h)
        return self.out(h) * self.s_z()


class RenormAdamW:
    """AdamW without weight decay; after every step the matrices are renormalised (nGPT).  Exposes `optimizers` for the harness's LR schedule."""
    def __init__(self, model, lr):
        self.model = model
        self.optimizers = [torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)]

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step(self):
        for o in self.optimizers:
            o.step()
        self.model.renormalize()


def build_model(vocab, aux=None):
    c = CONFIG
    return NGPT(vocab, c['d'], c['layers'], c['heads'], c['hidden'])


def make_optimizer(model, cfg):
    return RenormAdamW(model, cfg['lr'])


def lr_at(u, cfg):
    """No warmup (paper); cosine to 10% (same floor as the baseline)."""
    p = u / max(1, cfg['updates'])
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
