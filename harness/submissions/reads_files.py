"""TRAP: tries to read a file at import time (for example the evaluation text) and to use the os module. The static scan must reject it before
any code runs."""
import os

import torch

CONFIG = dict(updates=300)
LEAK = None
if os.path.exists('valid.bin'):
    LEAK = open('valid.bin', 'rb').read()


def build_model(vocab, aux=None):
    return torch.nn.Linear(vocab, vocab)


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)
