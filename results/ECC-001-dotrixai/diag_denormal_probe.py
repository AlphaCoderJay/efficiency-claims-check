"""ECC-001 diagnostic (not part of the frozen protocol; no decision uses it): is the per-update CPU cost of the tuning runs driven by subnormal
floats?  Times blocks of updates with the harness's own loop logic, counts subnormal values in parameters / gradients / optimiser state, and repeats
a block with torch.set_flush_denormal(True).  Writes logs/diag_denormal.json (internal; not a result file).

    python ecc001_denormal_probe.py
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

E = Path(__file__).resolve().parent; sys.path.insert(0, str(E / 'harness'))
import gym_harness as G                                                          # noqa: E402

TINY = torch.finfo(torch.float32).tiny


def subnormals(model, opt):
    n = dict(params=0, grads=0, state=0)
    for p in model.parameters():
        n['params'] += int(((p.detach().abs() < TINY) & (p.detach() != 0)).sum())
        if p.grad is not None:
            n['grads'] += int(((p.grad.abs() < TINY) & (p.grad != 0)).sum())
    for o in (opt.optimizers if hasattr(opt, 'optimizers') else [opt]):
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 1:
                    n['state'] += int(((v.abs() < TINY) & (v != 0)).sum())
    return n


def session(name, lr, upto, block, train):
    torch.set_num_threads(2); torch.set_flush_denormal(False)
    mod = G.load_submission(str(E / 'submissions' / f'{name}.py')); mod.CONFIG['lr'] = lr
    cfg = dict(updates=1200, eval_every=100); cfg.update(mod.CONFIG)
    torch.manual_seed(11); model = mod.build_model(G.VOCAB, None); opt = mod.make_optimizer(model, cfg)
    lr_at = getattr(mod, 'lr_at', G.default_lr)
    opts = opt.optimizers if hasattr(opt, 'optimizers') else [opt]; base = [[g['lr'] for g in o.param_groups] for o in opts]
    starts = G.batch_starts(len(train), cfg['updates'], seed=11)
    out = dict(name=name, lr=lr, blocks=[])

    def step(u):
        x = torch.stack([train[s:s + G.CTX] for s in starts[u]]); y = torch.stack([train[s + 1:s + G.CTX + 1] for s in starts[u]])
        m = lr_at(u, cfg)
        for o, bl in zip(opts, base):
            for g, b0 in zip(o.param_groups, bl):
                g['lr'] = b0 * m
        loss = F.cross_entropy(model(x).float().reshape(-1, G.VOCAB), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    u = 0
    while u < upto:
        c = time.process_time()
        for _ in range(block):
            step(u); u += 1
        out['blocks'].append(dict(to=u, cpu_s=round(time.process_time() - c, 2), subnormal=subnormals(model, opt), flush=False))
        print(name, out['blocks'][-1], flush=True)
    for flush in (True, False):
        torch.set_flush_denormal(flush); c = time.process_time()
        for _ in range(block):
            step(u); u += 1
        out['blocks'].append(dict(to=u, cpu_s=round(time.process_time() - c, 2), subnormal=subnormals(model, opt), flush=flush))
        print(name, out['blocks'][-1], flush=True)
    torch.set_flush_denormal(False)
    return out


def main():
    train = G.load_tokens('data/wikitext2_train.bin')
    res = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), note='diagnostic only; not part of the frozen ECC-001 protocol',
               runs=[session('ngpt', 0.016, 40, 20, train), session('baseline', 0.016, 950, 50, train)])
    (E / 'logs' / 'diag_denormal.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
