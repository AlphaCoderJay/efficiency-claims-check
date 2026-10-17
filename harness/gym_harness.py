"""Training-efficiency gym, CPU prototype (v0.1).

A submission is one Python file that defines a language model and how to train it. The harness owns everything that decides the score: the data
order, the batch shape, the sealed evaluation text, the timing and the checks. Reward = how much less compute the submission needs than the
reference recipe to reach the reference's final quality.

Submission interface (all names are module attributes):
    CONFIG                       dict, optional keys: updates (<= MAX_UPDATES), eval_every, lr, ... (passed back to the functions below)
    prepare(train_tokens)        optional; any preprocessing on the TRAINING tokens (a 1-D LongTensor); returns an object passed to build_model.
                                 Its CPU time is charged like training time.
    build_model(vocab, aux)      returns a torch.nn.Module; forward(x: LongTensor[B, T]) -> logits [B, T, vocab]
    make_optimizer(model, cfg)   returns an object with step() and zero_grad()
    lr_at(update, cfg)           optional; multiplier on every param group's base lr (default: 20-update warmup then cosine to 10%)

Checks (any failure -> reward 0):
    static    the source may import only ALLOWED_IMPORTS and may not use file, process or reflection built-ins
    budget    updates <= MAX_UPDATES, parameters <= MAX_PARAMS
    causal    perturb one token in row 0 of a 2-row batch: logits of row 0 before that position and every logit of row 1 must not change
              (fp32, tolerance 1e-5); run on the initial AND the trained model (catches lookahead and batch coupling)
    finite    loss finite throughout
Accounting: C_time = process CPU seconds of prepare + every training update (the harness's own batch sampling excluded);
C_flops = FLOPs of one forward + backward counted by torch.utils.flop_counter times the number of updates (table builds in numpy or Python are
NOT counted there, so they show up only in C_time). Evaluation is done by the harness outside the timed region.
Ratio R(Q) = C_submission(Q) / C_reference(Q) with Q = the reference's final bits per token, C_submission interpolated linearly in log C between
its evaluation points (lower / upper bound flags as in the CIR analyses). Reward = clip(log2(1 / R_time), 0, 2) / 2.
"""
import ast
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

VOCAB = 2048
CTX, BATCH = 128, 16
MAX_UPDATES, MAX_PARAMS = 2000, 16_000_000
EVAL_WINDOWS = 256
SEED = 11
ALLOWED_IMPORTS = {'torch', 'torch.nn', 'torch.nn.functional', 'math', 'numpy', 'collections', 'itertools', 'functools', 'dataclasses'}
FORBIDDEN_NAMES = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', 'breakpoint',
                   'input', 'memoryview', 'load', 'fromfile', 'save', 'tofile'}


def static_scan(path):
    tree = ast.parse(open(path, encoding='utf-8').read())
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    problems.append(f'import {a.name}')
        elif isinstance(node, ast.ImportFrom):
            if (node.module or '') not in ALLOWED_IMPORTS:
                problems.append(f'from {node.module} import ...')
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f'name {node.id}')
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            problems.append(f'attribute .{node.attr}')
    return dict(ok=not problems, problems=sorted(set(problems)))


def load_submission(path):
    spec = importlib.util.spec_from_file_location('submission', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tokens(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def batch_starts(n_tokens, updates, seed=SEED):
    """The data order is the harness's: the same windows for every submission."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tokens - CTX - 1, size=(updates, BATCH))


def causality_test(model, tol=1e-5):
    model.eval()
    g = torch.Generator().manual_seed(1234)
    x = torch.randint(0, VOCAB, (2, CTX), generator=g)
    y = x.clone(); t = CTX // 2; y[0, t] = (y[0, t] + 1) % VOCAB
    with torch.no_grad():
        a, b = model(x).float(), model(y).float()
    prefix = float((a[0, :t] - b[0, :t]).abs().max())
    other = float((a[1] - b[1]).abs().max())
    model.train()
    return dict(ok=prefix <= tol and other <= tol, max_change_before_t=prefix, max_change_other_row=other)


def flops_per_update(model, x, y):
    with FlopCounterMode(display=False) as fc:
        loss = F.cross_entropy(model(x).float().reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
    model.zero_grad(set_to_none=True)
    return int(fc.get_total_flops())


def evaluate(model, valid):
    """Bits per token on the first EVAL_WINDOWS non-overlapping windows of the sealed text (every target scored)."""
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, EVAL_WINDOWS, 32):
            idx = torch.arange(i, min(i + 32, EVAL_WINDOWS)) * (CTX + 1)
            w = torch.stack([valid[s:s + CTX + 1] for s in idx.tolist()])
            logits = model(w[:, :-1]).float()
            tot += float(F.cross_entropy(logits.reshape(-1, VOCAB), w[:, 1:].reshape(-1), reduction='sum')); n += w[:, 1:].numel()
    model.train()
    return tot / n / math.log(2)


def default_lr(u, cfg):
    warm = 20
    if u < warm:
        return (u + 1) / warm
    p = (u - warm) / max(1, cfg['updates'] - warm)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def run(path, train, valid, threads=2, log=print):
    """Train and evaluate one submission; returns the raw record (no reward)."""
    torch.set_num_threads(threads)
    rec = dict(submission=Path(path).name, checks={})
    rec['checks']['static'] = static_scan(path)
    if not rec['checks']['static']['ok']:
        rec['status'] = 'rejected: static scan'; return rec
    mod = load_submission(path)
    cfg = dict(updates=1200, eval_every=100); cfg.update(getattr(mod, 'CONFIG', {}))
    if cfg['updates'] > MAX_UPDATES:
        rec['status'] = f'rejected: {cfg["updates"]} updates > {MAX_UPDATES}'; return rec
    torch.manual_seed(SEED)
    c0 = time.process_time()
    aux = mod.prepare(train) if hasattr(mod, 'prepare') else None
    prep_cpu = time.process_time() - c0
    model = mod.build_model(VOCAB, aux)
    n_params = sum(p.numel() for p in model.parameters())
    rec.update(params=n_params, prepare_cpu_s=round(prep_cpu, 3), config=cfg)
    if n_params > MAX_PARAMS:
        rec['status'] = f'rejected: {n_params} parameters > {MAX_PARAMS}'; return rec
    rec['checks']['causal_init'] = causality_test(model)
    if not rec['checks']['causal_init']['ok']:
        rec['status'] = 'rejected: causality (initial model)'; return rec
    opt = mod.make_optimizer(model, cfg)
    lr_at = getattr(mod, 'lr_at', default_lr)
    base = [[g['lr'] for g in o.param_groups] for o in (opt.optimizers if hasattr(opt, 'optimizers') else [opt])]
    starts = batch_starts(len(train), cfg['updates'])
    x0 = torch.stack([train[s:s + CTX] for s in starts[0]]); y0 = torch.stack([train[s + 1:s + CTX + 1] for s in starts[0]])
    fpu = flops_per_update(model, x0, y0)
    points, cpu, snaps = [], prep_cpu, {}
    for u in range(cfg['updates']):
        x = torch.stack([train[s:s + CTX] for s in starts[u]]); y = torch.stack([train[s + 1:s + CTX + 1] for s in starts[u]])
        c = time.process_time()
        m = lr_at(u, cfg)
        for o, bl in zip(opt.optimizers if hasattr(opt, 'optimizers') else [opt], base):
            for g, b0 in zip(o.param_groups, bl):
                g['lr'] = b0 * m
        loss = F.cross_entropy(model(x).float().reshape(-1, VOCAB), y.reshape(-1))
        if not torch.isfinite(loss):
            rec['status'] = f'rejected: non-finite loss at update {u}'; return rec
        opt.zero_grad(); loss.backward(); opt.step()
        cpu += time.process_time() - c
        if (u + 1) % cfg['eval_every'] == 0 or u + 1 == cfg['updates']:
            snaps[u + 1] = (cpu, {k: v.detach().clone() for k, v in model.state_dict().items()})
            log(f'  update {u + 1}: train loss {float(loss):.3f}, cpu {cpu:.1f} s')
    rec['checks']['causal_trained'] = causality_test(model)
    if not rec['checks']['causal_trained']['ok']:
        rec['status'] = 'rejected: causality (trained model)'; return rec
    for u, (c, sd) in snaps.items():
        model.load_state_dict(sd)
        points.append(dict(update=u, cpu_s=round(c, 3), flops=fpu * u, bits_per_token=round(evaluate(model, valid), 5)))
    rec.update(status='done', flops_per_update=fpu, points=points, train_cpu_s=round(cpu - prep_cpu, 3), total_cpu_s=round(cpu, 3))
    return rec


def ratio(points, q, c_ref, key):
    """C_sub(q) / c_ref along the submission's points; (value, bound) with bound None / 'lower' / 'upper'."""
    pts = sorted(((p[key], p['bits_per_token']) for p in points), key=lambda t: t[0])
    if pts[-1][1] > q:
        return pts[-1][0] / c_ref, 'lower'
    if pts[0][1] <= q:
        return pts[0][0] / c_ref, 'upper'
    for (c0, b0), (c1, b1) in zip(pts, pts[1:]):
        if b0 > q >= b1:
            f = (b0 - q) / (b0 - b1)
            return math.exp(math.log(c0) + f * (math.log(c1) - math.log(c0))) / c_ref, None
    return pts[-1][0] / c_ref, 'lower'


def grade(rec, ref):
    """Reward against the reference record (same session)."""
    if rec.get('status') != 'done':
        return dict(reward=0.0, reason=rec.get('status'))
    q = ref['points'][-1]['bits_per_token']
    rt, bt = ratio(rec['points'], q, ref['points'][-1]['cpu_s'], 'cpu_s')
    rf, bf = ratio(rec['points'], q, ref['points'][-1]['flops'], 'flops')
    reward = 0.0 if bt == 'lower' else min(1.0, max(0.0, math.log2(1 / rt) / 2))
    curve = []                                                                # cost-to-quality curve: R at every reference checkpoint quality
    for p in ref['points']:
        r, b = ratio(rec['points'], p['bits_per_token'], p['cpu_s'], 'cpu_s')
        curve.append(dict(ref_update=p['update'], Q=p['bits_per_token'], R_time=round(r, 4), bound=b))
    return dict(reward=round(reward, 4), Q=q, R_time=round(rt, 4), R_time_bound=bt, R_flops=round(rf, 4), R_flops_bound=bf,
                prepare_share_of_cpu=round(rec['prepare_cpu_s'] / rec['total_cpu_s'], 4),
                final_bits_per_token=rec['points'][-1]['bits_per_token'], R_time_curve=curve)
