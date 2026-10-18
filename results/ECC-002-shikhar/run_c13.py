"""Pilot, tune, and evaluate the C13 dynamic-short-convolution check."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
import gym_harness as G

SUBS = {
    'baseline': ROOT / 'harness' / 'submissions' / 'reference.py',
    'dynamic': HERE / 'submissions' / 'dynamic_qkv.py',
}
LR_GRID = (2e-3, 4e-3, 8e-3, 1.6e-2)
SEEDS = (11, 22, 33)
LOG = HERE / 'logs'
WINDOW = G.CTX + 1
LAYERS, WIDTH, D_MODEL = 3, 4, 96


def data(data_dir):
    d = Path(data_dir)
    return G.load_tokens(d / 'en_train.bin'), G.load_tokens(d / 'en_validation.bin')


def extra_pointwise_flops():
    batch, seq, width, d = G.BATCH, G.CTX, WIDTH, D_MODEL
    # FlopCounterMode counts the low-rank projections but omits the elementwise
    # dynamic convolution and its backward pass.
    forward = (3 * width + 1) * batch * seq * d
    backward_products = 2 * width * batch * seq * d
    backward_input_sum = batch * d * (width * seq - width * (width - 1) // 2)
    backward_bias_sum = width * d * (batch * seq - 1)
    return LAYERS * 3 * (forward + backward_products + backward_input_sum + backward_bias_sum)


def one(name, lr, seed, train, valid, updates=None, threads=2):
    orig_load = G.load_submission
    orig_starts = G.batch_starts
    orig_flops = G.flops_per_update
    orig_seed = G.SEED

    def load(path):
        mod = orig_load(path)
        mod.CONFIG = dict(mod.CONFIG, lr=lr)
        if updates is not None:
            mod.CONFIG.update(updates=updates, eval_every=max(1, updates // 4))
        return mod

    correction = extra_pointwise_flops() if name == 'dynamic' else 0

    def flops_per_update(model, x, y):
        return orig_flops(model, x, y) + correction

    G.load_submission = load
    G.batch_starts = lambda n_tokens, n_updates, seed_=seed: orig_starts(n_tokens, n_updates, seed=seed_)
    G.flops_per_update = flops_per_update
    G.SEED = seed
    try:
        started = time.time()
        rec = G.run(str(SUBS[name]), train, valid, threads=threads, log=lambda _: None)
        rec.update(name=name, lr=lr, seed=seed, wall_s=round(time.time() - started, 1))
        rec['extra_pointwise_flops_per_update'] = correction
        rec['flops_per_update_torch'] = rec['flops_per_update'] - correction
        return rec
    finally:
        G.load_submission = orig_load
        G.batch_starts = orig_starts
        G.flops_per_update = orig_flops
        G.SEED = orig_seed


def save(path, value):
    LOG.mkdir(exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')


def tuning_data(valid):
    start = 256 * WINDOW
    end = 512 * WINDOW
    if len(valid) < end:
        raise ValueError(f'need at least {end} validation tokens for disjoint tuning windows')
    return valid[start:end]


def pilot(args):
    train, valid = data(args.data_dir)
    valid = tuning_data(valid)
    runs = []
    for name in SUBS:
        rec = one(name, 4e-3, 11, train, valid, updates=40, threads=args.threads)
        runs.append({k: rec.get(k) for k in (
            'name', 'status', 'params', 'checks', 'flops_per_update',
            'flops_per_update_torch', 'extra_pointwise_flops_per_update', 'points', 'wall_s',
        )})
        print(json.dumps(runs[-1]), flush=True)
    save(LOG / 'pilot.json', dict(runs=runs))


def tune(args):
    train, valid = data(args.data_dir)
    valid = tuning_data(valid)
    path = LOG / 'tune.json'
    out = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'runs': []}
    done = {(r['name'], r['lr']) for r in out['runs']}
    for name in SUBS:
        for lr in LR_GRID:
            if (name, lr) in done:
                continue
            rec = one(name, lr, 11, train, valid, threads=args.threads)
            out['runs'].append(rec)
            save(path, out)
            final = rec['points'][-1]['bits_per_token'] if rec.get('points') else None
            print(name, lr, rec['status'], final, flush=True)
    out['best'] = {}
    for name in SUBS:
        completed = [r for r in out['runs'] if r['name'] == name and r['status'] == 'done']
        if not completed:
            raise RuntimeError(f'no successful tuning runs for {name}')
        best = min(completed, key=lambda r: r['points'][-1]['bits_per_token'])
        out['best'][name] = dict(lr=best['lr'], bits=best['points'][-1]['bits_per_token'])
    save(path, out)
    print(json.dumps(out['best'], indent=1), flush=True)


def final(args):
    train, valid = data(args.data_dir)
    best = json.loads((LOG / 'tune.json').read_text(encoding='utf-8'))['best']
    path = LOG / 'final.json'
    out = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'runs': []}
    done = {(r['name'], r['seed']) for r in out['runs']}
    sealed = valid[:256 * WINDOW]
    for i, seed in enumerate(SEEDS):
        names = list(SUBS)
        if i % 2:
            names.reverse()
        for name in names:
            if (name, seed) in done:
                continue
            rec = one(name, best[name]['lr'], seed, train, sealed, threads=args.threads)
            out['runs'].append(rec)
            save(path, out)
            quality = rec['points'][-1]['bits_per_token'] if rec.get('points') else None
            print(name, seed, rec['status'], quality, rec.get('train_cpu_s'), flush=True)


def report(args):
    runs = json.loads((LOG / 'final.json').read_text(encoding='utf-8'))['runs']
    per_seed = {}
    for seed in SEEDS:
        ref = next(r for r in runs if r['name'] == 'baseline' and r['seed'] == seed)
        method = next(r for r in runs if r['name'] == 'dynamic' and r['seed'] == seed)
        if ref.get('status') != 'done' or method.get('status') != 'done':
            raise RuntimeError(f'incomplete final run for seed {seed}')
        per_seed[seed] = G.grade(method, ref)
    ratios = sorted(per_seed[seed]['R_flops'] for seed in SEEDS)
    bounds = [per_seed[seed]['R_flops_bound'] for seed in SEEDS]
    median = ratios[1]
    passed_checks = all(
        check.get('ok', True)
        for seed in SEEDS
        for run in runs if run['seed'] == seed
        for check in run.get('checks', {}).values()
        if isinstance(check, dict)
    )
    if not passed_checks or any(bound is not None for bound in bounds):
        verdict = 'INCONCLUSIVE'
    elif median <= 1 / 1.33 and all(ratio < 1 for ratio in ratios):
        verdict = 'HOLDS AT TESTED SCALE'
    else:
        verdict = 'DOES NOT HOLD AT TESTED SCALE'
    out = dict(
        per_seed=per_seed,
        R_flops_sorted=ratios,
        R_flops_bounds=bounds,
        median_R_flops=median,
        pass_threshold=1 / 1.33,
        verdict=verdict,
    )
    save(LOG / 'report.json', out)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pilot', 'tune', 'final', 'report'))
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dict(pilot=pilot, tune=tune, final=final, report=report)[args.phase](args)
