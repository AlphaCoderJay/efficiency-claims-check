"""ECC-001 runner (Efficiency Claims Check, claim C12: nGPT).  Uses the published open harness unchanged (harness/gym_harness.py, a byte copy of
efficiency-claims-check/harness/gym_harness.py at commit f9beaba; sha256 checked at start).  The runner only chooses: the submission, its learning
rate, the seed (data order + init) and which validation windows are scored.

    python run_ecc001.py pilot  --data-dir DIR                 # short code checks before the freeze (reported in PREREG.md)
    python run_ecc001.py tune   --data-dir DIR                 # LR grid, both recipes, seed 11, scored on TUNING windows 256-511
    python run_ecc001.py final  --data-dir DIR                 # chosen LRs, seeds 11 / 22 / 33, scored on the SEALED windows 0-255
    python run_ecc001.py report                                # R_FLOP / R_time per seed, verdict by the frozen pass rule

DIR holds en_train.bin / en_validation.bin (WikiText-2, byte-level BPE vocabulary 2,048, uint16).  Logs go to logs/.  One job at a time, 2 threads.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import sys                                                                       # noqa: E402
sys.path.insert(0, str(HERE / 'harness'))
import gym_harness as G                                                          # noqa: E402

HARNESS_SHA = '8fc0eb875d52e81e0eb5874ac6b7522d3cbfcdc7d1fce86d471c95897bd2a0fc'
SUBS = {'baseline': HERE / 'submissions' / 'baseline.py', 'ngpt': HERE / 'submissions' / 'ngpt.py'}
LR_GRID = {'baseline': [2e-3, 4e-3, 8e-3, 1.6e-2], 'ngpt': [2e-3, 4e-3, 8e-3, 1.6e-2]}
SEEDS = (11, 22, 33)
LOG = HERE / 'logs'
W = G.CTX + 1


def check_harness():
    h = hashlib.sha256((HERE / 'harness' / 'gym_harness.py').read_bytes()).hexdigest()
    assert h == HARNESS_SHA, f'harness copy differs from the published file: {h}'


def data(d):
    d = Path(d)
    return G.load_tokens(d / 'en_train.bin'), G.load_tokens(d / 'en_validation.bin')


def one(name, lr, seed, train, valid, updates=None, threads=2):
    """One harness run of a submission with a given LR and seed (data order + init)."""
    orig_load, orig_starts, orig_seed = G.load_submission, G.batch_starts, G.SEED

    def load(path):
        mod = orig_load(path); mod.CONFIG = dict(mod.CONFIG, lr=lr)
        if updates:
            mod.CONFIG['updates'] = updates; mod.CONFIG['eval_every'] = max(1, updates // 4)
        return mod
    G.load_submission = load
    G.batch_starts = lambda n_tokens, upd, seed_=seed: orig_starts(n_tokens, upd, seed=seed_)
    G.SEED = seed
    try:
        t0 = time.time(); rec = G.run(str(SUBS[name]), train, valid, threads=threads, log=lambda s: None)
        rec.update(name=name, lr=lr, seed=seed, wall_s=round(time.time() - t0, 1))
        return rec
    finally:
        G.load_submission, G.batch_starts, G.SEED = orig_load, orig_starts, orig_seed


def save(path, obj):
    LOG.mkdir(exist_ok=True); path.write_text(json.dumps(obj, indent=1) + '\n', encoding='utf-8')


def pilot(a):
    train, valid = data(a.data_dir); out = []
    for name in SUBS:
        rec = one(name, 4e-3, 11, train, valid, updates=40)
        out.append({k: rec.get(k) for k in ('name', 'status', 'params', 'checks', 'flops_per_update', 'train_cpu_s', 'points', 'wall_s')})
        print(json.dumps(out[-1])[:600], flush=True)
    save(LOG / 'pilot.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), runs=out))


def tune(a):
    train, valid = data(a.data_dir)
    tune_valid = valid[256 * W:]                                                  # tuning windows 256-511; the sealed windows 0-255 are not scored
    assert len(tune_valid) >= 256 * W, 'not enough validation text for disjoint tuning windows'
    p = LOG / 'tune.json'; out = json.loads(p.read_text()) if p.exists() else dict(runs=[])
    done = {(r['name'], r['lr']) for r in out['runs']}
    for name, grid in LR_GRID.items():
        for lr in grid:
            if (name, lr) in done:
                continue
            rec = one(name, lr, 11, train, tune_valid); out['runs'].append(rec); save(p, out)
            print(name, lr, rec['status'], rec['points'][-1]['bits_per_token'] if rec.get('points') else None, rec.get('train_cpu_s'), flush=True)
    best = {}
    for name in LR_GRID:
        rs = [r for r in out['runs'] if r['name'] == name and r['status'] == 'done']
        b = min(rs, key=lambda r: r['points'][-1]['bits_per_token'])
        edge = b['lr'] in (min(LR_GRID[name]), max(LR_GRID[name]))
        best[name] = dict(lr=b['lr'], bits=b['points'][-1]['bits_per_token'], at_grid_edge=edge)
    out['best'] = best; save(p, out); print(json.dumps(best), flush=True)


def final(a):
    train, valid = data(a.data_dir)
    best = json.loads((LOG / 'tune.json').read_text())['best']
    p = LOG / 'final.json'; out = json.loads(p.read_text()) if p.exists() else dict(runs=[])
    done = {(r['name'], r['seed']) for r in out['runs']}
    for i, seed in enumerate(SEEDS):
        order = list(SUBS) if i % 2 == 0 else list(SUBS)[::-1]                     # alternate order between seeds (timing drift)
        for name in order:
            if (name, seed) in done:
                continue
            rec = one(name, best[name]['lr'], seed, train, valid); out['runs'].append(rec); save(p, out)
            print(name, seed, rec['status'], rec['points'][-1]['bits_per_token'] if rec.get('points') else None, rec.get('train_cpu_s'), flush=True)


def report(a):
    out = json.loads((LOG / 'final.json').read_text()); per = {}
    for seed in SEEDS:
        ref = next(r for r in out['runs'] if r['name'] == 'baseline' and r['seed'] == seed)
        sub = next(r for r in out['runs'] if r['name'] == 'ngpt' and r['seed'] == seed)
        per[seed] = dict(grade=G.grade(sub, ref), baseline_final=ref['points'][-1]['bits_per_token'], ngpt_final=sub['points'][-1]['bits_per_token'],
                         flops_per_update=dict(baseline=ref['flops_per_update'], ngpt=sub['flops_per_update']),
                         checks=dict(baseline=ref['checks'], ngpt=sub['checks']))
    rf = sorted(per[s]['grade']['R_flops'] for s in SEEDS); rt = sorted(per[s]['grade']['R_time'] for s in SEEDS)
    bounds = [per[s]['grade']['R_flops_bound'] for s in SEEDS] + [per[s]['grade']['R_time_bound'] for s in SEEDS]
    med_f, med_t = rf[1], rt[1]
    checks_ok = all(c['ok'] for s in SEEDS for side in per[s]['checks'].values() for k, c in side.items() if isinstance(c, dict) and 'ok' in c)
    cls = lambda r: 'HOLDS' if r <= .25 else ('PARTIAL' if r <= .8 else 'DOES NOT HOLD')         # frozen classes on R_FLOP
    if not checks_ok:
        verdict, reason = 'INCONCLUSIVE', 'a validity check failed'
    elif cls(rf[0]) != cls(rf[2]):
        verdict, reason = 'INCONCLUSIVE', f'the three seeds fall in different classes ({cls(rf[0])} .. {cls(rf[2])})'
    elif cls(med_f) == 'HOLDS' and med_t > .35:
        verdict, reason = 'PARTIAL', 'FLOP ratio in the claimed range but time ratio above 0.35'
    else:
        verdict, reason = cls(med_f), 'median R_FLOP class, all seeds agree'
    rep = dict(per_seed=per, R_flops_sorted=rf, R_time_sorted=rt, bounds=bounds, median=dict(R_flops=med_f, R_time=med_t), verdict=verdict, reason=reason)
    save(LOG / 'report.json', rep); print(json.dumps(dict(R_flops=rf, R_time=rt, verdict=verdict), indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('which', choices=['pilot', 'tune', 'final', 'report']); p.add_argument('--data-dir')
    a = p.parse_args(); check_harness(); torch.set_num_threads(2)
    dict(pilot=pilot, tune=tune, final=final, report=report)[a.which](a)
