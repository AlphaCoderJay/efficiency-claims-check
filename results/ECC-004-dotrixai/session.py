"""ECC-004 session runner: one Kaggle GPU session (T4 x2) runs a fixed plan of ecc004.py runs, one queue per GPU, then (final sessions) the
timing block. The parent process never touches a GPU; every run is its own process.

    python session.py tune  --ctx 1024 --data-dir DIR --out DIR                              # the frozen LR grid at 1/4 budget, tuning seed
    python session.py extra --ctx 1024 --runs baseline:5e-4,ngpt:6.4e-2 [--eager-guard baseline:2e-3] --data-dir DIR --out DIR
    python session.py final --ctx 1024 --seed 11 --lr-baseline X --lr-ngpt Y --data-dir DIR --out DIR

All settings that the PREREG freezes are in this file (COMMON, MICRO, BUDGET, GRID, FRACTIONS, TUNE_SEED, TUNE_FRAC).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMMON = dict(d=384, layers=8, heads=6, hidden=1536, vocab=8192, tokens_per_update=131072, amp='fp16', compile='inductor', qk_norm=1)
UPDATES = 3842                                         # 20 tokens per parameter for about 25.2M parameters, at 2^17 tokens per update
BUDGET = UPDATES * COMMON['tokens_per_update']         # 503,578,624 training tokens per full run
MICRO = {('baseline', 1024): 32, ('ngpt', 1024): 16, ('baseline', 4096): 8, ('ngpt', 4096): 8}   # fastest that fit, calibration 2
GRID = {1024: {'baseline': [1e-3, 2e-3, 4e-3, 8e-3], 'ngpt': [4e-3, 8e-3, 1.6e-2, 3.2e-2]}}      # 4096: 3 points around the 1k best
FRACTIONS = {1024: [1, 1 / 2, 1 / 4, 1 / 8], 4096: [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]}               # nGPT budgets for the budget estimator
TUNE_SEED, TUNE_FRAC = 7, 1 / 4
DEADLINE_H = 11.0                                      # Kaggle sessions end at 12 h; runs still going at 11 h stop with an evaluation

if os.environ.get('ECC004_SMOKE'):                     # laptop test of the session plumbing only: tiny model, CPU, one thread per process
    COMMON = dict(d=64, layers=2, heads=2, hidden=256, vocab=8192, tokens_per_update=2048, amp='fp32', compile='none', qk_norm=1, eval_tokens=4096)
    UPDATES, MICRO = 8, {k: 2 for k in MICRO}
    BUDGET = UPDATES * COMMON['tokens_per_update']
    os.environ['OMP_NUM_THREADS'] = '1'


def run_args(arm, ctx, seed, lr, frac, split, compile_=None, tag=''):
    tokens = int(round(BUDGET * frac / COMMON['tokens_per_update'])) * COMMON['tokens_per_update']
    name = f'{arm}_c{ctx}_s{seed}_lr{lr:g}_f{frac:g}{tag}'
    a = dict(COMMON, arm=arm, ctx=ctx, seed=seed, lr=lr, tokens=tokens, micro_rows=MICRO[(arm, ctx)], eval_split=split)
    if compile_:
        a['compile'] = compile_
    return name, a


def est_hours(arm, ctx, frac):
    per = {('baseline', 1024): 1.55, ('ngpt', 1024): 1.83, ('baseline', 4096): 2.80, ('ngpt', 4096): 2.89}[(arm, ctx)]   # s per update, calib 2 + QK-norm pilot
    return UPDATES * frac * per / 3600 * 1.15                                                                             # + parallel slowdown


def balance(runs):
    """Two queues with about equal estimated hours (longest first)."""
    q = [[], []]; h = [0.0, 0.0]
    for r in sorted(runs, key=lambda r: -r[2]):
        i = 0 if h[0] <= h[1] else 1; q[i].append(r[:2]); h[i] += r[2]
    return q, h


def worker(queue_file, out, t_end):
    for name, a in json.loads(Path(queue_file).read_text()):
        o = Path(out) / name
        if (o / 'run.jsonl').exists() and '"status":"done"' in (o / 'run.jsonl').read_text().splitlines()[-1]:
            continue                                                                    # already done (a resumed session)
        left_min = (t_end - time.time()) / 60
        if left_min < 5:
            print(f'{name}: not started, session deadline', flush=True); continue
        cmd = [sys.executable, str(HERE / 'ecc004.py'), 'train'] + sum((['--' + k.replace('_', '-'), str(v)] for k, v in a.items()), [])
        cmd += ['--data-dir', DATA, '--out', str(o), '--deadline-min', f'{left_min - 3:.1f}']
        if (o / 'ckpt.pt').exists():
            cmd += ['--resume', str(o / 'ckpt.pt')]
        o.mkdir(parents=True, exist_ok=True)
        with open(o / 'stdout.txt', 'a', encoding='utf-8') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        last = (o / 'run.jsonl').read_text().splitlines()[-1] if (o / 'run.jsonl').exists() else ''
        if rc == 0 and '"status":"done"' in last:
            (o / 'ckpt.pt').unlink(missing_ok=True)                                     # keep checkpoints only for unfinished runs
        print(f'{name}: rc {rc}', flush=True)


def launch(queues, out, gpus):
    out.mkdir(parents=True, exist_ok=True); t_end = T0 + DEADLINE_H * 3600; procs = []
    for i, q in enumerate(queues):
        if not q:
            continue
        qf = out / f'queue_gpu{gpus[i]}.json'; qf.write_text(json.dumps(q, indent=1), encoding='utf-8', newline='\n')
        cmd = [sys.executable, __file__, 'worker', '--queue', str(qf), '--out', str(out), '--t-end', str(t_end), '--data-dir', DATA]
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[i]))))
    return [p.wait() for p in procs]


def main():
    global DATA, T0
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['tune', 'extra', 'final', 'worker', 'timing'])
    ap.add_argument('--ctx', type=int, default=1024); ap.add_argument('--seed', type=int); ap.add_argument('--lr-baseline', type=float)
    ap.add_argument('--lr-ngpt', type=float); ap.add_argument('--runs', default=''); ap.add_argument('--eager-guard', default='')
    ap.add_argument('--grid', default='', help='4096 tuning grid, e.g. baseline:1e-3,2e-3,4e-3;ngpt:8e-3,1.6e-2,3.2e-2')
    ap.add_argument('--data-dir', required=True); ap.add_argument('--out', required=True); ap.add_argument('--queue'); ap.add_argument('--t-end', type=float)
    ap.add_argument('--swap', action='store_true', help='swap the GPU of each queue (alternate between seeds)')
    a = ap.parse_args(); DATA = a.data_dir; T0 = time.time(); out = Path(a.out)
    if a.cmd == 'worker':
        return worker(a.queue, a.out, a.t_end)
    if a.cmd == 'timing':
        import ecc004 as E
        cfg = {arm: E.Config(**dict(run_args(arm, a.ctx, a.seed, lr, 1, 'tune')[1])) for arm, lr in (('baseline', a.lr_baseline), ('ngpt', a.lr_ngpt))}
        return E.timing_block(cfg, DATA, out / 'timing')
    gpus = [1, 0] if a.swap else [0, 1]
    plan = dict(cmd=a.cmd, ctx=a.ctx, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), common=COMMON, updates=UPDATES, budget_tokens=BUDGET)
    if a.cmd == 'tune':
        grid = GRID.get(a.ctx) or {k: [float(x) for x in v.split(',')] for k, v in (p.split(':') for p in a.grid.split(';'))}
        runs = [(*run_args(arm, a.ctx, TUNE_SEED, lr, TUNE_FRAC, 'tune'), est_hours(arm, a.ctx, TUNE_FRAC)) for arm, lrs in grid.items() for lr in lrs]
    elif a.cmd == 'extra':
        runs = [(*run_args(arm, a.ctx, TUNE_SEED, float(lr), TUNE_FRAC, 'tune'), est_hours(arm, a.ctx, TUNE_FRAC))
                for arm, lr in (p.split(':') for p in a.runs.split(',') if p)]
        runs += [(*run_args(arm, a.ctx, TUNE_SEED, float(lr), TUNE_FRAC, 'tune', compile_='none', tag='_eager'), est_hours(arm, a.ctx, TUNE_FRAC) * 1.6)
                 for arm, lr in (p.split(':') for p in a.eager_guard.split(',') if p)]
    else:
        runs = [(*run_args('baseline', a.ctx, a.seed, a.lr_baseline, 1, 'sealed'), est_hours('baseline', a.ctx, 1))]
        runs += [(*run_args('ngpt', a.ctx, a.seed, a.lr_ngpt, f, 'sealed'), est_hours('ngpt', a.ctx, f)) for f in FRACTIONS[a.ctx]]
    queues, hours = balance(runs)
    plan.update(queues={f'gpu{gpus[i]}': q for i, q in enumerate(queues)}, est_hours=hours)
    out.mkdir(parents=True, exist_ok=True); (out / 'plan.json').write_text(json.dumps(plan, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps(dict(est_hours=[round(h, 2) for h in hours], queues={k: [n for n, _ in v] for k, v in plan['queues'].items()}), indent=1), flush=True)
    rcs = launch(queues, out, gpus)
    if a.cmd == 'final':
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[0]))
        subprocess.run([sys.executable, __file__, 'timing', '--ctx', str(a.ctx), '--seed', str(a.seed), '--lr-baseline', str(a.lr_baseline),
                        '--lr-ngpt', str(a.lr_ngpt), '--data-dir', DATA, '--out', str(out)], env=env)
    plan.update(worker_returncodes=rcs, seconds=round(time.time() - T0, 1))
    (out / 'plan.json').write_text(json.dumps(plan, indent=1), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
