"""Demo of the training-efficiency gym prototype: train the reference recipe, grade six submissions against it in the same session, grade the
two single-lever submissions again against a STRONGER reference, rerun the reference at the end (run-to-run noise), and write demo_results.json
and DEMO_REPORT.md next to this script.

    python run_demo.py --data-dir DIR      (DIR holds train.bin / valid.bin or en_train.bin / en_validation.bin: uint16 token ids, vocabulary 2,048)
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch

import gym_harness as G

HERE = Path(__file__).resolve().parent
ORDER = ['ngram_input', 'counted_prior', 'strong_reference', 'no_causal_mask', 'batch_coupled', 'reads_files']


def find(d, names):
    for n in names:
        if (d / n).exists():
            return d / n
    raise FileNotFoundError(names)


def main(a):
    d = Path(a.data_dir)
    train = G.load_tokens(find(d, ['train.bin', 'en_train.bin']))
    valid = G.load_tokens(find(d, ['valid.bin', 'en_validation.bin']))
    out = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), machine=platform.processor() or platform.machine(),
               torch=torch.__version__, threads=a.threads, train_tokens=len(train), eval_tokens=G.EVAL_WINDOWS * G.CTX, runs={})
    print('reference')
    ref = G.run(HERE / 'submissions' / 'reference.py', train, valid, a.threads)
    out['runs']['reference'] = dict(record=ref)
    for name in ORDER:
        print(name)
        rec = G.run(HERE / 'submissions' / f'{name}.py', train, valid, a.threads)
        out['runs'][name] = dict(record=rec, grade=G.grade(rec, ref))
        print('  ', json.dumps({k: v for k, v in out['runs'][name]['grade'].items() if k != 'R_time_curve'}))
    strong = out['runs']['strong_reference']['record']
    for name in ('ngram_input', 'counted_prior'):
        out['runs'][name]['grade_vs_strong_reference'] = G.grade(out['runs'][name]['record'], strong)
    print('reference (repeat)')
    rep = G.run(HERE / 'submissions' / 'reference.py', train, valid, a.threads)
    out['runs']['reference_repeat'] = dict(record=rep, grade=G.grade(rep, ref))
    out['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (HERE / 'demo_results.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    report(out)


def fmt_r(g, k):
    if k not in g:
        return '-'
    b = g.get(f'{k}_bound')
    return f'{g[k]:.3f}' + (f' ({b} bound)' if b else '')


def report(out):
    ref = out['runs']['reference']['record']
    strong = out['runs']['strong_reference']['record']
    L = ['# Demo report: training-efficiency gym, CPU prototype', '',
         f'Run {out["started_utc"]} to {out["finished_utc"]} on {out["machine"]}, PyTorch {out["torch"]}, {out["threads"]} threads. '
         f'Training stream: {out["train_tokens"]:,} tokens (WikiText-2 train, BPE 2,048); sealed evaluation: {out["eval_tokens"]:,} tokens of '
         'WikiText-2 validation.', '',
         f'Reference recipe: {ref["params"]:,} parameters, {ref["config"]["updates"]} updates of {G.BATCH} x {G.CTX} tokens, final '
         f'{ref["points"][-1]["bits_per_token"]:.4f} bits per token (= Q), {ref["total_cpu_s"]:.1f} CPU-s, '
         f'{ref["flops_per_update"] * ref["config"]["updates"] / 1e9:,.0f} GFLOP.', '',
         '## Grades against the reference', '',
         '| submission | status | final bits per token | R_time at Q | R_FLOP at Q | prepare share of CPU | reward |', '|---|---|---|---|---|---|---|']
    for name, r in out['runs'].items():
        if name == 'reference':
            continue
        rec, g = r['record'], r['grade']
        fin = f'{rec["points"][-1]["bits_per_token"]:.4f}' if rec.get('status') == 'done' else '-'
        ps = f'{g["prepare_share_of_cpu"]:.4f}' if 'prepare_share_of_cpu' in g else '-'
        L.append(f'| {name} | {rec.get("status")} | {fin} | {fmt_r(g, "R_time")} | {fmt_r(g, "R_flops")} | {ps} | {g["reward"]:.3f} |')
    L += ['', '## Cost to quality: R_time at every quality the reference passes', '',
          'Each column is the reference\'s quality after that many updates; a value is the submission\'s CPU time to reach it divided by the '
          'reference\'s. "upper" = already better at the submission\'s first evaluation point (the true ratio is lower).', '']
    pts = [p['ref_update'] for p in out['runs']['ngram_input']['grade']['R_time_curve']]
    L.append('| submission | ' + ' | '.join(f'@{u}' for u in pts) + ' |'); L.append('|---|' + '---|' * len(pts))
    for name in ('ngram_input', 'counted_prior', 'strong_reference'):
        c = out['runs'][name]['grade']['R_time_curve']
        L.append(f'| {name} | ' + ' | '.join(f'{x["R_time"]:.2f}' + (' u' if x['bound'] == 'upper' else (' l' if x['bound'] == 'lower' else ''))
                                            for x in c) + ' |')
    L += ['', '(u = upper bound, l = lower bound: never reached within the budget.)', '',
          f'## The same two levers against a stronger reference (both levers together, final {strong["points"][-1]["bits_per_token"]:.4f} bits per token)', '',
          '| submission | R_time at the strong Q | reward |', '|---|---|---|']
    for name in ('ngram_input', 'counted_prior'):
        g = out['runs'][name]['grade_vs_strong_reference']
        L.append(f'| {name} | {fmt_r(g, "R_time")} | {g["reward"]:.3f} |')
    rep = out['runs']['reference_repeat']
    rt = rep['grade']['R_time']
    noise_reward = abs(math.log2(rt)) / 2
    L += ['', '## Noise', '',
          f'The reference rerun at the end gives identical bits (deterministic data order and seed: {rep["record"]["points"][-1]["bits_per_token"]:.4f}) '
          f'but R_time {rt:.3f}: the CPU time of two identical runs differs by {abs(rt - 1) * 100:.0f}% on this laptop, the same as a reward of '
          f'{noise_reward:.2f}. Rewards smaller than that are noise here; the FLOP ratio is exact.', '',
          '## Rejected submissions', '']
    for name, r in out['runs'].items():
        rec = r['record']
        if rec.get('status', '').startswith('rejected'):
            L.append(f'- **{name}**: {rec["status"]}; checks ' + json.dumps(rec['checks']))
    (HERE / 'DEMO_REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--report-only', action='store_true', help='rebuild DEMO_REPORT.md from demo_results.json')
    a = p.parse_args()
    if a.report_only:
        report(json.loads((HERE / 'demo_results.json').read_text(encoding='utf-8')))
    else:
        assert a.data_dir, '--data-dir is required'
        main(a)
