"""ECC-004 verify: recompute every ratio and the verdict from the published raw logs only (standard S-3, level 1).

    python verify.py [--logs logs/final] [--out result.json]      # write result.json
    python verify.py --check                                       # recompute and compare with the committed result.json

Reads every run.jsonl under --logs (written by ecc004.py: one 'run' line, one 'eval' line per evaluation point, one 'end' line).
Standard library only, CPU, well under a minute. Nothing here reads model weights, token files or the network.

Ratios at matched quality, per context and seed (METHOD.md section 1):
  Q        the baseline's bits per token at its final evaluation point (sealed validation windows)
  R_FLOP   training FLOPs the method needs to first reach Q, over the baseline's (analytic count, same counter for both arms)
  R_time   the same with measured GPU training time
  R_steps  the same with training tokens; tokens per update are equal on both arms, so this is the ratio of steps the paper reports
Two estimators: 'curve' reads the method's own evaluation points (one run, as in ECC-001); 'budgets' reads the FINAL points of method runs
with different token budgets, each with its own full schedule (the paper's way). Both interpolate linearly in log cost between measured
points; a quality never reached gives a lower bound. Interpolated ratios are labelled ESTIMATED, raw points MEASURED.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The pass rule, copied from PREREG.md when it is frozen. Until then it is a draft and verify.py says so in result.json.
RULES = {
    'status': 'DRAFT, not frozen',
    'primary_estimator': 'budgets',
    'time_source': 'timing block (R_time = R_steps x seconds-per-update ratio, both arms alone on one GPU); in-run time reported only',
    'contexts': {
        '1024': dict(claimed_R=0.25, holds_R_flop=0.25, holds_R_time=0.35, does_not_hold_R_flop=0.8),
        '4096': dict(claimed_R=0.10, holds_R_flop=0.10, holds_R_time=0.14, does_not_hold_R_flop=0.8),
    },
    'min_seeds_for_verdict': 3,
}
KEYS = {'R_FLOP': 'flops', 'R_time': 'train_time_s', 'R_steps': 'tokens'}


def ratio(points, q, c_ref, key):
    """C(q) / c_ref along points; (value, bound) with bound None, 'lower' (q never reached) or 'upper' (below q at the first point).
    Same rule as harness/gym_harness.py ratio() and ECC-001."""
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


def load_timing(logs):
    """timing.jsonl lines (ecc004.timing_block) by (ctx, seed): mean of the A B B A medians per arm, and their ratio."""
    t = {}
    for p in sorted(Path(logs).rglob('timing.jsonl')):
        raw = p.read_bytes()
        for x in (json.loads(l) for l in raw.decode('utf-8').splitlines() if l.strip()):
            t.setdefault((x['ctx'], x['seed']), dict(lines=[], files={}))['lines'].append(x)
            t[(x['ctx'], x['seed'])]['files'][p.relative_to(Path(logs)).as_posix()] = hashlib.sha256(raw).hexdigest()
    for v in t.values():
        per = {arm: statistics.mean(x['median_s'] for x in v['lines'] if x['arm'] == arm) for arm in ('baseline', 'ngpt')}
        v.update(seconds_per_update=per, ratio=per['ngpt'] / per['baseline'], gpus=sorted({x['gpu'] or 'cpu' for x in v['lines']}))
    return t


def load_runs(logs):
    runs = []
    for p in sorted(Path(logs).rglob('run.jsonl')):
        raw = p.read_bytes(); lines = [json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]
        head = lines[0]; end = lines[-1]
        assert head['type'] == 'run' and end['type'] == 'end', f'{p}: not a complete run.jsonl'
        runs.append(dict(path=p.relative_to(Path(logs)).as_posix(), sha256=hashlib.sha256(raw).hexdigest(), head=head, end=end,
                         evals=[x for x in lines[1:-1] if x['type'] == 'eval']))
    return runs


def run_ok(r):
    ch = r['end']['checks']
    ok = r['end']['status'] == 'done' and all(ch.get(k, {}).get('ok') for k in ('causal_init', 'causal_trained', 'finite'))
    return ok, dict(status=r['end']['status'], causal_init=ch.get('causal_init', {}).get('ok'), causal_trained=ch.get('causal_trained', {}).get('ok'),
                    finite=ch.get('finite', {}).get('ok'), skipped_steps=r['end']['skipped_steps'])


def gpus(r):
    return sorted({s['env'].get('gpu') or 'cpu' for s in r['end']['segments']})


def r3(x):
    return round(x, 4)


def verdict_for(ctx, seeds, est):
    rule = RULES['contexts'].get(str(ctx))
    if rule is None:
        return dict(verdict='NO RULE', reason=f'no pass rule for context {ctx}')
    if any(not s['valid'] for s in seeds):
        return dict(verdict='INCONCLUSIVE', reason='a validity check failed')
    tkey = 'R_time_timing' if all('R_time_timing' in s[est] for s in seeds) else 'R_time'
    rf = [s[est]['R_FLOP']['value'] for s in seeds]; rt = [s[est][tkey]['value'] for s in seeds]
    order = ['HOLDS', 'PARTIAL', 'DOES NOT HOLD']
    cls = lambda r: 'HOLDS' if r <= rule['holds_R_flop'] else ('PARTIAL' if r <= rule['does_not_hold_R_flop'] else 'DOES NOT HOLD')
    med_f, med_t = statistics.median(rf), statistics.median(rt)
    out = dict(median_R_FLOP=r3(med_f), median_R_time=r3(med_t), R_time_source=tkey, n_seeds=len(seeds), rule=rule,
               bounds=[s[est]['R_FLOP']['bound'] for s in seeds])
    ambiguous = []                                       # a bound leaves more than one class possible: 'upper' = at most, 'lower' = at least
    for s, r in zip(seeds, rf):
        b = s[est]['R_FLOP']['bound']
        possible = order[:order.index(cls(r)) + 1] if b == 'upper' else (order[order.index(cls(r)):] if b == 'lower' else [cls(r)])
        if len(possible) > 1:
            ambiguous.append(s['seed'])
    if ambiguous:
        out.update(verdict='INCONCLUSIVE', reason=f'a bound leaves the class open for seeds {ambiguous}')
    elif len({cls(r) for r in rf}) > 1:
        out.update(verdict='INCONCLUSIVE', reason=f'seeds fall in different classes ({sorted({cls(r) for r in rf})})')
    elif cls(med_f) == 'HOLDS' and med_t > rule['holds_R_time']:
        out.update(verdict='PARTIAL', reason='FLOP ratio in the claimed range, time ratio above the time limit')
    else:
        out.update(verdict=cls(med_f), reason='median R_FLOP class, all seeds agree')
    if len(seeds) < RULES['min_seeds_for_verdict']:
        out['verdict'] += f' ({len(seeds)} seed{"s" if len(seeds) != 1 else ""} only)'
    return out


def build(logs):
    runs = load_runs(logs); timing = load_timing(logs)
    assert runs, f'no run.jsonl under {logs}'
    data_sha = {json.dumps(r['head']['data_sha256'], sort_keys=True) for r in runs}
    stages = []
    for ctx in sorted({r['head']['ctx'] for r in runs}):
        per_seed = []
        for seed in sorted({r['head']['seed'] for r in runs if r['head']['ctx'] == ctx}):
            group = [r for r in runs if r['head']['ctx'] == ctx and r['head']['seed'] == seed]
            bases = [r for r in group if r['head']['arm'] == 'baseline']; meths = [r for r in group if r['head']['arm'] == 'ngpt']
            if not bases or not meths:
                per_seed.append(dict(seed=seed, missing='baseline' if not bases else 'ngpt')); continue
            base = max(bases, key=lambda r: r['head']['tokens_budget'])
            same = [m for m in meths if m['head']['tokens_budget'] == base['head']['tokens_budget']]
            assert {m['head']['config']['tokens_per_update'] for m in meths} == {base['head']['config']['tokens_per_update']}, 'tokens per update differ'
            okb, cb = run_ok(base); oks = [run_ok(m) for m in meths]
            final = base['evals'][-1]; q = final['bits_per_token']
            s = dict(seed=seed, valid=okb and all(o for o, _ in oks), Q=dict(value=q, label='MEASURED'),
                     baseline=dict(log=base['path'], final_bits_per_token=q, final_bits_per_byte=final['bits_per_byte'], tokens=final['tokens'],
                                   train_time_s=final['train_time_s'], gpus=gpus(base), checks=cb),
                     method_runs=[dict(log=m['path'], tokens_budget=m['head']['tokens_budget'], final_bits_per_token=m['evals'][-1]['bits_per_token'],
                                       train_time_s=m['evals'][-1]['train_time_s'], gpus=gpus(m), checks=c) for m, (o, c) in zip(meths, oks)])
            if same:
                m = same[0]; s['curve'] = {}
                for name, key in KEYS.items():
                    v, b = ratio(m['evals'], q, final[key], key)
                    s['curve'][name] = dict(value=r3(v), bound=b, label='ESTIMATED')
                s['curve']['by_baseline_point'] = []
                for p in base['evals']:
                    row = dict(baseline_update=p['update'], Q=p['bits_per_token'])
                    for name, key in KEYS.items():
                        v, b = ratio(m['evals'], p['bits_per_token'], p[key], key); row[name] = r3(v); row[name + '_bound'] = b
                    s['curve']['by_baseline_point'].append(row)
            if len(meths) >= 2:
                finals = [dict(m['evals'][-1]) for m in meths]
                s['budgets'] = {name: dict(zip(('value', 'bound'), (lambda v, b: (r3(v), b))(*ratio(finals, q, final[key], key))), label='ESTIMATED')
                                for name, key in KEYS.items()}
            tb = timing.get((ctx, seed))
            if tb:
                s['timing'] = dict(seconds_per_update={k: round(v, 5) for k, v in tb['seconds_per_update'].items()}, ratio=r3(tb['ratio']),
                                   gpus=tb['gpus'], label='MEASURED')
                for est in ('curve', 'budgets'):
                    if est in s:
                        st = s[est]['R_steps']
                        s[est]['R_time_timing'] = dict(value=r3(st['value'] * tb['ratio']), bound=st['bound'], label='ESTIMATED')
            per_seed.append(s)
        complete = [s for s in per_seed if 'missing' not in s]
        stage = dict(context=ctx, seeds_run=[s['seed'] for s in complete], seeds_missing=[s for s in per_seed if 'missing' in s], per_seed=complete)
        for est in ('curve', 'budgets'):
            seeds = [s for s in complete if est in s]
            if seeds:
                stage[f'verdict_{est}'] = verdict_for(ctx, seeds, est)
        stage['verdict'] = stage.get(f'verdict_{RULES["primary_estimator"]}', dict(verdict='NO DATA'))
        any_run = next(r for r in runs if r['head']['ctx'] == ctx)
        stage['scale'] = dict(params_baseline=next((r['head']['params'] for r in runs if r['head']['ctx'] == ctx and r['head']['arm'] == 'baseline'), None),
                              params_ngpt=next((r['head']['params'] for r in runs if r['head']['ctx'] == ctx and r['head']['arm'] == 'ngpt'), None),
                              d=any_run['head']['config']['d'], layers=any_run['head']['config']['layers'], vocab=any_run['head']['config']['vocab'],
                              tokens_per_update=any_run['head']['config']['tokens_per_update'],
                              flops_per_token_analytic=any_run['head']['flops_per_token']['analytic'])
        stage['hardware'] = sorted({g for r in runs if r['head']['ctx'] == ctx for g in gpus(r)})
        stages.append(stage)
    return dict(schema='ecc-result/1', check='ECC-004', claim='C12', rules=RULES, stages=stages,
                data_sha256_consistent=len(data_sha) == 1, data_sha256=json.loads(next(iter(data_sha))) if len(data_sha) == 1 else None,
                code_sha256=sorted({r['head']['code_sha256'] for r in runs}),
                logs={**{r['path']: r['sha256'] for r in runs}, **{f: h for v in timing.values() for f, h in v['files'].items()}},
                verify_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=str(HERE / 'logs' / 'final')); ap.add_argument('--out', default=str(HERE / 'result.json'))
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    res = build(a.logs); text = dumps(res)
    for st in res['stages']:
        print(f"context {st['context']}: seeds {st['seeds_run']}, verdict {st['verdict'].get('verdict')}, "
              f"median R_FLOP {st['verdict'].get('median_R_FLOP')}, median R_time {st['verdict'].get('median_R_time')}")
    if a.check:
        old = Path(a.out).read_text(encoding='utf-8')
        if old == text:
            print('OK: result.json matches the logs'); return 0
        print('MISMATCH: result.json differs from what the logs give'); return 1
    Path(a.out).write_text(text, encoding='utf-8', newline='\n'); print('wrote', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
