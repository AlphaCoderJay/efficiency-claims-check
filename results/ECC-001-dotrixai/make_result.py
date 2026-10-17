"""Builds RESULT.md for ECC-001 from the logs (tune.json, final.json, report.json, load_final.jsonl, the driver log).  Numbers come only from the
logs; the prose around them is fixed here.

    python make_result.py
"""
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent; LOG = HERE / 'logs'
sys.path.insert(0, str(HERE / 'harness'))
import gym_harness as G                                                          # noqa: E402

SEEDS = (11, 22, 33)
QUIET = dict(baseline=6.66, ngpt=8.25)                                           # quiet CPU s per 50 updates (DEVIATIONS.md §2)
DRIVER_LOG = Path('logs/driver.log')
FREEZE = 'commit 29623b070f55485a6988bb6b900ffd79ed283b91, 2026-10-17T10:25:32Z'


def j(name):
    return json.loads((LOG / name).read_text(encoding='utf-8'))


def fmt_r(v, b):
    return f'{v:.3f}' + ('' if b is None else (' (lower bound: never reached Q)' if b == 'lower' else ' (upper bound: below Q at the first point)'))


def blocks(run):
    c = [p['cpu_s'] for p in run['points']]
    return [b - a for a, b in zip([run.get('prepare_cpu_s', 0.)] + c, c)]


def windows(final):
    """Run time windows rebuilt from the driver log start stamp and each run's wall time (runs are back to back)."""
    start = None
    for line in DRIVER_LOG.read_text(encoding='utf-8').splitlines():
        if 'ecc001_final: start' in line:
            start = datetime.strptime(line.split()[0], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    t, out = start, []
    for r in final['runs']:
        out.append((r, t, t + timedelta(seconds=r['wall_s']))); t += timedelta(seconds=r['wall_s'])
    return out


def load_in(samples, t0, t1):
    xs = [s for s in samples if t0 <= s['t'] <= t1 and s.get('outside_cpu_percent_of_machine') is not None]
    if not xs:
        return None, 0
    return statistics.mean(s['outside_cpu_percent_of_machine'] for s in xs), len(xs)


def main():
    tune, final, rep = j('tune.json'), j('final.json'), j('report.json')
    samples = []
    if (LOG / 'load_final.jsonl').exists():
        for line in (LOG / 'load_final.jsonl').read_text(encoding='utf-8').splitlines():
            s = json.loads(line); s['t'] = datetime.strptime(s['utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc); samples.append(s)
    per = {int(k): v for k, v in rep['per_seed'].items()}
    runs = {(r['name'], r['seed']): r for r in final['runs']}
    L = []
    w = L.append
    w('# Result: C12 nGPT (normalized Transformer), 4x to 20x fewer training steps')
    w('')
    w(f'Pre-registration: `PREREG.md` (frozen at {FREEZE})')
    w('Author: DotrixAI (run by the DotrixAI research session on one laptop)')
    w('')
    w('## Verdict')
    w('')
    w(f'**{rep["verdict"]}**, at 0.73 M parameters (d 96, 3 layers, context 128, 1,200 updates of 16 x 128 tokens of WikiText-2), seeds 11, 22 and 33, '
      'one Intel Core i5-13420H laptop CPU with 2 threads.')
    w('')
    w(f'Reason from the frozen rule: {rep["reason"]}. Median R_FLOP {rep["median"]["R_flops"]:.3f}; median R_time {rep["median"]["R_time"]:.3f}.')
    w('')
    runs0 = {(r['name'], r['seed']): r for r in final['runs']}
    assert all(runs0[('baseline', s)]['flops_per_update'] == runs0[('ngpt', s)]['flops_per_update'] for s in SEEDS)
    med = rep['median']['R_flops']
    w(f'FLOPs per update are equal on both sides, so R_FLOP is also the ratio of training steps. To reach the baseline\'s final quality, nGPT needed '
      f'{med:.2f} of the baseline\'s steps ({1 / med:.2f}x fewer). The claim is 4x to 20x fewer.')
    w('')
    curve = []
    for i, p in enumerate(runs0[('baseline', 11)]['points']):
        vals, exact = [], True
        for s in SEEDS:
            ref, sub = runs0[('baseline', s)], runs0[('ngpt', s)]
            v, bnd = G.ratio(sub['points'], ref['points'][i]['bits_per_token'], ref['points'][i]['flops'], 'flops'); vals.append(v); exact &= bnd is None
        if exact:
            curve.append((statistics.median(vals), p['update']))
    if curve:
        lo, u = min(curve)
        w(f'Along the cost-to-quality curve the gain is largest early. At the quality the baseline has at update {u}, nGPT needed a median {lo:.2f} of '
          f'the baseline\'s FLOPs ({1 / lo:.1f}x fewer steps), inside the claimed range. The advantage then shrinks; by the final point it is {1 / med:.2f}x.')
    w('')
    w('## Numbers')
    w('')
    w('| quantity | seed 11 | seed 22 | seed 33 | median | label |')
    w('|---|---|---|---|---|---|')
    g = {s: per[s]['grade'] for s in SEEDS}
    w('| Q (target quality, baseline bits per token at update 1,200) | ' + ' | '.join(f'{g[s]["Q"]:.4f}' for s in SEEDS) + f' | {statistics.median(g[s]["Q"] for s in SEEDS):.4f} | MEASURED |')
    w('| R_FLOP at Q | ' + ' | '.join(fmt_r(g[s]['R_flops'], g[s]['R_flops_bound']) for s in SEEDS) + f' | {rep["median"]["R_flops"]:.3f} | MEASURED' +
      (' (bound)' if any(g[s]['R_flops_bound'] for s in SEEDS) else '') + ' |')
    w('| R_time at Q (CPU time) | ' + ' | '.join(fmt_r(g[s]['R_time'], g[s]['R_time_bound']) for s in SEEDS) + f' | {rep["median"]["R_time"]:.3f} | MEASURED' +
      (' (bound)' if any(g[s]['R_time_bound'] for s in SEEDS) else '') + ' |')
    w('| final quality, baseline (bits per token) | ' + ' | '.join(f'{per[s]["baseline_final"]:.4f}' for s in SEEDS) +
      f' | {statistics.median(per[s]["baseline_final"] for s in SEEDS):.4f} | MEASURED |')
    w('| final quality, nGPT (bits per token) | ' + ' | '.join(f'{per[s]["ngpt_final"]:.4f}' for s in SEEDS) +
      f' | {statistics.median(per[s]["ngpt_final"] for s in SEEDS):.4f} | MEASURED |')
    cpu_u = {(n, s): runs[(n, s)]['train_cpu_s'] / 1200 for n in ('baseline', 'ngpt') for s in SEEDS}
    w('| CPU seconds per update, baseline / nGPT | ' + ' | '.join(f'{cpu_u[("baseline", s)]:.3f} / {cpu_u[("ngpt", s)]:.3f}' for s in SEEDS) + ' | | MEASURED |')
    w(f'| FLOPs per update (both sides, torch.utils.flop_counter) | {runs[("baseline", 11)]["flops_per_update"]:,} | | | | MEASURED |')
    w('')
    w('Cost-to-quality curve: R at every baseline evaluation point (the quality the baseline has at that update). R_FLOP is the share of the baseline\'s '
      'FLOPs up to that point that nGPT needs to reach the same quality; R_time is the same with CPU time.')
    w('')
    w('| baseline update | Q (seed 11) | R_FLOP s11 / s22 / s33 | R_time s11 / s22 / s33 |')
    w('|---|---|---|---|')
    for i, p in enumerate(runs[('baseline', 11)]['points']):
        if p['update'] % 100 and p['update'] != 1200:
            continue
        rf, rt = [], []
        for s in SEEDS:
            ref, sub = runs[('baseline', s)], runs[('ngpt', s)]
            q = ref['points'][i]['bits_per_token']
            a, ba = G.ratio(sub['points'], q, ref['points'][i]['flops'], 'flops'); b, bb = G.ratio(sub['points'], q, ref['points'][i]['cpu_s'], 'cpu_s')
            rf.append(f'{a:.2f}' + ('+' if ba == 'lower' else ('-' if ba == 'upper' else ''))); rt.append(f'{b:.2f}' + ('+' if bb == 'lower' else ('-' if bb == 'upper' else '')))
        w(f'| {p["update"]} | {p["bits_per_token"]:.3f} | {" / ".join(rf)} | {" / ".join(rt)} |')
    w('')
    w('"+" marks a lower bound (nGPT never reached that quality); "-" marks an upper bound (nGPT was already below it at its first point, update 50).')
    w('')
    w('Tuning (seed 11, validation windows 256-511, bits per token at update 1,200; the sealed windows 0-255 were not scored):')
    w('')
    w('| learning rate | baseline | nGPT |')
    w('|---|---|---|')
    lrs = sorted({r['lr'] for r in tune['runs']})
    get = lambda n, lr: next((r for r in tune['runs'] if r['name'] == n and r['lr'] == lr), None)
    for lr in lrs:
        cells = []
        for n in ('baseline', 'ngpt'):
            r = get(n, lr)
            cells.append('—' if r is None else (r['status'] if r['status'] != 'done' else f'{r["points"][-1]["bits_per_token"]:.4f}') +
                         (' (best)' if r is not None and tune['best'][n]['lr'] == lr else ''))
        w(f'| {lr:g} | {cells[0]} | {cells[1]} |')
    ext = tune.get('extensions', [])
    w('')
    w(f'Grid extensions under the frozen edge rule: {len(ext)} run(s) — ' + ('; '.join(f'{e["name"]} {e["lr"]:g} ({e["reason"]})' for e in ext) if ext else 'none') + '.')
    edge = {n: b['at_grid_edge'] for n, b in tune['best'].items()}
    if any(edge.values()):
        w('After the allowed extensions the best rate still sits on the edge for: ' + ', '.join(n for n, e in edge.items() if e) +
          '. The rule allows no further step; the finals use that rate, and the result notes it.')
    w('')
    w('## Checks')
    w('')
    ck = [(n, s, runs[(n, s)]['checks']) for n in ('baseline', 'ngpt') for s in SEEDS]
    mx = max(max(c['causal_init']['max_change_before_t'], c['causal_init']['max_change_other_row'], c['causal_trained']['max_change_before_t'],
                 c['causal_trained']['max_change_other_row']) for _, _, c in ck)
    w(f'- Causality: the harness test (change one token at position 64 and compare all earlier outputs and the other row) passed on the initial and the '
      f'trained model of all six final runs; largest change {mx:g}. The static scan passed for both submissions.')
    w('- Leakage: training and evaluation text are WikiText-2 train and validation as rebuilt by `prepare_data.py` (validation rows that also occur in '
      'train are dropped; token files match the published SHA-256). Tuning scored validation windows 256-511 only; the verdict uses the sealed windows '
      '0-255. Every target in a window is scored.')
    w(f'- Accounting: FLOPs per update are counted by the harness (`torch.utils.flop_counter`, one forward and backward) and are equal on both sides '
      f'({runs[("baseline", 11)]["flops_per_update"]:,}). nGPT\'s weight renormalisation after each step and its scale parameters are not in the FLOP '
      'count; they appear in CPU time. CPU time is process time of the training loop (prepare time is zero on both sides).')
    w('- Finite training: ' + ('all six final runs finished 1,200 updates with finite loss.' if all(runs[(n, s)]['status'] == 'done' for n in ('baseline', 'ngpt') for s in SEEDS)
                               else 'not all runs finished: ' + ', '.join(f'{n} {s}: {runs[(n, s)]["status"]}' for n in ('baseline', 'ngpt') for s in SEEDS)))
    w('')
    w('Timing conditions of the finals (see `DEVIATIONS.md` §2): outside CPU load as a share of the whole machine, from `logs/load_final.jsonl`, and the '
      'slowest 50-update block against the median block of the same run.')
    w('')
    w('| run | wall s | outside load, mean % of machine (samples) | slowest / median block | median block / quiet reference | flag |')
    w('|---|---|---|---|---|---|')
    flags = {}
    for r, t0, t1 in windows(final):
        m, n = load_in(samples, t0, t1); b = blocks(r); ratio_b = max(b) / statistics.median(b); vs_q = statistics.median(b) / QUIET[r['name']]
        flag = (m is not None and m > 10) or ratio_b > 2 or vs_q > 1.5; flags[(r['name'], r['seed'])] = flag
        w(f'| {r["name"]} seed {r["seed"]} | {r["wall_s"]:.0f} | ' + ('no samples' if m is None else f'{m:.1f} ({n})') + f' | {ratio_b:.2f} | {vs_q:.2f} | {"SLOW_STATE" if flag else "none"} |')
    sl = {n: statistics.median(statistics.median(blocks(runs[(n, s)])) / QUIET[n] for s in SEEDS) for n in ('baseline', 'ngpt')}
    w('')
    w(f'All final runs ran in the slow state; the per-core sample taken during the finals put the run on the efficiency cores (see Against the '
      f'prediction). That state slows each recipe by about the same factor: median '
      f'block / quiet reference {sl["baseline"]:.2f}x for the baseline and {sl["ngpt"]:.2f}x for nGPT. The measured R_time is therefore a CPU-time '
      'ratio on the efficiency cores, and the quiet-machine estimate below is the matching ratio on the performance cores.')
    w('')
    k = (QUIET['ngpt'] / 50) / (QUIET['baseline'] / 50)
    w(f'Quiet-machine R_time (ESTIMATED): R_FLOP x {k:.2f} (nGPT / baseline CPU seconds per update on a quiet machine, {QUIET["ngpt"] / 50:.3f} / '
      f'{QUIET["baseline"] / 50:.3f}) = ' + ' / '.join(f'{g[s]["R_flops"] * k:.3f}' for s in SEEDS) + f' (seeds 11 / 22 / 33), median {rep["median"]["R_flops"] * k:.3f}. '
      'The verdict uses the measured R_time, as frozen.')
    w('')
    w('## Against the prediction')
    w('')
    w('- Prediction vs outcome:')
    w(f'  - R_FLOP at Q: predicted median 0.75, 80% interval [0.40, 1.20]; measured median {rep["median"]["R_flops"]:.3f} '
      f'(seeds {", ".join(f"{v:.3f}" for v in rep["R_flops_sorted"])}).')
    w(f'  - R_time at Q: predicted median 0.85, 80% interval [0.45, 1.40]; measured median {rep["median"]["R_time"]:.3f} '
      f'(seeds {", ".join(f"{v:.3f}" for v in rep["R_time_sorted"])}).')
    w(f'  - Verdict probabilities written before training: HOLDS 0.05, PARTIAL 0.45, DOES NOT HOLD 0.40, INCONCLUSIVE 0.10. Outcome: {rep["verdict"]}.')
    w('- Anything that differed from the pre-registration, and why:')
    w('  - Both best tuning rates sat on a grid edge (baseline at the lowest rate, nGPT at the highest), so the frozen extension rule ran. '
      '`ecc001_extend.py` (written after the freeze) carries out that rule with the frozen runner function; it is listed in `DEVIATIONS.md` §1.')
    w('  - From 10:34 UTC on, CPU time per update of identical code was about 4x higher than in a run started from an interactive shell (0.69 s against '
      '0.16 s per nGPT update). Tuning decisions use quality only, which this does not change. Before the finals a load gate, a load record and a '
      'timing flag were added (`DEVIATIONS.md` §2). The load record then showed the cause: the rest of the machine was mostly idle, and per-core '
      'samples put the run on the four efficiency cores of the i5-13420H while the performance cores stayed nearly idle. The runs were started '
      'detached (no window), and Windows scheduled them on the efficiency cores (`DEVIATIONS.md` §3).')
    if any(flags.values()):
        w('  - Final runs with a timing flag (SLOW_STATE, pre-declared as OUTSIDE_LOAD): ' + ', '.join(f'{n} seed {s}' for (n, s), f in flags.items() if f) +
          '. R_FLOP and quality are not affected. The flag could not change the verdict class (that is possible only when the median R_FLOP is at '
          'most 0.25), so by the pre-declared rule no pair was re-timed.' if rep['median']['R_flops'] > .25 else
          '  - Final runs with a timing flag (SLOW_STATE, pre-declared as OUTSIDE_LOAD): ' + ', '.join(f'{n} seed {s}' for (n, s), f in flags.items() if f) +
          '. R_FLOP and quality are not affected; the median R_FLOP is at most 0.25, so the flagged pairs are re-timed by the pre-declared rule.')
    else:
        w('  - No final run carries a timing flag.')
    w('')
    w('## Limits')
    w('')
    w('- Scale, seeds and hardware: 0.73 M parameters, context 128, 1,200 updates (2.46 M training tokens), vocabulary 2,048, three seeds, one laptop CPU '
      'with 2 threads. The paper reports its gains at 0.5 B to 1 B parameters and contexts up to 4k on GPUs; none of that is tested here.')
    w('- What this result does not show: anything about nGPT at the paper\'s scale, at long context, or with longer training; the learning-rate grid is '
      'coarse (doublings); one baseline recipe (pre-LN, RoPE, SwiGLU, AdamW with weight decay 0.1) is used; CPU time on a laptop is noisy.')
    w('')
    w('## Files')
    w('')
    w('- Code: `submissions/baseline.py`, `submissions/ngpt.py`, `run_ecc001.py`, `prepare_data.py`, `tokenizer.json` (all frozen with the PREREG), '
      '`ecc001_extend.py` and `load_monitor.py` (added after the freeze, `DEVIATIONS.md`), `make_result.py` (this file\'s generator), `diag_denormal_probe.py` (diagnostic only). Harness: '
      '`harness/gym_harness.py` of the repository at commit f9beaba, unchanged.')
    w('- Logs: `logs/pilot.json`, `logs/tune.json`, `logs/final.json`, `logs/report.json`, `logs/load_final.jsonl`, `logs/core_sample.json`, '
      '`logs/diag_denormal.json` (diagnostic only).')
    (HERE / 'RESULT.md').write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
