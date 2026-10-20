"""ECC-004 calibration: one GPU session before the pre-registration. It measures what the PREREG budget needs; no number here decides
a verdict, and only the TUNING validation windows are ever scored (the sealed windows stay unread).

    python calibrate.py --data-dir DIR --out DIR [--parts checks,throughput,stability,parallel] [--deadline-min 150] [--smoke]

Parts:
  checks      per arm and context: causality (fp32 and fp16 autocast) on the GPU, parameters, FLOPs per token (analytic vs flop_counter)
  throughput  per arm, context (1024, 4096) and micro-batch size: GPU seconds per update after warm-up, tokens per second, peak memory
  stability   short fp16 runs per arm at a moderate and a high learning rate (context 1024): skipped steps, non-finite losses, loss curve
  parallel    (two GPUs only) both arms at once, one per GPU, against the same runs alone: the interference factor for R_time
Every training measurement goes through ecc004.train, the same code the real runs use.
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

import ecc004 as E

HERE = Path(__file__).resolve().parent


def per_update_times(rec, skip):
    cur = rec['curve']; t = [b['train_time_s'] - a['train_time_s'] for a, b in zip(cur, cur[1:])]
    t = t[skip:]
    return dict(n=len(t), median_s=sorted(t)[len(t) // 2], mean_s=sum(t) / len(t), min_s=min(t), max_s=max(t)) if t else None


def base_config(smoke, **kw):
    c = dict(eval_split='tune', eval_tokens=2 ** 16, eval_fracs=(1.0,), log_every=1)
    if smoke:
        c.update(d=64, layers=2, heads=2, hidden=256, ctx=128, tokens_per_update=128 * 8, eval_tokens=128 * 16)
    c.update(kw)
    return E.Config(**c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--parts', default='checks,throughput,stability,parallel'); ap.add_argument('--deadline-min', type=float, default=150)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); cuda = dev.type == 'cuda'
    parts = a.parts.split(','); smoke = a.smoke
    rep = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), env=E.env_info(dev), gpus=torch.cuda.device_count() if cuda else 0,
               code_sha256=E.CODE_SHA, parts=parts, checks=[], throughput=[], stability=[], parallel=[])
    log = lambda s: print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)
    save = lambda: E.save_json(out / 'calibration.json', rep)
    late = lambda: (time.time() - t0) / 60 > a.deadline_min
    ctxs = (1024, 4096) if not smoke else (128,)
    vocab = json.loads((Path(a.data_dir) / 'meta.json').read_text())['vocab']
    first = True

    if 'checks' in parts:
        for arm in ('baseline', 'ngpt'):
            for ctx in ctxs:
                c = base_config(smoke, arm=arm, ctx=ctx, vocab=vocab); torch.manual_seed(0); m = E.LM(c).to(dev)
                r = dict(arm=arm, ctx=ctx, params=E.n_params(m), flops_analytic=E.flops_per_token_analytic(c),
                         flops_counter=E.flops_per_token_counter(m, c, dev), causal_fp32=E.causality(m, c, dev),
                         causal_amp=E.causality(m, c, dev, amp=True, tol=float('inf')))
                r['counter_over_analytic'] = r['flops_counter'] / r['flops_analytic']
                rep['checks'].append(r); log(f'checks {arm} {ctx}: causal {r["causal_fp32"]["ok"]}, counter/analytic {r["counter_over_analytic"]:.4f}')
                del m; torch.cuda.empty_cache() if cuda else None
        save()

    if 'throughput' in parts:
        micros = {1024: (32, 16, 8), 4096: (8, 4, 2), 128: (8, 4)}
        n_up = 25 if not smoke else 6
        for ctx in ctxs:
            for arm in ('baseline', 'ngpt'):
                for mr in micros[ctx]:
                    if late():
                        break
                    c = base_config(smoke, arm=arm, ctx=ctx, vocab=vocab, micro_rows=mr, lr=1e-3)
                    c = dataclasses.replace(c, tokens=float(n_up * c.tokens_per_update))
                    if cuda:
                        torch.cuda.reset_peak_memory_stats(dev)
                    r = dict(arm=arm, ctx=ctx, micro_rows=mr, tokens_per_update=c.tokens_per_update)
                    try:
                        rec = E.train(c, a.data_dir, out / 'tmp', device=dev, check_hashes=first, log=lambda s: None); first = False
                        tt = per_update_times(rec, skip=5 if not smoke else 1)
                        r.update(status=rec['status'], per_update=tt, tokens_per_s=c.tokens_per_update / tt['median_s'],
                                 peak_mem_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30 if cuda else None,
                                 skipped=rec['skipped_steps'], data_sha256=rec['data']['sha256_checked'] or None)
                    except torch.cuda.OutOfMemoryError as ex:
                        r.update(status='OOM', error=str(ex)[:200]); torch.cuda.empty_cache()
                    rep['throughput'].append(r); save()
                    log(f'throughput {arm} ctx {ctx} micro {mr}: {r.get("status")} {r.get("per_update")} mem {r.get("peak_mem_gb")}')

    if 'stability' in parts:
        n_up = 200 if not smoke else 10
        for arm, lrs in (('baseline', (2e-3, 8e-3)), ('ngpt', (1.6e-2, 6.4e-2))):
            for lr in lrs:
                if late():
                    break
                ctx = ctxs[0]
                best = [t for t in rep['throughput'] if t['arm'] == arm and t['ctx'] == ctx and t.get('status') == 'done']
                mr = min(best, key=lambda t: t['per_update']['median_s'])['micro_rows'] if best else (16 if not smoke else 4)
                c = base_config(smoke, arm=arm, ctx=ctx, vocab=vocab, micro_rows=mr, lr=lr, log_every=10, eval_fracs=(0.25, 0.5, 1.0))
                c = dataclasses.replace(c, tokens=float(n_up * c.tokens_per_update))
                rec = E.train(c, a.data_dir, out / 'tmp', device=dev, check_hashes=False, log=lambda s: None)
                r = dict(arm=arm, lr=lr, ctx=ctx, micro_rows=mr, updates=c.updates, status=rec['status'], skipped=rec['skipped_steps'],
                         nonfinite=rec['nonfinite_losses'], curve=[(p['update'], p['loss'], p['grad_norm'], p['scale']) for p in rec['curve']],
                         tune_bits=[(p['update'], p['bits_per_token']) for p in rec['points']], train_time_s=rec['train_time_s'])
                rep['stability'].append(r); save()
                log(f'stability {arm} lr {lr}: {r["status"]}, skipped {r["skipped"]}, nonfinite {r["nonfinite"]}, tune bits {r["tune_bits"]}')

    if 'parallel' in parts and cuda and torch.cuda.device_count() >= 2 and not late():
        ctx = 1024; n_up = 40
        procs = []
        for gpu, arm in ((0, 'baseline'), (1, 'ngpt')):
            best = [t for t in rep['throughput'] if t['arm'] == arm and t['ctx'] == ctx and t.get('status') == 'done']
            mr = min(best, key=lambda t: t['per_update']['median_s'])['micro_rows'] if best else 16
            o = out / f'par_{arm}'
            cmd = [sys.executable, str(HERE / 'ecc004.py'), 'train', '--arm', arm, '--ctx', str(ctx), '--vocab', str(vocab), '--micro-rows', str(mr),
                   '--lr', '1e-3', '--tokens', str(n_up * 131072), '--eval-split', 'tune', '--eval-tokens', str(2 ** 16), '--log-every', '1',
                   '--data-dir', a.data_dir, '--out', str(o)]
            procs.append((arm, mr, o, subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=subprocess.PIPE,
                                                        stderr=subprocess.STDOUT, text=True)))
        for arm, mr, o, p in procs:
            txt = p.communicate()[0]
            rec = json.loads((o / 'log.json').read_text()) if (o / 'log.json').exists() else None
            alone = next((t for t in rep['throughput'] if t['arm'] == arm and t['ctx'] == ctx and t['micro_rows'] == mr and t.get('status') == 'done'), None)
            tt = per_update_times(rec, skip=5) if rec else None
            r = dict(arm=arm, micro_rows=mr, returncode=p.returncode, per_update=tt, gpu=rec['segments'][0]['env'].get('gpu') if rec else None,
                     alone_median_s=alone['per_update']['median_s'] if alone else None, tail=txt[-500:])
            if tt and alone:
                r['slowdown_vs_alone'] = tt['median_s'] / alone['per_update']['median_s']
            rep['parallel'].append(r); save(); log(f'parallel {arm}: {r.get("slowdown_vs_alone")}')

    rep['seconds'] = round(time.time() - t0, 1); save()
    for f in [out / 'tmp' / 'ckpt.pt'] + [out / f'par_{arm}' / 'ckpt.pt' for arm in ('baseline', 'ngpt')]:   # large and not needed
        if f.exists():
            f.unlink()
    log('done')


if __name__ == '__main__':
    main()
