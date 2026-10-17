"""ECC-001 grid extension (added after the freeze; implements the frozen PREREG tuning rule, no new choice):
"If the best value sits at a grid edge, the grid is extended by one doubling step on that side, for that recipe only. This is allowed at most
twice per side and is reported."  Uses run_ecc001.one() unchanged; appends the extra runs to logs/tune.json and recomputes 'best'.

    python ecc001_extend.py --data-dir DIR
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_ecc001 as R                                                           # noqa: E402


def best_of(runs, name):
    rs = [r for r in runs if r['name'] == name and r['status'] == 'done']
    return min(rs, key=lambda r: r['points'][-1]['bits_per_token'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-dir'); a = p.parse_args(); R.check_harness()
    train, valid = R.data(a.data_dir); tune_valid = valid[256 * R.W:]
    path = R.LOG / 'tune.json'; out = json.loads(path.read_text()); out.setdefault('extensions', [])
    for name in R.LR_GRID:
        for _ in range(2):                                                       # at most twice per side
            lrs = sorted({r['lr'] for r in out['runs'] if r['name'] == name})
            b = best_of(out['runs'], name)['lr']
            if b == lrs[-1]:
                new = b * 2
            elif b == lrs[0]:
                new = b / 2
            else:
                break
            rec = R.one(name, new, 11, train, tune_valid); out['runs'].append(rec)
            out['extensions'].append(dict(name=name, lr=new, reason=f'best {b} at the grid edge', bits=rec['points'][-1]['bits_per_token'] if rec.get('points') else None))
            R.save(path, out); print(name, new, rec['status'], out['extensions'][-1]['bits'], flush=True)
    best = {}
    for name in R.LR_GRID:
        b = best_of(out['runs'], name); lrs = sorted({r['lr'] for r in out['runs'] if r['name'] == name})
        best[name] = dict(lr=b['lr'], bits=b['points'][-1]['bits_per_token'], at_grid_edge=b['lr'] in (lrs[0], lrs[-1]))
    out['best'] = best; R.save(path, out); print(json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
