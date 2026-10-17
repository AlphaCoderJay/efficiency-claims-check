"""ECC-001 machine-load gate and monitor for the final runs (added 2026-10-17 after tuning, before the finals; not part of the frozen protocol and
no quality decision uses it).  Reason: during tuning, CPU time per update of identical code rose 4x between about 10:34 and 11:29 UTC while the
runner was alone in the CIR lane (a quiet re-run on the same code gave 0.16 s per nGPT update against 0.69 s in tuning), so outside load on the
laptop inflated process CPU time.  R_time depends on CPU time, so the finals need a quiet machine and a record of the load.

    python load_monitor.py gate       # wait until the whole machine averages below 10% CPU over 60 s (checked every minute)
    python load_monitor.py watch      # every 15 s: machine CPU %, CPU % of the run_ecc001.py process, top other processes each minute;
                                      # appends logs/load_final.jsonl; stops when logs/load_monitor.stop exists
"""
import json
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
LOG = HERE / 'logs' / 'load_final.jsonl'
STOP = HERE / 'logs' / 'load_monitor.stop'
NCPU = psutil.cpu_count()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def runner():
    for p in psutil.process_iter(['cmdline']):
        try:
            c = p.info['cmdline'] or []
            if any(x.endswith('run_ecc001.py') for x in c) and 'final' in c:
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def gate():
    while True:
        load = psutil.cpu_percent(interval=60)
        print(json.dumps(dict(utc=utc(), machine_cpu_percent=load)), flush=True)
        if load < 10:
            return


def top_others(skip):
    procs = []
    for p in psutil.process_iter(['name']):
        if p.pid in skip:
            continue
        try:
            procs.append((p, p.cpu_percent(None)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs


def watch():
    STOP.unlink(missing_ok=True)
    psutil.cpu_percent(None); r, k = None, 0
    others = {p.pid: p for p, _ in top_others({psutil.Process().pid})}
    while not STOP.exists():
        time.sleep(15); k += 1
        if r is None or not r.is_running():
            r = runner()
            if r is not None:
                r.cpu_percent(None)
        rec = dict(utc=utc(), machine_cpu_percent=psutil.cpu_percent(None), runner_pid=r.pid if r else None)
        try:
            rec['runner_cpu_percent_of_machine'] = round(r.cpu_percent(None) / NCPU, 2) if r else None
        except psutil.NoSuchProcess:
            rec['runner_cpu_percent_of_machine'] = None; r = None
        if rec['runner_cpu_percent_of_machine'] is not None:
            rec['outside_cpu_percent_of_machine'] = round(max(0., rec['machine_cpu_percent'] - rec['runner_cpu_percent_of_machine']), 2)
        if k % 4 == 0:
            skip = {psutil.Process().pid} | ({r.pid} if r else set())
            rows = []
            for p, v in top_others(skip):
                if p.pid in others:
                    rows.append((round(v / NCPU, 2), p.info['name']))
                others[p.pid] = p
            rec['top_other_percent_of_machine'] = sorted(rows, reverse=True)[:4]
        with LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    dict(gate=gate, watch=watch)[sys.argv[1]]()
