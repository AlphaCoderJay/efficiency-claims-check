"""Render claims/claims.csv as claims/CLAIMS.md (the readable table). Run after editing the CSV.

    python tools/make_claims_md.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(ROOT / 'claims' / 'claims.csv', encoding='utf-8')))

STATUS = {
    'OPEN': 'no independent matched compute check found yet',
    'SUPPORTED': 'independent work points the same way',
    'CONTESTED': 'independent work disagrees',
    'SHRINKS_WITH_SCALE': 'holds, but the gain shrinks at larger scale',
    'CHECKED': 'checked in this project; the verdict and its scale are in the row and in results/',
}

out = ['# Claims to check', '',
       'Generated from `claims.csv` by `tools/make_claims_md.py`. Do not edit by hand.', '',
       'Each row is a published claim as we read it from the source. Check the source before relying on a row.',
       '"Claimed ratio" is compute of the method divided by compute of the baseline at equal quality, so below 1 means cheaper.', '',
       'Status: ' + '; '.join(f'**{k}** = {v}' for k, v in STATUS.items()) + '.', '']
fams = []
for r in rows:
    if r['family'] not in fams:
        fams.append(r['family'])
for f in fams:
    out += [f'## {f.capitalize()}', '',
            '| id | claim | source | claimed ratio | scale in source | independent evidence | status | small scale test |',
            '|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['family'] == f:
            src = f"[{r['source']}]({r['url']}) ({r['year']})"
            out.append(f"| {r['id']} | {r['claim']} | {src} | {r['claimed_ratio']} | {r['scale_in_source']} | "
                       f"{r['independent_evidence']} | {r['status']} | {r['small_scale_test']} |")
    out.append('')
(ROOT / 'claims' / 'CLAIMS.md').write_text('\n'.join(out), encoding='utf-8', newline='\n')
print(len(rows), 'claims written')
