# Demo report: training-efficiency gym, CPU prototype

Run 2026-10-16T12:26:32Z to 2026-10-16T12:34:28Z on Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, PyTorch 2.7.0+cpu, 2 threads. Training stream: 3,860,300 tokens (WikiText-2 train, BPE 2,048); sealed evaluation: 32,768 tokens of WikiText-2 validation.

Reference recipe: 542,016 parameters, 1200 updates of 16 x 128 tokens, final 6.6559 bits per token (= Q), 136.8 CPU-s, 7,791 GFLOP.

## Grades against the reference

| submission | status | final bits per token | R_time at Q | R_FLOP at Q | prepare share of CPU | reward |
|---|---|---|---|---|---|---|
| ngram_input | done | 5.9863 | 0.508 | 0.462 | 0.0000 | 0.489 |
| counted_prior | done | 6.1424 | 0.099 (upper bound) | 0.083 (upper bound) | 0.0007 | 1.000 |
| strong_reference | done | 5.6365 | 0.103 (upper bound) | 0.083 (upper bound) | 0.0005 | 1.000 |
| no_causal_mask | rejected: causality (initial model) | - | - | - | - | 0.000 |
| batch_coupled | rejected: causality (initial model) | - | - | - | - | 0.000 |
| reads_files | rejected: static scan | - | - | - | - | 0.000 |
| reference_repeat | done | 6.6559 | 1.170 | 1.000 | 0.0000 | 0.000 |

## Cost to quality: R_time at every quality the reference passes

Each column is the reference's quality after that many updates; a value is the submission's CPU time to reach it divided by the reference's. "upper" = already better at the submission's first evaluation point (the true ratio is lower).

| submission | @100 | @200 | @300 | @400 | @500 | @600 | @700 | @800 | @900 | @1000 | @1100 | @1200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ngram_input | 1.21 | 0.95 | 0.79 | 0.71 | 0.66 | 0.65 | 0.63 | 0.63 | 0.61 | 0.57 | 0.54 | 0.51 |
| counted_prior | 1.30 u | 0.64 u | 0.42 u | 0.31 u | 0.25 u | 0.20 u | 0.17 u | 0.15 u | 0.13 u | 0.12 u | 0.11 u | 0.10 u |
| strong_reference | 1.35 u | 0.67 u | 0.44 u | 0.33 u | 0.26 u | 0.21 u | 0.18 u | 0.16 u | 0.14 u | 0.12 u | 0.11 u | 0.10 u |

(u = upper bound, l = lower bound: never reached within the budget.)

## The same two levers against a stronger reference (both levers together, final 5.6365 bits per token)

| submission | R_time at the strong Q | reward |
|---|---|---|
| ngram_input | 0.882 (lower bound) | 0.000 |
| counted_prior | 0.927 (lower bound) | 0.000 |

## Noise

The reference rerun at the end gives identical bits (deterministic data order and seed: 6.6559) but R_time 1.170: the CPU time of two identical runs differs by 17% on this laptop, the same as a reward of 0.11. Rewards smaller than that are noise here; the FLOP ratio is exact.

## Rejected submissions

- **no_causal_mask**: rejected: causality (initial model); checks {"static": {"ok": true, "problems": []}, "causal_init": {"ok": false, "max_change_before_t": 0.06864636391401291, "max_change_other_row": 0.0}}
- **batch_coupled**: rejected: causality (initial model); checks {"static": {"ok": true, "problems": []}, "causal_init": {"ok": false, "max_change_before_t": 0.0, "max_change_other_row": 0.1910841315984726}}
- **reads_files**: rejected: static scan; checks {"static": {"ok": false, "problems": ["import os", "name open"]}}
