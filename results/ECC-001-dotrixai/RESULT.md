# Result: C12 nGPT (normalized Transformer), 4x to 20x fewer training steps

Pre-registration: `PREREG.md` (frozen at commit 29623b070f55485a6988bb6b900ffd79ed283b91, 2026-10-17T10:25:32Z)
Author: DotrixAI (run by the DotrixAI research session on one laptop)

## Verdict

**PARTIAL**, at 0.73 M parameters (d 96, 3 layers, context 128, 1,200 updates of 16 x 128 tokens of WikiText-2), seeds 11, 22 and 33, one Intel Core i5-13420H laptop CPU with 2 threads.

Reason from the frozen rule: median R_FLOP class, all seeds agree. Median R_FLOP 0.681; median R_time 0.882.

FLOPs per update are equal on both sides, so R_FLOP is also the ratio of training steps. To reach the baseline's final quality, nGPT needed 0.68 of the baseline's steps (1.47x fewer). The claim is 4x to 20x fewer.

Along the cost-to-quality curve the gain is largest early. At the quality the baseline has at update 250, nGPT needed a median 0.21 of the baseline's FLOPs (4.9x fewer steps), inside the claimed range. The advantage then shrinks; by the final point it is 1.47x.

## Numbers

| quantity | seed 11 | seed 22 | seed 33 | median | label |
|---|---|---|---|---|---|
| Q (target quality, baseline bits per token at update 1,200) | 5.7986 | 5.8288 | 5.8420 | 5.8288 | MEASURED |
| R_FLOP at Q | 0.671 | 0.685 | 0.681 | 0.681 | MEASURED |
| R_time at Q (CPU time) | 0.869 | 0.914 | 0.882 | 0.882 | MEASURED |
| final quality, baseline (bits per token) | 5.7986 | 5.8288 | 5.8420 | 5.8288 | MEASURED |
| final quality, nGPT (bits per token) | 5.4379 | 5.4923 | 5.4754 | 5.4754 | MEASURED |
| CPU seconds per update, baseline / nGPT | 0.540 / 0.699 | 0.525 / 0.700 | 0.538 / 0.697 | | MEASURED |
| FLOPs per update (both sides, torch.utils.flop_counter) | 6,492,782,592 | | | | MEASURED |

Cost-to-quality curve: R at every baseline evaluation point (the quality the baseline has at that update). R_FLOP is the share of the baseline's FLOPs up to that point that nGPT needs to reach the same quality; R_time is the same with CPU time.

| baseline update | Q (seed 11) | R_FLOP s11 / s22 / s33 | R_time s11 / s22 / s33 |
|---|---|---|---|
| 100 | 9.089 | 0.50- / 0.50- / 0.50- | 0.65- / 0.65- / 0.67- |
| 200 | 7.825 | 0.25- / 0.25- / 0.25- | 0.32- / 0.33- / 0.33- |
| 300 | 7.203 | 0.25 / 0.23 / 0.24 | 0.33 / 0.30 / 0.31 |
| 400 | 6.740 | 0.51 / 0.34 / 0.35 | 0.67 / 0.45 / 0.45 |
| 500 | 6.495 | 0.68 / 0.55 / 0.58 | 0.88 / 0.74 / 0.76 |
| 600 | 6.282 | 0.86 / 0.81 / 0.69 | 1.11 / 1.08 / 0.90 |
| 700 | 6.127 | 0.88 / 0.83 / 0.85 | 1.13 / 1.11 / 1.10 |
| 800 | 6.009 | 0.86 / 0.82 / 0.81 | 1.11 / 1.10 / 1.05 |
| 900 | 5.932 | 0.81 / 0.78 / 0.78 | 1.05 / 1.05 / 1.02 |
| 1000 | 5.880 | 0.76 / 0.75 / 0.75 | 0.98 / 1.00 / 0.97 |
| 1100 | 5.826 | 0.72 / 0.72 / 0.71 | 0.93 / 0.96 / 0.92 |
| 1200 | 5.799 | 0.67 / 0.69 / 0.68 | 0.87 / 0.91 / 0.88 |

"+" marks a lower bound (nGPT never reached that quality); "-" marks an upper bound (nGPT was already below it at its first point, update 50).

Tuning (seed 11, validation windows 256-511, bits per token at update 1,200; the sealed windows 0-255 were not scored):

| learning rate | baseline | nGPT |
|---|---|---|
| 0.001 | 6.4072 | — |
| 0.002 | 6.0776 (best) | 6.5540 |
| 0.004 | 6.1477 | 6.0934 |
| 0.008 | 6.3980 | 5.7876 |
| 0.016 | 6.4670 | 5.7669 (best) |
| 0.032 | — | 5.9681 |

Grid extensions under the frozen edge rule: 2 run(s) — baseline 0.001 (best 0.002 at the grid edge); ngpt 0.032 (best 0.016 at the grid edge).

## Checks

- Causality: the harness test (change one token at position 64 and compare all earlier outputs and the other row) passed on the initial and the trained model of all six final runs; largest change 0. The static scan passed for both submissions.
- Leakage: training and evaluation text are WikiText-2 train and validation as rebuilt by `prepare_data.py` (validation rows that also occur in train are dropped; token files match the published SHA-256). Tuning scored validation windows 256-511 only; the verdict uses the sealed windows 0-255. Every target in a window is scored.
- Accounting: FLOPs per update are counted by the harness (`torch.utils.flop_counter`, one forward and backward) and are equal on both sides (6,492,782,592). nGPT's weight renormalisation after each step and its scale parameters are not in the FLOP count; they appear in CPU time. CPU time is process time of the training loop (prepare time is zero on both sides).
- Finite training: all six final runs finished 1,200 updates with finite loss.

Timing conditions of the finals (see `DEVIATIONS.md` §2): outside CPU load as a share of the whole machine, from `logs/load_final.jsonl`, and the slowest 50-update block against the median block of the same run.

| run | wall s | outside load, mean % of machine (samples) | slowest / median block | median block / quiet reference | flag |
|---|---|---|---|---|---|
| baseline seed 11 | 531 | 5.4 (34) | 1.04 | 4.05 | SLOW_STATE |
| ngpt seed 11 | 708 | 5.2 (46) | 1.03 | 4.24 | SLOW_STATE |
| ngpt seed 22 | 700 | 5.3 (44) | 1.02 | 4.25 | SLOW_STATE |
| baseline seed 22 | 508 | 6.3 (33) | 1.05 | 3.94 | SLOW_STATE |
| baseline seed 33 | 521 | 5.5 (34) | 1.02 | 4.05 | SLOW_STATE |
| ngpt seed 33 | 707 | 5.1 (45) | 1.03 | 4.23 | SLOW_STATE |

All final runs ran in the slow state; the per-core sample taken during the finals put the run on the efficiency cores (see Against the prediction). That state slows each recipe by about the same factor: median block / quiet reference 4.05x for the baseline and 4.24x for nGPT. The measured R_time is therefore a CPU-time ratio on the efficiency cores, and the quiet-machine estimate below is the matching ratio on the performance cores.

Quiet-machine R_time (ESTIMATED): R_FLOP x 1.24 (nGPT / baseline CPU seconds per update on a quiet machine, 0.165 / 0.133) = 0.831 / 0.849 / 0.844 (seeds 11 / 22 / 33), median 0.844. The verdict uses the measured R_time, as frozen.

## Against the prediction

- Prediction vs outcome:
  - R_FLOP at Q: predicted median 0.75, 80% interval [0.40, 1.20]; measured median 0.681 (seeds 0.671, 0.681, 0.685).
  - R_time at Q: predicted median 0.85, 80% interval [0.45, 1.40]; measured median 0.882 (seeds 0.869, 0.882, 0.914).
  - Verdict probabilities written before training: HOLDS 0.05, PARTIAL 0.45, DOES NOT HOLD 0.40, INCONCLUSIVE 0.10. Outcome: PARTIAL.
- Anything that differed from the pre-registration, and why:
  - Both best tuning rates sat on a grid edge (baseline at the lowest rate, nGPT at the highest), so the frozen extension rule ran. `ecc001_extend.py` (written after the freeze) carries out that rule with the frozen runner function; it is listed in `DEVIATIONS.md` §1.
  - From 10:34 UTC on, CPU time per update of identical code was about 4x higher than in a run started from an interactive shell (0.69 s against 0.16 s per nGPT update). Tuning decisions use quality only, which this does not change. Before the finals a load gate, a load record and a timing flag were added (`DEVIATIONS.md` §2). The load record then showed the cause: the rest of the machine was mostly idle, and per-core samples put the run on the four efficiency cores of the i5-13420H while the performance cores stayed nearly idle. The runs were started detached (no window), and Windows scheduled them on the efficiency cores (`DEVIATIONS.md` §3).
  - Final runs with a timing flag (SLOW_STATE, pre-declared as OUTSIDE_LOAD): baseline seed 11, ngpt seed 11, ngpt seed 22, baseline seed 22, baseline seed 33, ngpt seed 33. R_FLOP and quality are not affected. The flag could not change the verdict class (that is possible only when the median R_FLOP is at most 0.25), so by the pre-declared rule no pair was re-timed.

## Limits

- Scale, seeds and hardware: 0.73 M parameters, context 128, 1,200 updates (2.46 M training tokens), vocabulary 2,048, three seeds, one laptop CPU with 2 threads. The paper reports its gains at 0.5 B to 1 B parameters and contexts up to 4k on GPUs; none of that is tested here.
- What this result does not show: anything about nGPT at the paper's scale, at long context, or with longer training; the learning-rate grid is coarse (doublings); one baseline recipe (pre-LN, RoPE, SwiGLU, AdamW with weight decay 0.1) is used; CPU time on a laptop is noisy.

## Files

- Code: `submissions/baseline.py`, `submissions/ngpt.py`, `run_ecc001.py`, `prepare_data.py`, `tokenizer.json` (all frozen with the PREREG), `ecc001_extend.py` and `load_monitor.py` (added after the freeze, `DEVIATIONS.md`), `make_result.py` (this file's generator), `diag_denormal_probe.py` (diagnostic only). Harness: `harness/gym_harness.py` of the repository at commit f9beaba, unchanged.
- Logs: `logs/pilot.json`, `logs/tune.json`, `logs/final.json`, `logs/report.json`, `logs/load_final.jsonl`, `logs/core_sample.json`, `logs/diag_denormal.json` (diagnostic only).
