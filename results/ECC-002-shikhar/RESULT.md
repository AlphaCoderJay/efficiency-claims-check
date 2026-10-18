# Result: C13 dynamic short convolutions

Pre-registration: [PREREG.md](./PREREG.md), frozen in commit `e9f8251` before tuning and final runs.

## Verdict

**HOLDS AT TESTED SCALE**, under the preregistered rule, at 0.54M baseline / 0.58M method parameters, context 128, three seeds, and CPU.

The median R_FLOP at baseline final quality was 0.5555, below the preregistered 0.7519 threshold. All three seeds were below 1.0. The method required a median 1.80x less estimated training compute to reach the baseline target. This is a small-scale result only; it does not establish the paper's result at 150M–2B parameters.

## Setup and tuning

- Harness: the repository CPU harness, unchanged.
- Data: WikiText-2 with the published 2,048-token BPE; 3,860,300 training tokens. Learning-rate tuning used validation windows 256–511; final scoring used the separate sealed windows 0–255.
- Model: the harness's 96-wide, 3-layer, 4-head GPT. The method adds independent rank-7, width-4 low-rank causal dynamic convolutions to Q, K and V in every layer.
- Both recipes used AdamW, 1,200 updates, batch 16 x 128, and the same learning-rate grid. The selected learning rate was 0.004 for both.
- The implementation uses the authors' published PyTorch reference math for the convolution; it does not use their custom Triton kernels.
- Pilot runs before freezing used 40 updates per recipe on tuning windows only. They confirmed the static and causality checks passed.

## Final measurements

Bits/token and CPU time are measured; R_FLOP combines the harness's measured operator count with the preregistered analytic estimate for pointwise convolution operations that its FLOP counter omits.

| seed | baseline Q (bits/token) | method final (bits/token) | R_FLOP at Q | R_time at Q | label |
|---|---:|---:|---:|---:|---|
| 11 | 6.66667 | 6.32285 | 0.6415 | 1.1602 | ESTIMATED FLOPs; MEASURED quality/time |
| 22 | 6.75251 | 6.16113 | 0.5047 | 0.9200 | ESTIMATED FLOPs; MEASURED quality/time |
| 33 | 6.77174 | 6.31906 | 0.5555 | 0.9738 | ESTIMATED FLOPs; MEASURED quality/time |
| **median** | — | — | **0.5555** | **0.9738** | **ESTIMATED FLOPs; MEASURED time** |

Per-update FLOPs were 6,492,782,592 for the baseline and 6,915,600,000 for the method after adding 51,228,288 analytically estimated pointwise FLOPs. The method therefore costs more per update; its matched-quality advantage comes from reaching the baseline target in fewer updates. Median CPU time at matched quality was approximately equal, with one seed slower than baseline.

All runs passed the static scan, initial and trained causality checks, and finite-loss check. Full run records, including cost-to-quality curves and tuning results, are in `logs/`.

## Limitations

The result measures a 0.58M-parameter CPU fallback, not the custom GPU kernels or paper-scale training. The method's CPU runtime is implementation-dependent and should not be read as a GPU latency result. The FLOP ratio includes an analytic pointwise-operation estimate because the harness counter omits those ops.
