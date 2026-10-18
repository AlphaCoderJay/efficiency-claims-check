# Pre-registration: C13 dynamic short convolutions

Frozen at: the timestamp of the commit that adds this file
Author: DotrixAI

## Claim

- Claim id: C13.
- Source: [Sieberling et al., Dynamic Short Convolutions Improve Transformers](https://arxiv.org/abs/2606.03825) (2026).
- Testable part: adding low-rank causal dynamic short convolutions to Transformer query, key and value streams yields the quality of a standard Transformer at no more than 1/1.33 of its training FLOPs.
- The paper reports scaling-law results at 150M–2B parameters. This check is only at the harness's small CPU scale and will not establish results at those model scales.

## Setup

- Model: harness GPT, width 96, 3 layers, 4 heads, context 128; the method adds separate rank-7, width-4 low-rank dynamic convolutions to Q, K and V in every layer.
- Data: WikiText-2 train and validation token streams in `data/`, byte-level BPE vocabulary 2,048; same data for both recipes.
- Training: batch 16, 1,200 updates, evaluation every 100 updates.
- Baseline: `harness/submissions/reference.py`, AdamW, learning-rate grid 0.002, 0.004, 0.008, 0.016.
- Method: `submissions/dynamic_qkv.py`, same optimizer, schedule and learning-rate grid. Select each recipe's learning rate using its final bits/token on validation windows 256–511.
- Evaluation: final target Q is each seed's baseline quality at update 1,200 on sealed validation windows 0–255. Hyperparameter selection never uses these windows.
- Seeds: 11, 22 and 33. Run baseline and method once per seed; alternate order by seed.
- Hardware: laptop CPU, 2 PyTorch threads. CPU time will be reported, but it cannot test the paper's optimized Triton/GPU runtime.
- Pilot runs before freeze: one 40-update run per recipe, seed 11, scored only on tuning windows 256–511. Both passed static and initial/trained causality checks. At update 40, baseline and method scored 9.22090 and 9.22031 bits/token; these pilot scores do not select hyperparameters or affect the sealed evaluation. The reference PyTorch method took 32.3 CPU seconds versus 13.8 for baseline, so CPU time is descriptive only.

## Accounting

- Primary measure is R_FLOP at Q, with the paper's QKV result interpreted as a pass threshold of 1/1.33 = 0.752.
- The harness FLOP counter records the dynamic low-rank projections but omits pointwise convolution operations. `run_c13.py` adds an analytic forward/backward estimate for the omitted multiply/add operations; both corrected and raw counter values are retained in the run records.
- R_time is reported descriptively. The paper does not claim a CPU speedup, and its GPU kernels are not used here.

## Prediction

- Expected R_FLOP at Q: 0.75–1.05.
- Expected R_time at Q: 1.0–1.6.
- Probability the FLOP pass rule is met: 30%.

## Pass rule

The claim holds at this tested scale only if all runs pass static, causality and finite-loss checks, each seed has R_FLOP below 1, and the median R_FLOP is at most 0.752. A bound at Q, failed validity check or missing final run makes the verdict inconclusive. Otherwise it does not hold at this scale.

## Stop rules

- Stop a run on a harness validity failure or non-finite loss.
- A failed run is reported; it is not silently replaced. Any rerun and reason will be recorded.
