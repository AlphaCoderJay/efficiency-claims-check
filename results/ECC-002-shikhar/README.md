# ECC-002: C13 dynamic short convolutions (exploratory)

**Run by:** Shikhar, contributor.
**Reviewed by:** DotrixAI, 2026-10-18.
**Status:** EXPLORATORY. This is not an official check, and C13 stays OPEN in the claim list.

The files in this folder are published as the contributor sent them. `PREREG.md`, `RESULT.md`, `run_c13.py`, `submissions/dynamic_qkv.py` and `logs/` are unedited, apart from line endings, which git normalises to LF.

## What the run found

At 0.54M (baseline) and 0.58M (method) parameters, context 128, 1,200 updates on WikiText-2, seeds 11, 22 and 33, on a laptop CPU:

- **R_FLOP:** median 0.556 (0.642, 0.505, 0.556). The method is about 1.8x cheaper in FLOPs at matched quality.
- **CPU time:** median R_time 0.974 (1.160, 0.920, 0.974), so there is no clear time saving.
- **FLOP accounting:** FLOPs combine the harness count with an analytic estimate for pointwise operations that the counter omits. This is stated in `PREREG.md`.
- **Quality:** the numbers in `RESULT.md` match `logs/final.json`.
- **Checks:** all causality, static and finite-loss checks passed.

## Why it is exploratory

1. **The pre-registration was not frozen publicly.** `RESULT.md` cites commit `e9f8251`, which is not in this repository. The freeze time therefore cannot be checked. Official checks submit `PREREG.md` as a pull request before any full run (see `CONTRIBUTING.md`).
2. **The baseline is the harness's simple reference GPT** (`harness/submissions/reference.py`), not a modern tuned recipe. The method's final quality, about 6.2 to 6.3 bits per token, is worse than the stronger baseline in [ECC-001](../ECC-001-dotrixai/), which reaches about 5.8 at a similar size. `METHOD.md` §3 asks for a strong baseline, so this result shows a gain over a weak baseline only.
3. **Tuning was incomplete for the method.** `PREREG.md` and `run_c13.py` define a grid of four learning rates for both recipes, but `logs/tune.json` holds only 0.002 and 0.004 for the method. Untested higher rates could only have helped the method, so this does not bias the result in its favour. It is still a deviation from the pre-registration.
4. **Author line:** `PREREG.md` lists "DotrixAI" as author. The work was done by Shikhar.

## How to make C13 official

Rerun against the ECC-001 baseline (`results/ECC-001-dotrixai/submissions/baseline.py`) with the full learning-rate grid for both sides. Submit the pre-registration as a pull request before training. The data steps are in `harness/README.md`.
