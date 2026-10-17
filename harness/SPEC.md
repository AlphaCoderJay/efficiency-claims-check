# Training-efficiency gym: environment specification (v1)

**Status:** specification plus a CPU prototype. The prototype (this folder, 2026-10-16) implements the grader loop on a laptop-scale task: static scan, causality check, sealed evaluation, CPU-time and FLOP accounting, the matched-quality ratio and the reward. Its demo results and the design changes they forced are in §9. The full environment on the 20M / 85M GPU testbed, in a container, does not exist yet. Unless §9 says otherwise, numbers below are not runs of this environment.

## 1. Purpose

An agent improves the training recipe of a small language model so that it reaches a fixed quality with less compute. Evaluation is sealed and the accounting is honest. The reward is the measured compute ratio at matched quality against a strong reference recipe. This is the task the CIR research program performed for weeks, turned into episodes with a verifiable reward.

It measures research and engineering skill that existing benchmarks cover only partly:

- forming hypotheses about training efficiency;
- choosing cheap discriminating probes;
- implementing changes correctly (causal, numerically sound, memory-safe);
- not fooling yourself: weak baselines, early-curve artefacts, hidden costs.

## 2. Episode

| part | content |
|---|---|
| Start state | Repository with a strong published reference recipe at the 20M size (model, Muon + AdamW optimiser, trainer with WSD schedule and decay branches, evaluator). Pre-tokenised training data (one epoch, 380M tokens, English + Indonesian). The reference learning curve. A README of the rules. |
| Exploration budget | A fixed GPU budget for probe runs, for example 2 to 4 A100-hours, spent through a tool `run(config, updates)`. The tool returns the training-loss curve, ms per update and peak memory, never evaluation scores. |
| Actions | Read and edit code and configuration; launch budgeted probes; read probe logs; submit. |
| Submission | A code diff plus configuration, and a short claim (expected R and why). |
| End | Submission, or budget exhausted (the last valid recipe is graded). |

## 3. Grader (sealed harness)

1. **Static checks.** The diff applies. Evaluation text is never mounted in the agent's container (held-out sets are hashed). No network. The tokenizer and training data are fixed in the core track.
2. **Causality test.** Perturb token *t* in row *r* of a batch. The logits of every other row, and of positions before *t* in row *r*, must not change (fp32, tolerance 1e-5). This catches lookahead and batch coupling; a capacity-limited MoE fails it.
3. **Memory and stability.** The full run must fit the grader GPU and finish without NaN or inf.
4. **Full training run** of the submitted recipe on grader hardware: evaluation points at 0.5U and 1.0U, plus a stop point if the time limit is hit. Bits per byte on the five held-out sets. BLiMP guard.
5. **Accounting.**
   - C_time: GPU milliseconds measured with CUDA events, including auxiliary GPU work such as table builds and feature precomputation.
   - CPU preprocessing: counted separately under a stated conversion rule.
   - C_flops: from an analytic counter on the submitted model, cross-checked with a profiler.
   - Same-session anchor: a short anchor run of the reference recipe normalises hardware and session drift.
6. **Reward.** R_FLOP and R_time at Q_b (the strong reference's 1.0U quality per set, geometric mean of the four core sets), with the estimator described in this section.
   - Default reward: `clip(log2(1 / R_FLOP), 0, 2) / 2`. A reward of 1.0 means 4x fewer FLOPs. R_FLOP is exact; measured time was too noisy on the prototype hardware (§9).
   - Measured time on dedicated GPUs with same-session anchors acts as a guard. A time ratio clearly worse than the FLOP ratio flags hidden cost: off-counter preprocessing or slow kernels.
   - The reward is zero if any check fails, or if BLiMP drops beyond its measured noise (about 3 points per run; paired test).
   - A variant averages R over the upper half of the cost-to-quality curve, so that front-loaded levers cannot win on one low target.
7. **Report.** Per-set ratios with MEASURED or ESTIMATED labels, the measured curve, and the checks that passed or failed.

## 4. Difficulty: what is already known on this testbed

Public conclusions of the CIR program at 20M and 85M against a strong optimized Transformer recipe (details in github.com/dotrixai/cir-research):

- Cheap published levers on the 20M recipe (Canon layers, gated attention, sequence-length warmup) gave small FLOP gains and no wall-clock gain.
- A gated delta hybrid (20M, 85M) and a mixture of experts (85M) also gave small FLOP gains, far from a ratio of 0.75, and none saved wall-clock time with the kernels available. The hybrid's edge shrank from 20M to 85M.

Exact per-candidate ratios are withheld because they are answer keys of a separate task set.

## 5. Known traps (the hidden tests come from these)

- Batch coupling from capacity-limited expert dispatch (fails the causality test).
- Statistics built from text that includes evaluation data. The evaluation text is not available inside the container.
- Compute shifted outside the timed region (precomputation, CPU-side tables). The harness times the whole process.
- Compile or fused-kernel numerics. The harness compares against eager on fixed inputs before the full run.
- Deadline breaks without an evaluation. The harness always evaluates at the stop point.
- Memory blow-ups from larger tables or more experts. The harness reports OOM as a failed episode.
- Decisions from early learning-curve gaps that fade by 0.2U. Agents see their own probe curves and must avoid this.

## 6. Cost per episode (ESTIMATED)

| item | estimate |
|---|---|
| One 20M grading run (1.0U plus evaluation) | about 0.6 A100-hours (the reference trains 1.0U in about 30 minutes) |
| Exploration budget | 2 to 4 A100-hours, configurable |
| Episode total | about 3 to 5 A100-hours |
| 85M track, one grading run | about 1.7 A100-hours |

## 7. Variants

- **85M track:** same rules at d 768, 12 layers; this is where a fused-kernel hybrid and MoE behave differently.
- **Data-efficiency track:** the number of unique tokens is fixed and repeated epochs are allowed.
- **Deployment track (secondary):** int8 CPU decode speed and RAM of the trained model.
- **Audit track:** the agent receives a run log and a claim and must find the flaw. These are the audit tasks of this sample, generated from more episodes.

## 8. What exists and what must be built

**Exists (CIR research code):**

- trainer, models and evaluator;
- R estimator and analysis scripts;
- FLOP counters;
- a perturbation causality test;
- reference curves at 20M and 85M;
- about ten measured candidate arms with logs;
- cloud orchestration with budget guards, deadlines and watchdogs.

**To build:**

- a container image with pinned PyTorch and CUDA;
- the sealed grader service and the budgeted `run` tool;
- the causality, accounting and memory checks as one gate;
- a public / private task split;
- a red-team pass against reward hacking;
- calibration runs: a reference rerun per hardware type, plus pilot episodes with two or three agents.

**Rough effort:** 2 to 3 weeks of engineering plus 20 to 40 A100-hours for calibration (ESTIMATED).

## 9. Prototype results and the design changes they forced (2026-10-16)

The CPU prototype (this folder) runs the grader loop on a 542K-parameter GPT trained on WikiText-2 (BPE 2,048) for 1,200 updates on a laptop CPU. Full numbers are in `DEMO_REPORT.md`.

**What worked**

- The causality check rejected bidirectional attention (prefix logits changed by 0.069) and batch-centred embeddings (the other row changed by 0.191) before any training.
- The static scan rejected file access.
- A real lever (hashed n-gram input embeddings) earned a measured, interpolated ratio: R_time 0.51, R_FLOP 0.46, reward 0.49.

**What changed in the design**

1. **Reference strength.** A counted bigram prior earned the full reward against the plain reference, although its final quality was worse than the n-gram lever's. Against a reference that already contains both levers, each single lever earned 0. The production reference is therefore the strong recipe (the strong published recipe at 20M and its dense 85M counterpart), and every reference must include the cheap published levers.
2. **Quality target.** The cost-to-quality curve shows a front-loaded lever (large advantage at low targets) next to a lever whose advantage grows with the target (1.21 down to 0.51). Production sets Q near the end of the strong reference's budget and reports the whole curve. A variant reward averages R over the upper half of the curve.
3. **Primary unit.** On the laptop, identical runs differed by 17% in CPU time, the same as a reward of 0.11, while bits were identical. Production uses the exact FLOP ratio as the primary reward. Measured time on dedicated GPUs with same-session anchors acts as a guard: a submission whose time ratio is clearly worse than its FLOP ratio is flagged for hidden cost.
4. **Evaluation grid.** Front-loaded submissions hit bounds at their first evaluation point, so production evaluates more often early in training.
5. **Isolation.** The static scan is a stub. Production runs submissions in a container without network access and without the evaluation text, and evaluates in a separate process.
