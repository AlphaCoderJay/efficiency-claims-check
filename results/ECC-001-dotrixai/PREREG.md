# Pre-registration: C12 Normalized Transformer (nGPT) — ECC-001

Frozen at: (the commit time of this file in github.com/dotrixai/efficiency-claims-check, `results/ECC-001-dotrixai/PREREG.md`)
Author: DotrixAI (run by the CIR research session; prepared prepared 2026-10-17T09:28:03Z UTC)

## Claim

- **Claim id and text:** C12. "The normalized Transformer (nGPT) needs 4x to 20x fewer training steps."
- **Source and claimed ratio:** Loshchilov, Hsieh, Sun, Ginsburg, *nGPT: Normalized Transformer with Representation Learning on the Hypersphere*, [arXiv 2410.01131](https://arxiv.org/abs/2410.01131) (2024). The claim is stated in training steps (4x to 20x fewer, growing with context length). Claimed ratio in `claims/claims.csv`: 0.05 to 0.25. This check measures compute (FLOPs and CPU time) to the same quality, which includes nGPT's extra work per step.

## Setup

- **Model and size:** both sides use the same shapes. d 96, 3 layers, 4 heads (head dim 24), SwiGLU MLP with hidden size 256, RoPE, separate (untied) input and output embeddings, vocabulary 2,048. Parameters: baseline 726,336, nGPT 729,224 (the 2,888 extra are nGPT's scaling vectors). FLOPs per update counted by the harness: 6.49 GFLOP for both (matrix multiplications; nGPT's normalisations add under 1%, ESTIMATED, and are visible in CPU time).
  - **Baseline (modern small GPT):** pre-LayerNorm blocks, AdamW (betas 0.9 / 0.95, weight decay 0.1 on matrices except the input embedding), harness default schedule (20-update warmup, cosine to 10%). Code: `submissions/baseline.py`.
  - **Method (nGPT, from the paper's equations):**
    - hidden states on the unit sphere: h ← Norm(h + α_A (Norm(h_A) − h)), and the same with α_M after the MLP;
    - attention on normalised q, k scaled by s_qk, with softmax scale √d_k;
    - SwiGLU with u · s_u and ν · s_ν · √d;
    - logits s_z · (E_out h);
    - no LayerNorm, no weight decay, no warmup (cosine to 10%);
    - all matrices renormalised along the embedding dimension after every optimiser step;
    - scaling vectors use the paper's init / scale restoration (α init 0.05, scale 1/√d; s_qk init 1, scale 1/√d; s_u, s_ν init 1, scale 1; s_z init 1, scale 1/√d).

    Code: `submissions/ngpt.py`.
- **Data and tokenizer:** WikiText-2 (public). Training text: the train split. Evaluation text: the validation split. Tokenized with a byte-level BPE of 2,048 tokens. That tokenizer was trained on the WikiText-2 train split, an Indonesian Wikipedia sample and TinyStories train, never on validation text. It is published with this check: `tokenizer.json` (SHA-256 ee98f08a2af1f3ef075394ca7f4197dfced91a34ed468cd891dc231c5ae5b6ed) and `prepare_data.py`. The script rebuilds the token files from the public Hugging Face dataset Salesforce/wikitext (revision b08601e0, config wikitext-2-raw-v1): strip each row, drop empty rows, drop validation rows identical to a train row, join with newlines. Expected token files: train 3,860,300 tokens (SHA-256 dba66a84…), validation 401,875 tokens (SHA-256 5d50fda5…), both reproduced on 2026-10-17.
- **Context length, batch size, updates:** context 128, batch 16 rows (the harness's fixed data order per seed), 1,200 updates for both sides. Evaluation every 50 updates.
- **Harness:** `harness/gym_harness.py` of this repository at commit f9beaba (SHA-256 8fc0eb87…), unchanged. It provides the data order, causality checks, FLOP counting (one forward and backward, `torch.utils.flop_counter`) and CPU-time accounting. The runner `run_ecc001.py` only chooses the submission, its learning rate, the seed and the validation windows that are scored.
- **Tuning (same effort for both sides):**
  - Learning-rate grid {2e-3, 4e-3, 8e-3, 1.6e-2} for each side, seed 11, scored on validation windows 256-511 (the tuning windows). The 256 windows used for Q (windows 0-255) are never used for tuning.
  - If the best value sits at a grid edge, the grid is extended by one doubling step on that side, for that recipe only. This is allowed at most twice per side and is reported.
  - No other hyperparameter is tuned on either side.
- **Hardware and threads:** one Intel Core i5-13420H laptop, 2 threads, on AC power, nothing else running. Runs go one at a time, baseline and nGPT alternating order between seeds.
- **Seeds:** 11, 22 and 33. Each seed sets both the data order and the initialisation. Each seed gives one baseline run and one nGPT run at the tuned learning rates.
- **Pilot runs before freezing:** One pilot per recipe, 40 updates each (learning rate 4e-3, seed 11), run 2026-10-17 09:24-09:25 UTC. Both passed the harness static scan and the causality checks on the initial and trained models (maximum change 0.0). FLOPs per update were equal (6.49 GFLOP). CPU time per update was 0.55 s (baseline) and 0.71 s (nGPT, +29%), from per-step normalisations the FLOP counter does not see. The pilot losses are not used for any decision. Expected full budget: 8 tuning runs (about 1.7 h) and 6 final runs (about 1.3 h), one at a time.

## Target quality

- **How Q is chosen:** for each seed, Q = the baseline's bits per token at its last evaluation point (update 1,200) on the sealed windows. R(Q) uses the nGPT run of the same seed. The harness also reports R at every baseline evaluation point (the cost-to-quality curve).
- **Evaluation text:** WikiText-2 validation, the first 256 non-overlapping windows of 129 tokens (the harness's sealed evaluation). Every target is scored.

## Prediction

- **Expected R_FLOP at Q:** median over seeds 0.75, 80% interval [0.40, 1.20].
- **Expected R_time at Q:** median 0.85, 80% interval [0.45, 1.40]. nGPT's per-step normalisations add CPU time that the FLOP counter does not see.
- **Probabilities by verdict:** HOLDS 0.05, PARTIAL 0.45, DOES NOT HOLD 0.40, INCONCLUSIVE 0.10.
- **Reasoning, written before training:** the paper reports its largest gains at long context (up to 4k) and at 0.5B-1B parameters; at context 128 and under 1M parameters the expected gain is smaller. Matrix-optimizer and normalisation gains reported at scale often shrink against a tuned baseline (C02).

## Pass rule

The verdict uses R_FLOP at Q, computed per seed by the harness's `ratio` function (log-linear interpolation; a run that never reaches Q gives a lower bound).

- **HOLDS:** median R_FLOP ≤ 0.25 and median R_time ≤ 0.35.
- **PARTIAL:** median R_FLOP in (0.25, 0.8], or median R_FLOP ≤ 0.25 with median R_time above 0.35.
- **DOES NOT HOLD:** median R_FLOP above 0.8, or lower bounds above 0.8 because nGPT never reaches Q.
- **INCONCLUSIVE:** a validity check fails (causality, finite loss, harness scan), or the three seeds fall in different R_FLOP classes (≤ 0.25 / ≤ 0.8 / above).

The verdict is computed by `run_ecc001.py report`.

## Stop rules

- **A run is ended early if:** the harness rejects it (static scan, causality on the initial or trained model, non-finite loss, budget). Nothing else ends a run early. Quality is never used to stop.
- **What an early stop means:** a rejected nGPT run makes that seed INCONCLUSIVE, and the result says why. A divergence at one learning rate during tuning removes that learning rate from the grid; it is reported, not retried.
- **Data or code changes after the freeze:** none. Any correction goes into RESULT.md, never into this file.

## Separation (DotrixAI)

This check uses public data and published code only. It is independent of DotrixAI's confidential research records: no internal results, values or predictions from them are used here.
