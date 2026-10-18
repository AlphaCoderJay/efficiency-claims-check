# Training-efficiency gym: CPU prototype (v0.1)

A runnable prototype of the environment in `SPEC.md`, small enough to run on a laptop CPU in minutes.

A submission is one Python file that defines a language model and how to train it. The harness owns everything that decides the score: the data order, the batch shape, the sealed evaluation text, the timing and the checks. The reward is how much less compute the submission needs than the reference recipe to reach the reference's final quality.

## Files

| file | role |
|---|---|
| `gym_harness.py` | Harness and grader: static scan, budget, causality check, training loop with CPU-time and FLOP accounting, sealed evaluation, ratio and reward. |
| `submissions/reference.py` | The reference recipe: a small pre-norm GPT (542K parameters) trained with AdamW for 1,200 updates of 16 x 128 tokens. |
| `submissions/ngram_input.py` | A real lever: hashed 2-gram / 3-gram input embeddings (SparseAdam). |
| `submissions/counted_prior.py` | A real lever with off-counter compute: a counted bigram prior built in `prepare()` with numpy. |
| `submissions/strong_reference.py` | Both levers together, used as a stronger baseline. |
| `submissions/no_causal_mask.py` | Trap: bidirectional attention (sees the target). |
| `submissions/batch_coupled.py` | Trap: centring over the batch couples rows (the failure mode of capacity-limited MoE dispatch). |
| `submissions/reads_files.py` | Trap: tries to read files (for example the evaluation text). |
| `run_demo.py` | Runs everything in one session and writes `demo_results.json` and `DEMO_REPORT.md`. |

## Run it

The harness reads two uint16 token files with vocabulary 2,048: `train.bin` and `valid.bin` (the names `en_train.bin` and `en_validation.bin` also work). The token files are not stored in the repository; build them once from public WikiText-2 with the script and tokenizer in [`dataprep/`](dataprep/):

```bash
pip install torch numpy pyarrow tokenizers
REV=b08601e04326c79dfdd32d625aee71d232d685c3
BASE=https://huggingface.co/datasets/Salesforce/wikitext/resolve/$REV/wikitext-2-raw-v1
curl -L -o train-00000-of-00001.parquet $BASE/train-00000-of-00001.parquet
curl -L -o validation-00000-of-00001.parquet $BASE/validation-00000-of-00001.parquet
python dataprep/prepare_data.py --train train-00000-of-00001.parquet --valid validation-00000-of-00001.parquet --tokenizer dataprep/tokenizer.json --out DATA_DIR
```

`prepare_data.py` checks the SHA-256 of the token files it writes. The expected result is `en_train.bin` with 3,860,300 tokens and `en_validation.bin` with 401,875 tokens. These are the same files used in [ECC-001](../results/ECC-001-dotrixai/).

Then run the demo:

```bash
python run_demo.py --data-dir DATA_DIR --threads 2
```

It takes about 8 minutes on one Intel Core i5-13420H with 2 threads.

## Submission interface

| name | meaning |
|---|---|
| `CONFIG` | Optional dict. `updates` (at most 2,000), `eval_every`, and any keys the functions below read. |
| `prepare(train_tokens)` | Optional. Any preprocessing of the training tokens. Its CPU time is charged like training time. |
| `build_model(vocab, aux)` | Returns a `torch.nn.Module` whose `forward(x[B, T])` returns logits `[B, T, vocab]`. |
| `make_optimizer(model, cfg)` | Returns an object with `step()` and `zero_grad()`. |
| `lr_at(update, cfg)` | Optional learning-rate multiplier. The default is 20 warmup updates, then cosine to 10%. |

## Checks

Any failure gives reward 0.

- **static:** only `torch`, `math`, `numpy` and a few standard modules may be imported; no file, process or reflection built-ins.
- **budget:** at most 2,000 updates and 16M parameters.
- **causality:** change one token in row 0 of a two-row batch. The logits of row 0 before that position, and every logit of row 1, must stay the same (fp32, tolerance 1e-5). The check runs on the initial and on the trained model.
- **finite:** the loss stays finite.

## Accounting and reward

- **C_time:** process CPU seconds of `prepare()` plus every training update.
- **C_flops:** FLOPs of one forward and backward pass (`torch.utils.flop_counter`) times the number of updates. Table builds in numpy appear only in C_time.
- **Q:** the reference's final bits per token on the sealed text.
- **R:** C_submission(Q) / C_reference(Q), with C_submission interpolated linearly in log C between evaluation points. Bound flags mark targets outside the measured range.
- **Reward:** clip(log2(1 / R_time), 0, 2) / 2.
- **Cost-to-quality curve:** R at every quality the reference passes, reported next to the reward.

## Demo results (2026-10-16, 2 threads; full table in `DEMO_REPORT.md`)

| submission | final bits per token | R_time at Q | R_FLOP at Q | reward |
|---|---|---|---|---|
| reference (Q) | 6.656 | 1 | 1 | |
| ngram_input | 5.986 | 0.508 | 0.462 | 0.489 |
| counted_prior | 6.142 | 0.099 (upper bound) | 0.083 (upper bound) | 1.000 |
| strong_reference | 5.637 | 0.103 (upper bound) | 0.083 (upper bound) | 1.000 |
| no_causal_mask | rejected: causality (prefix logits changed by 0.069) | | | 0 |
| batch_coupled | rejected: causality (other row changed by 0.191) | | | 0 |
| reads_files | rejected: static scan (import os, open) | | | 0 |
| ngram_input vs strong_reference | never reached the strong Q | 0.882 (lower bound) | | 0 |
| counted_prior vs strong_reference | never reached the strong Q | 0.927 (lower bound) | | 0 |
| reference rerun | 6.656 (identical) | 1.170 | 1.000 | |

## What the demo teaches the design

1. **The reference must be strong.** Against the plain recipe, a counted bigram prior earns the full reward although its final quality is worse than the n-gram input lever's. Against a reference that already has both levers, each single lever earns 0. The real environment's reference is therefore the strong recipe of the CIR program, not a plain Transformer.
2. **The target quality decides what is rewarded.** The counted prior helps most early; its advantage looks huge at low targets. The n-gram input lever's advantage grows with the target (R from 1.21 down to 0.51). Report the cost-to-quality curve, and set Q from a strong reference near the end of its budget.
3. **CPU time is noisy.** Identical runs differed by 17% in CPU time, the same as a reward of 0.11, while bits were identical. Use the exact FLOP ratio as the primary reward and time as a guard, or repeated same-session anchors on dedicated hardware.
4. **Front-loaded wins need finer early evaluation.** Bounds at the first evaluation point hide the true ratio, so evaluate more often early in training.
5. **Sandboxing is a stub.** The static scan blocks naive file access only. Production needs a container with no network and no access to the evaluation text.
