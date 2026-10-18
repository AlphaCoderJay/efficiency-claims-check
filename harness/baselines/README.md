# Standard baselines

[`METHOD.md`](../../METHOD.md) §3 requires a strong baseline. To make checks comparable, ECC keeps standard baselines here. A check uses one of them unless its pre-registration argues for a different one, and that argument is reviewed.

| baseline | file | what it is | used by |
|---|---|---|---|
| **ECC baseline v1** | [`ecc_baseline_v1.py`](ecc_baseline_v1.py) | A small modern GPT: pre-LayerNorm, RoPE, SwiGLU MLP, untied embeddings, and AdamW with weight decay 0.1. d 96, 3 layers, 4 heads, MLP hidden 256 (about 0.73M parameters at vocabulary 2,048). Harness default schedule. The learning rate is tuned per check. | [ECC-001](../../results/ECC-001-dotrixai/) |

**Not a standard baseline:**
- `harness/submissions/reference.py` is the harness's simple demo recipe. It is kept for the demo, but it is too weak for a check. [ECC-002](../../results/ECC-002-shikhar/) used it and is therefore exploratory.
- `harness/submissions/strong_reference.py` adds two lookup levers for the demo. It is not a general baseline.

**Tuning rule:** tune the learning rate of the baseline with the same grid and the same effort as the method under test. A baseline v1 run reaches about 5.8 bits per token on WikiText-2 validation after 1,200 updates (ECC-001, three seeds), which is a useful sanity check for a new setup.

New standard baselines, for example for larger models or GPUs, are added through a pull request under [`CHARTER.md`](../../CHARTER.md) §11.
