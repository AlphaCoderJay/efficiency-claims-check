# Efficiency Claims Check

An open project to check published claims about cheaper language model training, honestly and in public.

Many papers report that a new optimizer, architecture, schedule or data trick reaches the same quality with less compute. Some of these claims hold. Some shrink against a well tuned baseline or at a larger scale. Some measure FLOPs but not wall clock time. Practitioners rarely know which is which before they spend their own compute.

This project keeps a list of such claims and invites anyone to test one at small scale, under rules written before the run:

1. **Strong baseline.** Compare against a well tuned modern recipe, not a weak default.
2. **Matched quality.** Report the compute each side needs to reach the same quality, not the loss at one step.
3. **FLOPs and time.** Report both. A FLOP saving that does not survive in measured time is reported as such.
4. **Pre-registration.** Write the prediction and the pass rule before training, and do not edit them afterwards.
5. **Negative results count.** A claim that does not hold at small scale is a result, published with the same care.

Small scale cannot settle a claim made at a billion parameters. It can show where a claim depends on a weak baseline, a particular scale or an accounting choice. Every result states its scale next to its numbers.

## What is here

| path | what it is |
|---|---|
| [`claims/CLAIMS.md`](claims/CLAIMS.md) | The claims to check, with sources, claimed ratios and what independent evidence we found. |
| [`claims/claims.csv`](claims/claims.csv) | The same list as data. |
| [`METHOD.md`](METHOD.md) | How a check is measured and reported. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to take a claim, pre-register, run and submit. |
| [`templates/`](templates/) | Pre-registration and result templates. |
| [`harness/`](harness/) | A CPU harness that trains a small language model under fixed rules and measures the compute ratio at matched quality. |
| [`results/`](results/) | Finished checks, one folder each. Empty so far. |
| [`OPEN_AND_CLOSED.md`](OPEN_AND_CLOSED.md) | What is open here, and what DotrixAI keeps separate. |

## The harness

`harness/` runs on a laptop CPU in about 8 minutes. A submission is one Python file that defines a model, an optimizer and a learning rate schedule. The harness owns the data order, the hidden evaluation text, the timing and the checks. Before any training it rejects code that looks at future tokens, couples rows of a batch, or reads files. It reports the compute ratio at matched quality, in FLOPs and in CPU time. See [`harness/README.md`](harness/README.md).

The CPU harness fits optimizer, schedule and small architecture claims. Claims about data or large memories may need other setups, and a check may use any setup that follows [`METHOD.md`](METHOD.md).

## Who runs this

The project is started by [DotrixAI](https://dotrixai.com), an independent AI research lab. Our own research program, CIR, tried to reach a strong optimized Transformer's quality with less training compute and did not succeed. We learned how easy it is to fool yourself with a weak baseline, a single scale or FLOPs read as time. Our public record, failures included, is at [dotrixai.com/cir](https://dotrixai.com/cir) and [github.com/dotrixai/cir-research](https://github.com/dotrixai/cir-research).

We maintain the list and review submissions. Credit for every check goes to the people who ran it.

## Status

Draft. The claim list is a first reading of the literature and will have errors. Corrections are welcome as issues.

## License

Code: MIT. Documentation and data: CC BY 4.0. See [`LICENSE`](LICENSE).
