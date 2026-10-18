# Technique notes

Practical techniques for training models with less compute, shared by contributors. Examples: a learning rate schedule that works well at small scale, a data preparation trick, an implementation detail that saves time, or a measurement pitfall to avoid.

Every note states its **evidence level** ([`CHARTER.md`](../CHARTER.md) §5) and links the checks behind it. A note at level IDEA is welcome; it simply says so. A note that claims a saving should become a proposal or a check, so that the saving can be tested.

## How to submit

1. Open an issue with the **Technique note** template. A maintainer assigns an ID (T001, T002 and so on).
2. Copy [`../templates/TECHNIQUE.md`](../templates/TECHNIQUE.md) to `techniques/T###-<short-name>.md` and open a pull request.

## Index

| ID | technique | by | level |
|---|---|---|---|
| [T001](T001-early-gains-fade.md) | Early gains fade: judge efficiency at the end of the budget | DotrixAI | CHECKED (ECC-001) |
