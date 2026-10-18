# ECC Charter

Version 1.0, adopted 2026-10-18. This charter is the constitution of the Efficiency Claims Check project (ECC). It says what the project is for, who may take part, how work is judged and how the rules change. Where another document in this repository disagrees with the charter, the charter wins.

## 1. Purpose

ECC is an open project to find out what really makes the training of AI models efficient.

It does this in three ways:

1. **Checking published claims.** Papers often report that a method reaches the same quality with less compute. ECC tests those claims under fair, pre-registered conditions.
2. **Testing new ideas from anyone.** Contributors may propose their own techniques, papers or hypotheses. Every proposal can be tested by anyone, at any time, under the same rules.
3. **Building shared knowledge and standards.** Verified results, practical techniques and measurement rules are collected in the open, so that practitioners and industry can rely on them. Over time, the project aims to turn its best-tested practices and its measurement method into standards others can adopt.

## 2. Principles

Every contribution follows these principles. They are not optional.

1. **Evidence before claims.** A technique is "efficient" only after a check shows it, at a stated scale.
2. **Pre-registration.** Predictions, pass rules and stop rules are made public before the training they judge. Frozen text is never edited; corrections go into the result.
3. **Strong baselines.** Methods are compared against a well tuned modern recipe, never a weak default (see [`METHOD.md`](METHOD.md) §3).
4. **Matched quality, FLOPs and time.** Compute is compared at equal quality, and both FLOPs and measured time are reported.
5. **Negative results count.** A method that does not work is a result, published with the same care as one that does.
6. **Scale honesty.** Every number states the scale it was measured at. Small scale informs a claim; it rarely settles it.
7. **Independent review.** No one approves their own work. Every pre-registration and every result is reviewed by someone other than its author.
8. **Open by default, credit always.** Work is public, and the name of the person who did it stays on it.

## 3. Who can contribute

Anyone, from any background: researchers, engineers, students, independent practitioners, and authors of the papers under test. No affiliation, degree or prior track record is required. The only condition is that the work follows this charter and [`METHOD.md`](METHOD.md).

The use of AI assistants and coding agents is allowed. The contributor remains responsible for the work: it must be correct, and the contributor must understand it well enough to answer questions about it in review.

## 4. Kinds of contribution

| kind | ID | what it is | where it lives |
|---|---|---|---|
| **Claim** | C## | A published claim of more efficient training, with its source | [`claims/`](claims/) |
| **Proposal** | P### | A contributor's own idea, technique or paper, written so that others can test it | [`proposals/`](proposals/) |
| **Check** | ECC-### | A pre-registered test of a claim or a proposal | [`results/`](results/) |
| **Review** | (in the check) | An independent review of a pre-registration or a result | `REVIEW.md` in the check's folder, and the pull request |
| **Technique note** | T### | A practical, efficiency-related training technique, with its evidence level | [`techniques/`](techniques/) |
| **Standard** | S-# | A measurement or reporting rule proposed for wider adoption | [`standards/`](standards/) |
| **Tooling** | (none) | Improvements to the harness, scripts and checks | [`harness/`](harness/), [`tools/`](tools/) |

IDs are assigned by a maintainer when the pull request or issue is opened, so that numbers never collide.

## 5. Evidence levels

Every claim, proposal and technique carries one evidence level. The level only goes up through checks, and it can go down when new checks disagree.

| level | meaning |
|---|---|
| **IDEA** | Proposed, not tested in this project. |
| **PRE-REGISTERED** | A check is frozen and running. |
| **EXPLORATORY** | Tested, but outside the full protocol (for example not frozen publicly, or against a weak baseline). Informative, not conclusive. |
| **CHECKED** | One check followed the full protocol. Its verdict (HOLDS, PARTIAL, DOES NOT HOLD, INCONCLUSIVE) and scale are recorded. |
| **REPLICATED** | Two or more independent checks, by different people, agree at a stated scale. |
| **CONTESTED** | Independent checks disagree. Both results stay visible. |

A technique note may describe practice at any level, but it must state its level and link the checks behind it.

## 6. Review

1. **Pre-registration review.** Before training, at least one reviewer checks that the plan is testable: there is a strong baseline, equal tuning effort on both sides, a pass rule and stop rules. The review is about the design, not about the expected outcome.
2. **Result review.** After the run, at least one reviewer checks that the result follows the pre-registration and the method. Every deviation must be listed in the result.
3. **Independence.** A reviewer may not be the author of the work. Review of a check of one's own proposal is allowed only if a second, unrelated reviewer also approves.
4. **Openness.** Reviews are written in the pull request and summarised in `REVIEW.md`. Disagreement is resolved by discussion in public. If it cannot be resolved, the check is recorded with both views.
5. **What review never does.** It never rejects a result because the outcome is unwelcome, to anyone.

## 7. Roles

- **Contributor:** anyone who opens an issue or a pull request.
- **Reviewer:** a contributor with at least one accepted check or review, or one invited by a maintainer. Reviewers approve pre-registrations and results.
- **Maintainer:** keeps the repository, assigns IDs, merges pull requests that passed review, and guards this charter. DotrixAI provides the first maintainers. Contributors who show sustained, careful work may be invited as maintainers by the existing maintainers.

Roles are earned by work, never bought.

## 8. Integrity

The following lead to a result being withdrawn, with a public note explaining why:

- editing a pre-registration after it was frozen;
- tuning on, or otherwise leaking, the sealed evaluation text;
- hiding compute, for example work done outside the counted region;
- reporting only favourable seeds or runs;
- presenting someone else's work as one's own.

Honest mistakes are not misconduct. They are corrected openly, with credit to whoever found them.

## 9. Credit and licensing

- The proposer of an idea is credited on the proposal and on every check of it.
- The person who runs a check is credited on that check.
- Reviewers are credited on the reviews they write.
- Code is licensed under MIT, and documentation and data under CC BY 4.0 (see [`LICENSE`](LICENSE)). Contributing means agreeing to these licenses for the contribution.

## 10. Relationship with DotrixAI

ECC is stewarded by DotrixAI, an independent AI research lab. The separation between this open project and DotrixAI's confidential work is set out in [`OPEN_AND_CLOSED.md`](OPEN_AND_CLOSED.md). In short:

- Nothing confidential enters this repository.
- Contributors' work is never sold as exclusive.
- DotrixAI's internal research may learn from the public results here, like anyone else.
- DotrixAI's own checks follow the same rules as everyone's.

## 11. Changing this charter

1. Changes to the charter or to [`METHOD.md`](METHOD.md) are proposed as a pull request.
2. The pull request stays open for comments for at least 7 days.
3. A maintainer merges it if the discussion supports it, and records the change in the changelog below.
4. The version number increases with every change.

## 12. Conduct

Be precise and kind. Disagree with numbers, not people. Authors whose claims do not reproduce here are welcome to respond, and a check that does not reproduce a result is not an accusation.

## Changelog

| version | date | change |
|---|---|---|
| 1.0 | 2026-10-18 | First version. The project opens to proposals, technique notes, reviews and standards, in addition to checks of published claims. |
