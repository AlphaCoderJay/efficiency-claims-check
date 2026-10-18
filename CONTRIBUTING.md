# Contributing

Anyone can contribute, from any background. Read the [charter](CHARTER.md) first: it is short, and every contribution follows it. Discussion happens in GitHub issues and pull requests, in the open.

Using AI assistants or coding agents is fine. You stay responsible for the work and should be able to explain it in review.

## Ways to contribute

### 1. Propose an idea, technique or paper

You have a technique you believe trains models more efficiently, or a paper of your own. Make it testable so others can check it.

1. Open an issue with the **Proposal** template. A maintainer assigns an ID (P###).
2. Copy [`templates/PROPOSAL.md`](templates/PROPOSAL.md) to `proposals/P###-<short-name>/PROPOSAL.md`, fill it in, add code if you have it, and open a pull request.
3. A reviewer checks that it is clear and testable. Once merged, its evidence level is IDEA, and anyone may check it.

### 2. Run a check (of a claim or a proposal)

1. Pick a claim from [`claims/CLAIMS.md`](claims/CLAIMS.md) or a proposal from [`proposals/`](proposals/). Open an issue with the **Check** template, so others know. A maintainer assigns an ID (ECC-###). Several people may check the same subject at different scales.
2. **Pre-register.** Copy [`templates/PREREG.md`](templates/PREREG.md) to `results/ECC-###-<your-handle>/PREREG.md` and open a pull request with only that file. Use a [standard baseline](harness/baselines/) unless you argue otherwise.
3. **Pre-registration review.** A reviewer checks the design. Once merged, the commit time is the freeze time. Do not start the full runs before that.
4. **Run.** Use the CPU harness in [`harness/`](harness/) or any setup that follows [`METHOD.md`](METHOD.md). Keep all logs and all seeds.
5. **Report.** Add `RESULT.md` from [`templates/RESULT.md`](templates/RESULT.md), your code and the logs to the same folder, and open a second pull request.
6. **Result review.** A reviewer checks the result against the pre-registration and the method, and writes `REVIEW.md`.

A run that skipped part of this process can still be published, labelled EXPLORATORY.

### 3. Review someone's work

Reviews are as valuable as checks. Pick an open pull request with a pre-registration or a result, and review it with [`templates/REVIEW.md`](templates/REVIEW.md). You may not review your own work. After one accepted check or review you can be listed as a reviewer ([charter](CHARTER.md) §7).

### 4. Share a technique

A practical trick for training with less compute, or a pitfall to avoid. Open an issue with the **Technique note** template, then add `techniques/T###-<short-name>.md` from [`templates/TECHNIQUE.md`](templates/TECHNIQUE.md). State its evidence level honestly; IDEA is fine.

### 5. Add or correct a claim

A published claim that is missing, a misread number, a better source, or an independent check we missed. Open an issue with the **Claim** template, or a pull request that edits [`claims/claims.csv`](claims/claims.csv) and reruns `python tools/make_claims_md.py`.

### 6. Improve the tools or the standards

Harness accounting and checks, new standard baselines, scripts, or comments on the [standards](standards/). Changes to the charter, the method or a standard follow [charter](CHARTER.md) §11: a pull request open for at least 7 days.

## Conduct

Be precise and kind. Disagree with numbers, not people. Authors of the work under test are welcome to respond, and a check that does not reproduce their result is not an accusation.

## Credit

Proposers, the people who run checks, and reviewers are all credited by name on the work. Code is published under MIT, and documentation and data under CC BY 4.0.
