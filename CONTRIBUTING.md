# Contributing

Anyone can check a claim. Small, careful checks are more useful than large, loose ones.

## Steps

1. **Pick a claim** from [`claims/CLAIMS.md`](claims/CLAIMS.md). Open an issue titled `Check C07: <your setup>` so others know it is taken. Several people may check the same claim at different scales.
2. **Pre-register.** Copy [`templates/PREREG.md`](templates/PREREG.md) to `results/C07-<your-handle>/PREREG.md`, fill it in, and open a pull request with only that file. Once merged, its commit time is the freeze time.
3. **Run.** Use the CPU harness in [`harness/`](harness/) or any setup that follows [`METHOD.md`](METHOD.md). Keep logs.
4. **Report.** Add `RESULT.md` from [`templates/RESULT.md`](templates/RESULT.md), your code and the logs to the same folder, and open a second pull request.
5. **Review.** A maintainer checks that the result follows the method and the pre-registration. Review is about procedure, not about whether the outcome is welcome.

## Also welcome

- Corrections to the claim list: a misread number, a missing independent check, a better source.
- New claims, with a link to the primary source.
- Improvements to the harness, especially accounting and checks.

## Conduct

Be precise and kind. Disagree with numbers, not people. Authors of the original claims are welcome here, and a check that does not reproduce their result is not an accusation.

## Credit

Each result folder lists who ran it. Results are published under CC BY 4.0 and code under MIT, with your name on your work.
