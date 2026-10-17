# ECC-001: additions after the freeze

The PREREG was frozen at commit 29623b07 (2026-10-17T10:25:32Z). Nothing below changes a frozen rule, the grid, the seeds, the recipes, the target quality or the verdict classes. Each item is dated, and each was written before the run it governs.

## 1. Grid extension script (written 2026-10-17, before tuning ended)

`ecc001_extend.py` carries out the frozen rule "If the best value sits at a grid edge, the grid is extended by one doubling step on that side, for that recipe only. This is allowed at most twice per side and is reported." It calls `run_ecc001.one()` unchanged and appends the extra runs to `logs/tune.json` under `extensions`.

Tuning ended 2026-10-17T11:29:08Z with both best rates on a grid edge:
- baseline best 2e-3 (lowest);
- nGPT best 1.6e-2 (highest).

The rule therefore applies to both recipes.

## 2. Machine-load gate and load record for the finals (written 2026-10-17T11:36Z, before any final run)

**What was seen in tuning.** Within the same code, CPU time per 50 updates rose sharply from about 10:34 to 11:29 UTC:
- baseline at 1.6e-2: about 11 s per 50 updates, then 50-134 s per 50 updates from update 850;
- nGPT: 34.7 s per 50 updates at every rate.

A quiet re-check at 11:32 UTC (`logs/diag_denormal.json`, diagnostic only) showed:
- nGPT at 0.16 s per update;
- the baseline at 1.6e-2 with no late slowdown;
- subnormal floats too few to explain it (at most 576 optimiser-state values).

So outside load on the laptop inflated process CPU time. With two threads, a waiting thread spins and its CPU time is counted. (This explanation turned out wrong; see §3.) Tuning used only quality, which load does not change, so the tuned rates stand. R_time, however, is a ratio of CPU times.

**Added for the finals:**
- **Gate.** The finals start only after the whole machine averages below 10% CPU over 60 s (`load_monitor.py gate`).
- **Record.** Every 15 s, `load_monitor.py watch` writes to `logs/load_final.jsonl`:
  - machine CPU %;
  - the runner's share of the machine;
  - the outside share;
  - each minute, the top other processes.
- **Timing flag.** A final run's time is flagged OUTSIDE_LOAD when either holds:
  - outside load averages above 10% of the machine during the run;
  - its slowest 50-update block takes more than 2x its median block.

  A flag does not change R_FLOP or any quality number.
- **Re-timing.** A flagged pair is re-timed only if the flag could change the verdict class. That is possible only when the median R_FLOP is at most 0.25, where HOLDS needs R_time at most 0.35. The re-timing:
  - happens once, after the six final runs, with identical settings;
  - must reproduce the quality points to 1e-5 bits;
  - reports both timings.
- **Third flag condition (added 2026-10-17T12:14Z, before any final run).** The grid-extension runs (11:47-12:11Z) were slow again, at 25-58 s per 50 updates. At those times the whole machine was at about 9% CPU. Machine load alone therefore does not catch the slow state; it may come from how the OS schedules a detached job while other apps are in front, though that is not confirmed. So a final run is also flagged OUTSIDE_LOAD when its median 50-update block exceeds 1.5x the quiet reference for its recipe. The quiet references, at the tuned rates, are:
  - baseline: 6.66 s (tuning run at 2e-3, 10:27Z);
  - nGPT: 8.25 s (`logs/diag_denormal.json`, 11:32Z, 20-update blocks scaled to 50).

  RESULT.md also reports an ESTIMATED quiet-machine R_time: R_FLOP x (nGPT / baseline quiet CPU per update). The verdict still uses the measured R_time, as frozen.

## 3. Cause of the slow state (written 2026-10-17T12:54Z during the finals, completed 13:40Z; explanation only, no rule change)

The explanation in §2 ("outside load") was wrong.
- During the finals the load record shows the rest of the machine mostly idle: outside load averaged 5.4% and peaked at 12.8% over the first 136 samples.
- The runs were still about 4x slower than the quiet references.
- Per-core samples at 12:54Z (`logs/core_sample.json`) show the work on logical CPUs 8-11, the four efficiency cores of the i5-13420H, while the performance cores (0-7) stayed nearly idle. The runner had normal priority and could use all 12 logical CPUs.

The runs are started detached (WMI, no window), and Windows scheduled them on the efficiency cores. This is a known behaviour of this laptop. The owner's other experiments here opt out per process: power throttling off and affinity to the performance cores. The ECC-001 runner is the published code and has no such opt-out, so it ran on the efficiency cores.

The quiet references are fast runs on the performance cores:
- nGPT: a run started from an interactive shell;
- baseline: the first tuning run, which was fast for a reason not identified, most likely a scheduler decision.

Both recipes ran in the same process and the same state, and they are slowed by about the same factor. The measured R_time is therefore a CPU-time ratio on the efficiency cores, and the quiet estimate is the ratio on the performance cores. The flag is renamed SLOW_STATE in RESULT.md; its conditions are unchanged.

Nothing was changed in how the runs are launched. The docstring of `load_monitor.py` still gives the earlier "outside load" explanation; the file is left as it ran.
