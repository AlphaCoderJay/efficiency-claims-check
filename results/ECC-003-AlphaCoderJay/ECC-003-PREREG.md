# Pre-registration: C__ <short title>

Frozen at: To be recorded when the preregistration is committed
Author: AlphaCoderJay

## Claim

- Claim id and text: C27 - Successive halving can find good hyperparameter configurations faster by a wide margin compared to random search under a comparable compute budget.
- Source and claimed ratio:  Jamieson & Talwalkar (2015), Non-stochastic Best Arm Identification and Hyperparameter Optimization; Li et al. (2018), Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.

- Claim being tested: Whether successive halving requires a lot less search compute than random search to find a configuration reaching a common specified target quaility.

- Scope: This experiment tests the claim at the ECC baseline v1 scale. It does not establish that this result can be the same to other model sizes, datasets or search spaces. This is only for the ECC baseline v1.

## Setup

- Model and size: ECC baseline v1, a GPT styled Transformer Architecture with a width of 96,3 layers, 4 attention heads and MLP hidden dimension 256; approximately 0.73M parameters at a vocabulary size of 2,048. Search configs will vary model width and depth with the preregistered search space, subject to the harness parameter limit.
- Data and tokenizer: WikiText-2, It uses a custom tokenizer.
- Context length: 128.
- Batch size: 16 rows, as said in the harness
- Full training budget: 1,200 updates per configuration
- Baseline recipe: harness/baselines/ecc_baseline_v1.py . Tune the baseline learning rate using the same learning rate range and selection protocol. Use the baseline's final score after 1,200 updates to define the target quality.
- Methods: Random search and successive halving over the same hyperparameter space. Account for actual training FLOPs, including partial runs, rather than assume equal updates mean equal compute.
- Hardware and threads: Record CPU model, thread count, operating system, Python/Pytorch version, and relevant hardware and software settings. Keep conditions as consistent as possible so we can have fair runs.
- Seeds: Baseline seed: 3000. Search-repeat seeds: 3003, 3004, and 3005, paired between the two methods. Final evaluation seeds: 3101, 3102, and 3103 for each selected configuration. Record any changes before the relevant experiment begins.
- Pilot runs before freezing (how many, what they showed): Record the number of pilots, their purpose, and what they showed. Pilots may verify that the harness runs correctly; do not use sealed evaluation results to select configurations 

### Search Space
-------------------------
 - Use the same space for the methods

 - Learning rate: 1e-3 - 3e-2
 - Model width: Choice from {64, 96, 128 , 160}
 - Depth: Choice from {2, 3 , 4 ,6}
 - Warmup updates: Choice from {0, 20, 50, 100}
 - Lr-Schedule: Choice from {cosine decay to 10% of the initial rate, constant with cooldown}.
 - Weight decay: Choice from {0, 0.05, 0.1}
 - AdamW beta2: Choice from {0.95, 0.99}
 - Parameter limit: 2M parameters per configuration.


### Search Protocol
----------------------------------
  #### Random search
  - Sample 16 configs from the search space.
  - Train each config for up to 1,200 updates.
  - Evaluate on the selection-validation at checkpoints.
  - Track cumulative training FLOPs until a config first reaches target quality Q, or until the search budget has been used up.
  - If no config reaches the target quality, record each search as a failure to reach the target in its budget.


  ### Successive halving
  - Start with 32 configs.
  - Use a reduction factor eta = 2.
  - Training budgets per rung: 150, 300, 600 and 1,200 updates.
  - At each rung, evaluate all surviving configs on the selection-validation windows and retain the top half, ranked pregistered quality metric
  - Record every configs, its score, cumulative FLOP's, and elimination decision
  - Count all compute spent on eliminated configs toward the search budget
  - The intended budget is 16 baseline-equivalent full runs per search repeat
    
## Target quality
- Target Q: The ECC baseline v1's final bits-per-token score after 1,200 updates.
- Selection-validation data: WikiText-2 validation windows 256-511.
- Selection-evaluation data: WikiText-2 validation windows 0-255.
- Metric direction: Lower bits per token is better.
- Eval protocol: Verify that search selection and stopping decisions cannot access the sealed windows. The same baseline define Q must be used for both methods. We want to ensure fair evaluation so we can see which method is better under fair settings.
- Target attainment: A config reaches Q when its selection-validation meets or surpasses the predefined target.


## Prediction
- Expected R_FLOP: Predict a median R of 0.5, with a plausible range from 0.1-1.0. This is only my prediction not a result
- Expected R_time: Predict a median CPU-time ratio of 0.5, with a plausible range of 0.1-0.5. The time results are secondary as background load can interrupt comparisions. Which would make it not a fair comparison
- Probability of meeting the order-of-magnitude claim: 15% before running the experiment
- Additional hypothesis: Successive halving has at least a 60% probability of achieving at least 20% lower FLOPs-to-target than random search. This is just my prediction not a actual result

## Pass rule
Use the median paired FLOPs ratio R as the primary decision statistic.

- HOLDS: Median R ≤ 0.1, consistent with an order-of-magnitude reduction in search compute.

- PARTIAL: 0.1 < median R ≤ 0.8, showing an improvement of at least 20% but not an order-of-magnitude reduction.

- DOES NOT HOLD: Median R > 0.8 when both methods reach Q in all three paired repeats.

- INCONCLUSIVE: Either method fails to reach Q in one or more repeats, invalid runs compromise the comparison, compute budgets differ materially without a defensible correction, or the observed repeats are too inconsistent for a reliable verdict.


## Stop rules

- A run is ended early if: The harness reports an invalid run, causality or a finite-loss check fails, the configs exceed the set model parameter budget, or successive halving eliminates the configuration
- What an early stop means for the verdict: nvalid runs must be investigated and reported; do not silently omit their compute. A configuration eliminated by successive halving counts toward the search budget but is not treated as a completed 1,200-update run. If a selected configuration never reaches Q, report a bound rather than extrapolating beyond observed evaluation points. If these issues prevent a fair comparison, report INCONCLUSIVE.
