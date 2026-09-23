# AdaptiveRL — Experiment Lifecycle and Reproducibility

This document describes the experiment management system, including the output
schema, seed management, manifest format, and comparison workflows.

---

## 1. Experiment Lifecycle

AdaptiveRL experiments are managed through the `ExperimentManager` class and the
`adaptive-rl experiment` CLI sub-commands.

### 1.1 Running an Experiment

```bash
adaptive-rl experiment run --config configs/gridworld_ppo.yaml
adaptive-rl experiment run --config configs/gridworld_astar.yaml
adaptive-rl experiment run --config configs/gridworld_ppo.yaml --seed 43
adaptive-rl experiment run --config configs/smoke_gridworld_ppo.yaml --timesteps 1000
```

Each run:
1. Validates the YAML configuration against the schema.
2. Generates a unique experiment ID.
3. Creates a structured output directory.
4. Records provenance metadata (git commit, Python version, packages).
5. Runs training (RL) or planning (A*).
6. Evaluates the trained agent/planner.
7. Saves all artifacts and a machine-readable manifest.

### 1.2 Experiment ID Format

```
YYYY-MM-DD_<environment>_<algorithm>_seed<seed>
```

Examples:
- `2026-09-17_gridworld_ppo_seed42`
- `2026-09-17_gridworld_astar_seed42`
- `2026-09-17_navigation_sac_seed43`

> [!NOTE]
> Experiment IDs are filesystem-safe and include the date, environment, algorithm,
> and seed. If you run the same experiment twice on the same day, the second run
> will create a duplicate directory. Use `--seed` to differentiate runs.

---

## 2. Output Directory Schema

Each experiment produces the following artifacts under `experiments/results/<experiment_id>/`:

```
experiments/results/<experiment_id>/
├── config.yaml          — copy of the configuration used (reproducibility)
├── manifest.json        — machine-readable provenance record
├── metrics.json         — evaluation metrics (JSON)
├── metrics.csv          — flat scalar metrics (CSV for spreadsheet analysis)
├── evaluation.json      — full evaluation report (RL only)
├── model/               — saved model weights (RL only)
│   └── <name>_final.zip
├── logs/                — training logs (RL only)
└── plots/               — reserved for future visualization artifacts
```

---

## 3. Manifest Schema

Every completed experiment writes `manifest.json` with the following fields:

```json
{
  "experiment_id": "2026-09-17_gridworld_ppo_seed42",
  "created_at": "2026-09-17T03:00:00+00:00",
  "algorithm": "ppo",
  "environment": "gridworld",
  "seed": 42,
  "config_path": "configs/gridworld_ppo.yaml",
  "training_timesteps": 5000,
  "git_commit": "abc1234",
  "python_version": "3.12.3 (main, ...)",
  "platform_info": "Linux 5.15.0 x86_64",
  "package_versions": {
    "adaptive-rl": "0.1.0",
    "gymnasium": "1.0.0",
    "stable-baselines3": "2.3.0",
    "torch": "2.2.0",
    "pydantic": "2.10.0"
  },
  "evaluation_seeds": [42, 43, 44],
  "artifact_paths": {
    "config": "experiments/results/.../config.yaml",
    "model": "experiments/results/.../model/..._final.zip",
    "metrics": "experiments/results/.../metrics.json",
    "metrics_csv": "experiments/results/.../metrics.csv",
    "evaluation": "experiments/results/.../evaluation.json"
  },
  "evaluation_status": "completed",
  "notes": ""
}
```

For planner experiments (A*), `training_timesteps` is `null`.

---

## 4. Metrics Schema

### 4.1 RL Algorithm Metrics (EvaluationMetrics)

| Field | Type | Description |
|:------|:-----|:------------|
| `episodes` | int | Total evaluation episodes |
| `mean_reward` | float | Mean cumulative episodic reward |
| `std_reward` | float | Reward standard deviation |
| `min_reward` | float | Minimum episodic reward |
| `max_reward` | float | Maximum episodic reward |
| `success_rate` | float | Fraction of successful episodes [0, 1] |
| `collision_rate` | float | Fraction of collision episodes [0, 1] |
| `recovery_time` | Optional[float] | Mean recovery time in environment steps over episodes with a measured recovery (null when unavailable/censored everywhere; never 0.0 as a placeholder) |
| `mean_episode_length` | float | Mean steps per episode |
| `std_episode_length` | float | Episode length standard deviation |

### 4.2 Planner Metrics (PlannerEvaluationMetrics)

| Field | Type | Description |
|:------|:-----|:------------|
| `episodes` | int | Total evaluation episodes |
| `success_rate` | float | Fraction where a valid path was found |
| `mean_path_length` | Optional[float] | Mean path length (steps/distance) over successes (null if no successes) |
| `std_path_length` | Optional[float] | Path length standard deviation |
| `min_path_length` | Optional[float] | Minimum path length |
| `max_path_length` | Optional[float] | Maximum path length |
| `mean_planning_time` | Optional[float] | Mean wall-clock planning time in seconds (null if unmeasured) |
| `std_planning_time` | Optional[float] | Planning time standard deviation (null if unmeasured) |
| `collision_rate` | Optional[float] | Always null for offline planners (no dynamic environment step execution) |

> [!IMPORTANT]
> Planner metrics are distinct from RL metrics. Path length and planning time
> are not applicable to RL agents. Reward is not applicable to planners.
> Never compare reward directly between RL and planners.

---

## 5. Seed Management

### 5.1 Environment Randomness vs Algorithm Randomness

The evaluation framework explicitly separates **Environment Randomness** from **Algorithm Randomness**:

- **Environment Randomness**:
  Determines procedural benchmark world generation (grid dimensions, obstacle placement, start/goal coordinates).
  Derived deterministically via:
  $$\text{seed}_{\text{env}}(i) = \text{derive\_evaluation\_seed}(\text{experiment\_seed}, i) = \text{experiment\_seed} + i$$
  Evaluators for both RL agents and classical planners (A*, RRT*) reset the environment using this exact seed, ensuring that episode $i$ presents an identical benchmark problem to every algorithm.

- **Algorithm Randomness**:
  Governs algorithm-internal stochastic exploration (such as RRT* state-space sampling via `derive_planner_seed`, policy network weight initialization, or stochastic action sampling).
  Crucially, changing algorithm randomness (e.g. evaluating with different planner seeds) does **not** alter the benchmark environment's layout, ensuring true ceteris paribus comparisons.

- **Manifest Provenance**:
  The complete sequence of evaluation seeds executed during the run is permanently recorded in `manifest.json` under `evaluation_seeds`.

### 5.2 Generalization Train/Test Split

The generalization framework maintains strictly disjoint
seed distributions:
- Training seeds: `[1000, 1015)` (15 seeds)
- Test seeds: `[2000, 2015)` (15 seeds)

These ranges must never overlap. The framework includes an overlap check.

### 5.3 Determinism and Reproducibility Scoping

The following components are fully deterministic given the same seed:
- A* planner path computation.
- RRT* sampling-based trajectory search given identical planner seed and environment.
- GridWorld procedural generation.
- ContinuousNavigation2DEnv obstacle placement.

The following may not be bit-for-bit reproducible across platforms or torch versions:
- PPO/SAC neural network training (GPU/CPU floating-point non-associativity).
- SB3 parallel environment sampling.

We do not claim bitwise reproducibility for neural network training across disparate hardware, but multi-seed descriptive benchmarking with recorded configurations and environment parity.

---

## 6. Listing and Inspecting Experiments

```bash
# List all experiments
adaptive-rl experiment list

# Inspect a specific experiment
adaptive-rl experiment inspect 2026-09-17_gridworld_ppo_seed42

# Full terminal dashboard
adaptive-rl dashboard

# Compare two experiments
adaptive-rl dashboard --compare 2026-09-17_gridworld_ppo_seed42,2026-09-17_gridworld_astar_seed42
```

---

## 7. Benchmarking and Ablations

Multi-seed benchmarking runs the same configuration across multiple seeds
and reports aggregate statistics:

```bash
# Single config, default seeds [42, 43, 44]
adaptive-rl benchmark --config configs/gridworld_ppo.yaml

# Custom seeds
adaptive-rl benchmark --config configs/gridworld_ppo.yaml --seeds 42,43,44,45,46

# Ablation comparison
adaptive-rl benchmark \
  --config configs/gridworld_ppo.yaml \
  --compare configs/gridworld_astar.yaml \
  --seeds 42,43,44 \
  --output-report experiments/comparison_ppo_vs_astar.json
```

Reported statistics:
- **mean**: Average metric value across seeds.
- **std**: Standard deviation (uncertainty estimate).
- **min**: Best/worst case value.
- **max**: Best/worst case value.

> [!WARNING]
> Results from a small number of seeds (< 5) have high variance. Use more seeds
> for statistically meaningful comparisons. Never present a single-seed result
> as a reliable conclusion.

---

## 8. Controlled Distribution-Shift Benchmark (Issue #105)

The distribution-shift benchmark measures policy robustness under explicit
environment distribution shifts — not just unseen seeds with identical physics.
**The policy is trained only on TRAIN conditions and evaluated without any
adaptation on TEST conditions** (no per-test fine-tuning; a single frozen
policy, verified by an identical policy fingerprint across all scenarios).

### 8.1 Train/Test Protocol

1. Build the TRAIN environment from shared base parameters + TRAIN overrides.
2. Train the RL policy exclusively on TRAIN (training seeds drawn only from the
   TRAIN seed set; a post-training audit raises on any test-seed observation).
3. Freeze the policy (SHA-256 fingerprint of the policy weights is recorded).
4. Evaluate the frozen policy on independent TRAIN/TEST environments with
   disjoint, explicitly recorded seed sets.
5. Record effective environment parameters per scenario and compute TEST-vs-TRAIN
   generalization gaps.

```bash
adaptive-rl benchmark-shifts --config configs/drone_distribution_shift.yaml
```

### 8.2 Scenario Definitions (`configs/drone_distribution_shift.yaml`)

Scenarios are controlled combinations, not single-factor isolations (except TEST-A):

| Scenario | Shift type | Static obstacles | Dynamic obstacles | Wind (steady / gust σ) | Hidden disturbance |
|:---------|:-----------|:-----------------|:------------------|:-----------------------|:-------------------|
| TRAIN | Nominal/static baseline | 8 | 0 | low: 0.5 m/s / 0.15 | 0.0 |
| TEST-A | Obstacle-density shift | 10 | 0 | low: 0.5 m/s / 0.15 | 0.0 |
| TEST-B | Obstacle-density + moderate-wind compound shift | 12 | 0 | moderate: 4.0 m/s / 0.6 | 0.0 |
| TEST-C | Dynamic-obstacle + moderate-wind compound shift | 8 | 4 | moderate: 4.0 m/s / 0.6 | 0.0 |
| TEST-D | High hidden-disturbance shift | 8 | 0 | low: 0.5 m/s / 0.15 | high: 6.0 |

Wind force scale: `F = linear_damping * wind` with `linear_damping = 0.05`
(max acceleration 4.0 m/s²), so moderate wind is a 0.2 m/s² persistent bias.
`disturbance_strength` is the per-axis stationary std (m/s of equivalent wind)
of a hidden OU process excluded from wind observations. Train seeds are
`[1000, 1015)`; every TEST scenario uses the shared paired block `[2000, 2015)`.
Do not claim TEST-B/C/D isolate individual causal factors: several environment
parameters change simultaneously by design (Issue #105 prescribes these
combinations).

Event detection is comparable across scenarios: a single fixed
`recovery_event_threshold` (0.7 m/s, TRAIN-calibrated) is passed to every
scenario as `disturbance_event_threshold_override` and recorded per scenario
with source `benchmark_override`. The event definition is fixed; only the
disturbance distribution shifts.

### 8.3 Metric and Recovery-Time Definitions

Per scenario: `success_rate`, `collision_rate`, `mean_reward`, `recovery_time`
(environment steps), episode count, truncation rate. Recovery is defined by the
environment-step telemetry contract (`disturbance_active`,
`disturbance_onset_step`, `recovered`, `episode_recovery_time`, ...):

- Event: transient magnitude `||gust + injected||` ≥ the fixed
  `recovery_event_threshold` shared by all scenarios. Onset: first step
  at/above threshold (no overlap).
- Recovery: magnitude below threshold AND speed within `recovery_speed_tolerance`
  (1.0 m/s) of the onset speed for `recovery_hold_steps` (5) consecutive steps.
- `recovery_time = recovery_step − onset_step` (≥ 1; never wall-clock).
- Still open at episode end → censored (counted, never averaged, never 0).
- Zero events in an episode → unavailable (null, never 0.0).
- Aggregation is **event-weighted**: `recovery_time` pools every completed event
  time across episodes (episodes with many events weigh proportionally more).
  Reported alongside: total/completed/censored event counts, completion rate,
  and censoring rate.
- **Conditionality warning**: `recovery_time` is conditional on recovery and must
  never be read as an unconditional robustness score. A heavily censored scenario
  can show a low conditional mean precisely because its hardest events never
  recovered — always interpret it jointly with the completion/censoring rates.

### 8.4 Generalization-Gap Formulas

Success/reward/collision gaps keep the positive-means-TEST-worse convention:

- `success_gap = train_success_rate − test_success_rate`
- `reward_gap = train_mean_reward − test_mean_reward`
- `collision_gap = test_collision_rate − train_collision_rate`

Recovery gaps are censoring-aware:

- `recovery_time_gap = test_mean_completed − train_mean_completed`, defined
  **only** when both sides recorded events and both completion rates are exactly
  1.0; otherwise null, so a censored TEST can never look spuriously faster.
- `recovery_completion_gap = train_completion_rate − test_completion_rate`
  (positive = TEST recovers a smaller share of its events).
- `recovery_censoring_gap = test_censoring_rate − train_censoring_rate`
  (positive = TEST leaves a larger share unrecovered).

Unavailable inputs propagate null. No composite score is produced.

### 8.5 Output Artifact

`experiments/results/distribution_shift/<name>_shift_report.json` (schema
version `1.1`) contains: experiment/algorithm/training provenance (timesteps,
seed sets, sampled training seeds, contamination audit, policy fingerprint),
environment provenance, config SHA-256, scenario definitions with overrides,
effective environment parameters (including the event threshold and its source),
per-scenario metrics, per-scenario `recovery` summaries (total/completed/
censored events, completion/censoring rates, event-weighted mean), raw
per-seed `episodes` records (sufficient to recompute every aggregate),
per-scenario gaps, and the recovery definition. The report is sufficient to
reproduce and independently verify the evaluation.

---

## 9. Logged Experiments

The following table records experiments that have been run during development.
Only actual runs are listed here — no fabricated results.

| Experiment | Phase | Environment | Algorithm | Timesteps | Status |
|:-----------|:------|:------------|:----------|:----------|:-------|
| Smoke tests | Current experiment workflow | GridWorld | A* | N/A (planner) | Infrastructure validated |

Full benchmark results require running `adaptive-rl benchmark` and will depend
on available compute. Sample configurations are provided in `configs/`.
