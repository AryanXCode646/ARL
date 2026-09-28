# AdaptiveRL Hypothesis and Evaluation Protocol

## 1. Purpose
This document specifies an auditable, pre-specified experimental protocol for evaluating a proposed adaptive capability for RL agents.

> [!WARNING]
> **Implementation Status**
>
> **CURRENT REPOSITORY INFRASTRUCTURE**: The current distribution-shift benchmark trains on a nominal scenario, freezes the resulting policy, fingerprints it, and rejects any mutation during evaluation. See `src/adaptive_rl/experiments/shift_runner.py` (commit `267d267235bc55785116fc625cedf251e29723ba`).
>
> **FUTURE EXPERIMENT**: The online Adaptive treatment defined in this document.
>
> **FUTURE IMPLEMENTATION REQUIREMENT**: An online update/adaptation harness. No such harness exists in the current repository.
>
> **PR #168 / Issue #98**: Documentation only. This document does not imply that PR #168 makes the Adaptive experiment runnable. Do NOT interpret any section of this protocol as describing currently executable Adaptive treatment cells.

## 2. Research Question
Does an agent equipped with a future online environment-shift adaptation treatment recover performance faster after a distribution shift than a matched fixed-policy baseline (PPO or SAC) trained on the same nominal distribution?

## 3. Hypotheses
The primary analysis targets the **horizon-truncated recovery time** under the moderate shift condition.

Let $\tau$ be the first valid post-shift episode index at which the persistence-confirmed recovery criterion is met (see §5). We define a finite-horizon analysis value:

$$T_H = \min(\tau, H)$$

where $H = 15$ completed post-shift episodes. **$H = 15$ is a protocol-level pre-registered horizon for Issue #98 and is not an ARL-wide evaluation default.**

The primary estimand is the finite-horizon recovery time $T_H$, not the unobserved true recovery time that might occur beyond $H$.

For runs that do not achieve recovery within the horizon:

$$T_H = H, \quad \texttt{recovery\_status} = \texttt{right\_censored}$$

Let $D_i = T_{H,i}(\text{Adaptive}) - T_{H,i}(\text{Fixed})$ be the paired treatment difference for seed $i$. **A negative $D_i$ favors the Adaptive treatment** (lower recovery time is better). The primary estimand is the mean paired difference $\mu_D = \mathbb{E}[D_i]$.

### H0 — Null Hypothesis
$\mu_D \ge 0$. The AdaptiveRL treatment does not reduce the horizon-truncated recovery time compared to the fixed baseline.

### H1 — Alternative Hypothesis
$\mu_D < 0$. The AdaptiveRL treatment achieves a strictly lower horizon-truncated recovery time than the fixed baseline.

## 4. Treatment Definition
* **Algorithm**: The base learning algorithm (PPO or SAC). Algorithm and treatment are separate factors.
* **Treatment**: Adaptive vs Fixed.

The base algorithm and its hyperparameters remain identical between treatments within every environment × algorithm cell. The sole intervention is the presence of the online adaptation mechanism and its associated compute overhead during the post-shock phases (episodes 6–15). The Fixed arm never updates its policy weights at any point.

## 5. Operational Definition of Adaptation

**Time axis**: **Completed post-shift episode index**. All recovery definitions use this single canonical axis.

* **Pre-shift performance ($P_{pre}$)**: Mean episodic return over the pre-shift evaluation episodes (see §14.1 for seed definition).

* **Immediate post-shift performance ($P_0$)**: Mean episodic return over post-shift episodes 1–5 (the shock-measurement window).

  > **CRITICAL PROTOCOL REQUIREMENT — TREATMENT FREEZE DURING $P_0$**: Both the Adaptive and Fixed arms MUST evaluate with their frozen initial post-training policies during episodes 1–5. The Adaptive arm is NOT permitted to perform any online update before $P_0$ has been measured. This requirement ensures $P_0$ is identical in meaning across treatments and eliminates treatment-dependent normalization baselines.

* **Performance at episode $t$ ($P(t)$)**: Mean return over the **causal trailing window** of the five most recently completed post-shift episodes ending at episode $t$. The first valid window measurement occurs at $t = 5$. No future episode is ever used to declare recovery at an earlier episode index.

* **Recovery metric ($R(t)$)**:
  $$R(t) = \frac{P(t) - P_0}{P_{pre} - P_0}$$

  **Edge-case handling**:
  1. If $P_{pre} \le P_0$ (no measurable post-shift degradation), the run is assigned $\texttt{recovery\_status} = \texttt{no\_degradation}$, $\tau = 0$, and $T_H = 0$. This is a protocol convention meaning that no recovery was required; it does not constitute evidence of instantaneous adaptation. Its statistical consequence (anchoring the recovery-time distribution at 0) is documented as a limitation (§22).
  2. If $P(t) < P_0$, $R(t)$ is permitted to be negative.
  3. If $P(t) > P_{pre}$, $R(t)$ is permitted to exceed 1.
  4. The denominator is exactly zero only when $P_{pre} = P_0$, which is fully handled by rule 1.

* **Persistence rule**: The earliest recovery time $\tau$ is the smallest eligible $t \ge 5$ such that $R(t) \ge 0.9$, $R(t+1) \ge 0.9$, and $R(t+2) \ge 0.9$. No future observation may be used to determine a recovery event that is declared at an earlier time. Given window = 5, persistence = 3, and $H = 15$, **the latest possible recovery-start index is $t = 13$** (because $\tau = 13$ requires windows ending at 13, 14, and 15, all within $H$).

## 6. Independent Variables
| Variable | Levels |
|---|---|
| **Algorithm** | PPO, SAC |
| **Treatment** | Adaptive vs Fixed |
| **Environment** | `gridworld`, `navigation_2d`, `traffic_signal`, `drone_disturbed` |
| **Shift magnitude** | Mild, Moderate, Severe |

## 7. Dependent Variables
* **Primary**: Horizon-truncated recovery time ($T_H$) (lower is better; defined per §5).
* **Secondary**: Final post-adaptation episodic return, episodic success rate.

## 8. Controlled Variables
* Pre-registered training timestep budget, pinned to repository commit SHA (see §15).
* Fixed checkpoint selection rule (final training checkpoint, see §16).
* Shared evaluation seeds for paired testing (see §14).

## 9. Baselines
* **Fixed PPO/SAC baseline**: Trained on nominal distribution, policy frozen immediately after training, evaluated under shifted distribution with policy weights locked at all times.

## 10. Evaluation Environments
Environments verified in the ARL registry at commit `267d267235bc55785116fc625cedf251e29723ba`:
* `gridworld` (Action space: Discrete)
* `navigation_2d` (Action space: Continuous)
* `traffic_signal` (Action space: Discrete)
* `drone_disturbed` (Action space: Continuous)

## 11. Training Distribution
All agents train exclusively on the nominal configuration of the environment as specified in the pinned config (§15). Test seeds and shift parameters must not leak into the training phase. Post-hoc scenario selection is strictly prohibited. The existing runner enforces this via `TrainingDistributionWrapper` and the seed-containment audit in `shift_runner.py`.

## 12. Distribution Shifts

Values listed as "VERIFIED" appear literally in the cited config at the pinned commit. Values listed as "[FUTURE PROTOCOL VALUE]" are pre-registered targets that require future implementation or configuration before the experiment is executable.

### GridWorld — `num_obstacles`
Source: `configs/gridworld_ppo.yaml` (nominal), `configs/generalization_gridworld.yaml` (mild), `configs/benchmark_planners_gridworld.yaml` (moderate as design reference).
Commit SHA: `267d267235bc55785116fc625cedf251e29723ba`.

| Condition | num_obstacles | Status |
|---|---|---|
| Nominal | 3 | VERIFIED (`configs/gridworld_ppo.yaml`) |
| Mild | 4 | VERIFIED (`configs/generalization_gridworld.yaml`) |
| Moderate | 6 | DESIGN REFERENCE (`configs/benchmark_planners_gridworld.yaml`; that config serves a different benchmark purpose) |
| Severe | 8 | [FUTURE PROTOCOL VALUE] |

A dedicated distribution-shift config for `gridworld` does not exist at the pinned commit. Moderate and severe shift values are pre-registered design targets. **No gridworld shift test is currently executable.**

### Navigation 2D — `lidar_noise`
Source: `configs/navigation.yaml`. `lidar_noise` does not appear in any source file or config at the pinned commit.

| Condition | lidar_noise | Status |
|---|---|---|
| All conditions | — | [FUTURE PROTOCOL PARAMETER — NOT CURRENTLY IMPLEMENTED IN ARL] |

### Traffic Signal — `arrival_rates`
Source: `configs/traffic_ppo.yaml`.
Commit SHA: `267d267235bc55785116fc625cedf251e29723ba`.

| Condition | arrival_rates (4-lane tuple) | Status |
|---|---|---|
| Nominal | (0.35, 0.35, 0.25, 0.25) | VERIFIED (`configs/traffic_ppo.yaml`) |
| Mild | (0.5, 0.5, 0.5, 0.5) | [FUTURE PROTOCOL VALUE] |
| Moderate | (0.7, 0.7, 0.7, 0.7) | [FUTURE PROTOCOL VALUE] |
| Severe | (1.0, 1.0, 1.0, 1.0) | [FUTURE PROTOCOL VALUE] |

A dedicated distribution-shift config for `traffic_signal` does not exist at the pinned commit.

### Drone Disturbed — `wind_speed` (m/s) and `gust_sigma`
Source: `configs/drone_distribution_shift.yaml`.
Commit SHA: `267d267235bc55785116fc625cedf251e29723ba`.
Physical units: `wind_speed` in m/s steady wind; `gust_sigma` is OU gust volatility (per-axis stationary std ≈ sigma/sqrt(2·theta)).

| Condition | wind_speed | gust_sigma | Status |
|---|---|---|---|
| Nominal (TRAIN) | 0.5 | 0.15 | VERIFIED (`drone_distribution_shift.yaml`, TRAIN scenario) |
| Mild | 2.0 | 0.3 | [FUTURE PROTOCOL VALUE] |
| Moderate | 4.0 | 0.6 | VERIFIED (`drone_distribution_shift.yaml`, TEST-B and TEST-C scenarios) |
| Severe | 8.0 | 1.2 | [FUTURE PROTOCOL VALUE] |

## 13. Evaluation Phase Sequence

The following sequence is mandatory. Any deviation is a protocol violation.

```
Step 1 — TRAIN
  Agent trains on nominal parameters for the pre-registered budget (§15).
  Policy is frozen immediately after training.

Step 2 — CHECKPOINT
  Final training checkpoint is selected (§16).
  Policy fingerprint is computed and recorded.
  Policy weights are immutable for the remainder of the experiment.

Step 3 — PRE-SHIFT EVALUATION
  Both arms evaluate on the nominal environment.
  Episode seeds: pre_shift_episode_seed(i, j) for j = 1 ... K_pre (see §14).
  Both arms use IDENTICAL pre-shift seeds.
  Ppre is computed from these episodes.

Step 4 — SHIFT INTRODUCTION
  Environment parameters are updated to the moderate shift condition.
  No policy update occurs at this transition.

Step 5 — SHOCK-MEASUREMENT WINDOW (post-shift episodes 1–5)
  BOTH the Adaptive and Fixed arms evaluate with the FROZEN initial policy.
  The Adaptive arm MUST NOT perform any online update during this window.
  Episode seeds: post_shift_episode_seed(i, 1) ... post_shift_episode_seed(i, 5).
  P0 is computed from these 5 episodes.
  Both arms observe the SAME P0 from the SAME frozen policy.

Step 6 — ADAPTATION PHASE (post-shift episodes 6–15)
  Adaptive arm: online adaptation is enabled beginning at episode 6.
  Fixed arm: policy remains frozen throughout.
  Episode seeds: post_shift_episode_seed(i, 6) ... post_shift_episode_seed(i, 15).
  The SAME seeds are used by both arms.

Step 7 — CAUSAL RECOVERY MEASUREMENT
  P(t) is computed at each t = 5, 6, ..., 15 using the causal trailing window.
  Recovery criterion is checked at each t.
  tau is recorded as the earliest eligible t satisfying the persistence rule.
  T_H = min(tau, H).
```

## 14. Seed Protocol

### 14.1 Training Seeds
Ten independent experimental replicates: training seeds are a pre-registered set of 10 distinct integers. This count represents a pre-registered computational budget, not a powered sample size derived from a formal power calculation.

### 14.2 Pre-shift Episode Seeds
For each training seed $i$, define $K_{pre}$ pre-shift episode seeds:

```
pre_shift_episode_seed(i, j) = hash(training_seed(i), "pre", j)  for j = 1 ... K_pre
```

where the hash function is a deterministic, reproducible integer derivation (e.g., `int(hash((training_seed(i), "pre", j))) & 0xFFFFFFFF`). These seeds must be:
* **Deterministically derived** from the training seed and position index.
* **Globally unique** (no two seeds in the full schedule share a value).
* **Strictly disjoint from training seeds**.
* **Identical for both the Adaptive and Fixed arms**.

### 14.3 Post-shift Episode Seeds
For each training seed $i$, define exactly 15 post-shift episode seeds:

```
post_shift_episode_seed(i, j) = hash(training_seed(i), "post", j)  for j = 1 ... 15
```

The same derivation rules as §14.2 apply. Additionally:
* **Seeds for $j = 1..5$ (shock window)** must be identical across arms.
* **Seeds for $j = 6..15$ (adaptation phase)** must be identical across arms.
* **All 15 post-shift seeds must be disjoint from all pre-shift seeds** for the same training run.

### 14.4 RNG Initialization
At each episode $j$ for training seed $i$, the environment is reset with the computed seed. NumPy, PyTorch, and environment-specific RNG state must be seeded deterministically from this value.

### 14.5 Train/Test Disjointness
All pre-shift and post-shift episode seeds must be disjoint from training seeds. This requirement is validated by the existing seed-containment audit in `src/adaptive_rl/experiments/shift_runner.py`.

## 15. Training Budgets (Pinned)

Budgets are pinned to commit SHA `267d267235bc55785116fc625cedf251e29723ba`. The same budget applies to both Adaptive and Fixed arms within every cell.

| Environment | Algorithm | Config File | total_timesteps |
|---|---|---|---|
| `gridworld` | PPO | `configs/gridworld_ppo.yaml` | 5,000 |
| `traffic_signal` | PPO | `configs/traffic_ppo.yaml` | 10,000 |
| `drone_disturbed` | PPO | `configs/drone_distribution_shift.yaml` | 60,000 |
| `drone_disturbed` | SAC | `configs/drone_distribution_shift.yaml` (future SAC extension) | [FUTURE PROTOCOL VALUE] |
| `navigation_2d` | PPO | `configs/navigation.yaml` | 100,000 |
| `navigation_2d` | SAC | [FUTURE CONFIG] | [FUTURE PROTOCOL VALUE] |

> [!NOTE]
> These values are protocol-frozen at the pinned commit. If a YAML file changes in a future commit, the above table remains the authoritative pre-registered budget for this experiment.

## 16. Checkpoint Selection
The deterministic pre-declared rule is the **final training checkpoint** at the end of the training budget. Post-shift evaluation performance is never used to select checkpoints. The same selection rule applies to both Adaptive and Fixed conditions. A policy fingerprint is computed before evaluation begins and verified after each scenario (see `shift_runner.py`).

## 17. Primary Endpoint and Family

The single primary estimand is the mean paired difference $\mu_D$ in horizon-truncated recovery time $T_H$ under the **moderate shift condition**, where $D_i = T_{H,i}(\text{Adaptive}) - T_{H,i}(\text{Fixed})$. A negative $\mu_D$ indicates the Adaptive treatment recovers faster.

### Primary Family — Preregistered Cells

**Option B is chosen**: All 6 cells belong to the preregistered primary family. No full-family inferential conclusion (including Holm-adjusted significance declarations) is made until all 6 cells are executable. Partial results from currently executable cells are reported descriptively pending family completion.

The 6 preregistered primary contrasts are:

| # | Environment | Algorithm | Contrast | Executable Now? |
|---|---|---|---|---|
| 1 | `gridworld` | PPO | Adaptive-vs-Fixed | No — no shift config; shift values partially FUTURE |
| 2 | `traffic_signal` | PPO | Adaptive-vs-Fixed | No — shift values FUTURE; Adaptive harness FUTURE |
| 3 | `drone_disturbed` | PPO | Adaptive-vs-Fixed | No — Adaptive harness FUTURE |
| 4 | `drone_disturbed` | SAC | Adaptive-vs-Fixed | No — Adaptive harness FUTURE |
| 5 | `navigation_2d` | PPO | Adaptive-vs-Fixed | No — `lidar_noise` FUTURE; Adaptive harness FUTURE |
| 6 | `navigation_2d` | SAC | Adaptive-vs-Fixed | No — `lidar_noise` FUTURE; Adaptive harness FUTURE |

**No cell is currently executable for the Adaptive treatment**. The Fixed arm frozen-policy evaluation is executable for cells 3–4 via `configs/drone_distribution_shift.yaml`. Full Holm-Bonferroni correction over all 6 cells is applied only when all 6 produce valid p-values.

## 18. Statistical Analysis Plan

### 18.1 Experimental Unit
The independent experimental unit is the **independent training run / seed** ($n = 10$). Evaluation episodes are repeated observations nested within that unit. Individual episodes are never treated as independent experimental replicates.

### 18.2 Primary Test
An **exact paired permutation (sign-flip) test** on the mean paired difference $D_i = T_{H,i}(\text{Adaptive}) - T_{H,i}(\text{Fixed})$. Since $n = 10$, all $2^{10} = 1024$ possible sign assignments of the vector $(D_1, \dots, D_{10})$ are enumerated. The test statistic is $\bar{D} = \frac{1}{10}\sum_{i=1}^{10} D_i$. The one-sided p-value is:

$$p = \frac{|\{s \in \{-1,+1\}^{10}: \bar{D}(s) \le \bar{D}_\text{obs}\}|}{1024}$$

**Required assumption**: The test is exact conditional on the symmetry/exchangeability assumption — under $H_0$, the $D_i$ are exchangeable in sign (i.e., the distribution of $D_i$ is symmetric around 0). This assumption is not guaranteed in general; it is violated if, e.g., the distribution of $D_i$ is skewed under the null. The test is not assumption-free.

The test is **one-sided** consistent with $H_1$ ($\mu_D < 0$, Adaptive faster). A negative $D_i$ favors the Adaptive treatment everywhere in this protocol.

### 18.3 Effect Size
**Cohen's $d_z$** for paired samples:

$$d_z = \frac{\bar{D}}{\text{SD}(D_i)}$$

A negative $d_z$ favors the Adaptive treatment (consistent with the sign convention throughout).

**Edge case**: If $\text{SD}(D_i) = 0$ and $\bar{D} = 0$, report $d_z = 0$ (no effect). If $\text{SD}(D_i) = 0$ and $\bar{D} \ne 0$, $d_z$ is undefined; report the raw mean difference $\bar{D}$ instead.

### 18.4 Confidence Interval
A 95% parametric paired t-interval on $\bar{D}$, computed at the independent training-seed level (never treating individual episodes as independent replicates):

$$\bar{D} \pm t_{0.025, 9} \cdot \frac{\text{SD}(D_i)}{\sqrt{10}}$$

This interval is used alongside the primary non-parametric test because the permutation test does not natively yield a confidence interval. **Limitation**: with $n = 10$, the t-interval assumes approximate normality of the $D_i$ distribution, which may not hold. It is an uncertainty interval for the mean paired difference, not a claim of exact coverage.

### 18.5 Multiple Comparisons
**Holm-Bonferroni** step-down procedure applied to the full primary family of 6 tests to control FWER at $\alpha = 0.05$. This correction is applied only when all 6 cells produce valid p-values. Secondary exploratory tests remain uncorrected and are reported descriptively.

## 19. Failure / Invalid / Censored Runs

Identical failure rules apply to **both** the Adaptive and Fixed arms. The following categories are reported separately for each arm.

| Category | Definition | Primary Analysis | Sensitivity Analysis |
|---|---|---|---|
| **Planned** | 10 training seeds per cell per arm | — | — |
| **Completed** | Successful training + evaluation | Included | Included |
| **Failed** | Training divergence, NaN, or execution crash | Excluded; reported separately | $T_H = H$ assigned |
| **No-degradation** | $P_{pre} \le P_0$; $\tau = 0$, $T_H = 0$ | Included (as $T_H = 0$) | Included |
| **Right-censored** | No recovery within $H$ episodes; $T_H = H$ | Included (as $T_H = H$) | Included |

> Excluding failed runs from the primary estimand can introduce attrition bias if failure rates differ between arms. The pre-registered sensitivity analysis bounds this risk by assigning $T_H = H$ (worst case) to every failed run in either arm and re-running the primary test.

## 20. Reporting Requirements

### 20.1 Current ARL Artifact (Fixed-Policy Benchmark)
The existing `ShiftBenchmarkReport` JSON schema stores:
* `experiment_name`, `environment_name`, `algorithm_name`
* `training_provenance` (includes `total_training_timesteps`, `policy_frozen: true`, `policy_fingerprint`)
* `scenario_results`: per-seed episode records (`seed`, `reward`, `length`, `success`, `collision`, `recovery_times`, `recovery_events`)
* `config_sha256`

### 20.2 Derived Research Quantities
The research quantities $P_{pre}$, $P_0$, $P(t)$, $R(t)$, $\tau$, and $T_H$ **are not native fields** of the current `ShiftBenchmarkReport`. They are derived from the stored episode-level reward observations. Any future implementation of the Adaptive experiment must derive these values from episode records and document the derivation code.

### 20.3 Future Adaptive Experiment Artifact
A future derived artifact must additionally store the per-episode $P(t)$ trajectory, the $P_0$ value, the $\tau$ determination, the $T_H$ assignment, and the `recovery_status` for each seed and arm.

## 21. Success Criteria
Statistical evidence requires $p < 0.05$ (Holm-adjusted, one-sided) on the primary endpoint across all 6 family members. Practical significance is assessed independently via $d_z$.

## 22. Threats to Validity

| Threat | Why it matters | Mitigation | Remaining limitation |
|---|---|---|---|
| **Treatment-contaminated P0** | If Adaptive updates before P0 is measured, P0 differs between arms, invalidating the recovery ratio | Protocol mandates frozen policy during episodes 1–5 for both arms | Depends on correct future implementation of the freeze rule |
| **No-degradation convention** | Assigning $\tau = 0$ when $P_{pre} \le P_0$ anchors the distribution at 0 without evidence of true adaptation | Retained as protocol convention with explicit labeling; not evidence of instantaneous adaptation | Inflates apparent speed of adaptation in the $\tau = 0$ subset |
| **Finite-horizon truncation** | Non-recovering runs at $T_H = H$ make the estimand a finite-horizon mean, not a true recovery-time mean | Explicitly defined as the finite-horizon estimand; right-censoring labeled | Test is not conservative in general; it simply measures a different (truncated) quantity |
| **Limited n = 10** | High variance across seeds may obscure true treatment effects | Paired evaluation seeds reduce within-pair variance | Small sample; formal power undetermined |
| **Failed-run attrition** | Excluding crashes can introduce attrition bias if Adaptive is less stable | Symmetric failure rules; worst-case sensitivity analysis for both arms | True behavior of crashed runs unrecoverable |
| **Online-adaptation future implementation** | The treatment is not executable; no adaptation harness exists | Protocol pre-registered before implementation | Actual adaptation behavior entirely untested |
| **Unequal compute** | Adaptive agent incurs additional online compute overhead | Declared as part of the intervention; Fixed arm's budget remains constant | Deployment latency and hardware costs unmeasured |
| **Config drift** | YAML files may change in future commits | Training budgets pinned to commit `267d267235bc55785116fc625cedf251e29723ba` | Reproducibility depends on SHA remaining accessible |
| **Environment heterogeneity** | Effects may be environment-specific, not generalizable | Protocol evaluates across 4 environments | Small environment count; all simulated |
| **Shift realism** | Synthetic parameter shifts may not reflect real deployment conditions | Shift values target documented ARL physical parameters | Simulation-to-reality gap remains |
| **Sign-symmetry assumption** | The exact sign-flip test assumes $D_i$ are symmetric under $H_0$ | Assumption explicitly stated; test result is conditional on it | Type I error may be inflated if assumption is violated |

## 23. Reproducibility Checklist
* [ ] Repository SHA `267d267235bc55785116fc625cedf251e29723ba` recorded in `ShiftBenchmarkReport`.
* [ ] All config YAML files archived at the pinned commit.
* [ ] All training seeds, pre-shift episode seeds, and post-shift episode seeds logged.
* [ ] Post-shift seeds proven disjoint from training seeds via existing runner checks.
* [ ] Policy fingerprint computed before and verified after each evaluation scenario.
* [ ] $P_0$ measurement phase explicitly verified as pre-update for both arms.
* [ ] Derivation code for $P_{pre}$, $P_0$, $P(t)$, $R(t)$, $\tau$, $T_H$ committed alongside results.

## 24. Open Research Questions
* How does the adaptation compute overhead scale with observation dimensionality?
* Can representation learning improve the sample efficiency of the online adaptation phase?
* What minimum n is required to achieve 80% power for the expected effect size of online adaptation?
