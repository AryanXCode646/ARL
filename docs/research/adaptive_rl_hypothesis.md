# AdaptiveRL Hypothesis and Evaluation Protocol

## 1. Purpose
This document specifies an auditable, pre-specified experimental protocol for evaluating a proposed adaptive capability for RL agents.

> [!WARNING]
> **Implementation Status**
>
> **Current ARL capability**: The current distribution-shift benchmark (e.g., in `configs/drone_distribution_shift.yaml`) evaluates a strictly frozen trained policy and enforces policy fingerprints.
>
> **Protocol-defined future treatment**: The Adaptive treatment described here is a future experimental condition that permits online adaptation after the shift.
>
> **Scope of this issue**: Issue #98 defines the experimental contract only. It does not implement the treatment. This document does not imply that PR #168 makes online adaptation executable.

## 2. Research Question
Does an agent equipped with a future online environment-shift adaptation treatment recover performance faster after a distribution shift than a matched fixed-policy baseline (PPO or SAC) trained on the same nominal distribution?

## 3. Hypotheses
The primary analysis targets the **horizon-truncated recovery time** under the moderate shift condition.

Let $\tau$ be the first valid post-shift episode index at which the recovery criterion is met. We define a finite-horizon analysis value:
$$ T_H = \min(\tau, H) $$
where $H = 15$ completed post-shift evaluation episodes (derived from `eval_episodes` in `configs/drone_distribution_shift.yaml`).

For runs that do not recover within the horizon:
$T_H = H$
and `recovery_status = right_censored`.

Let $D_i = T_{H,i}(\text{Adaptive}) - T_{H,i}(\text{Fixed})$ be the paired treatment difference for seed $i$. The primary estimand is the mean paired difference $\mu_D$. A negative $D_i$ favors the Adaptive treatment.

### H0 — Null Hypothesis
$\mu_D \ge 0$. The AdaptiveRL treatment does not reduce the horizon-truncated recovery time compared to the fixed baseline.

### H1 — Alternative Hypothesis
$\mu_D < 0$. The AdaptiveRL treatment achieves a lower horizon-truncated recovery time than the fixed baseline.

## 4. Treatment Definition
* **Algorithm**: The base learning algorithm (PPO or SAC).
* **Treatment**: Adaptive vs Fixed.

The base algorithm and hyperparameters remain identical between treatments. The intervention is strictly the presence of the online adaptation mechanism and its associated compute overhead during the post-shift phase.

## 5. Operational Definition of Adaptation
**Time axis**: **Completed post-shift episode index**. All recovery definitions use this single canonical axis.

* **Pre-shift performance ($P_{pre}$)**: Mean episodic return over the pre-shift evaluation episodes.
* **Immediate post-shift performance ($P_0$)**: Mean return over the first 5 evaluation episodes after shift introduction.
* **Performance at episode $t$ ($P(t)$)**: Mean return over a **causal trailing window** of 5 evaluation episodes ending at index $t$. The first valid measurement occurs at $t = 5$. A future episode is never used to declare recovery at an earlier time.
* **Recovery metric ($R(t)$)**:
  $$ R(t) = \frac{P(t) - P_0}{P_{pre} - P_0} $$

  **Edge-case handling**:
  1. If $P_{pre} \le P_0$ (no measurable post-shift degradation), the recovery time $\tau$ is defined as 0, because no recovery is required.
  2. If $P(t) < P_0$, $R(t)$ is permitted to be negative.
  3. If $P(t) > P_{pre}$, $R(t)$ is permitted to be greater than 1.
  4. The denominator is exactly zero only when $P_{pre} = P_0$, which is handled by rule 1.
* **Persistence rule**: Recovered at $t$ iff $R(t) \ge 0.9$, $R(t+1) \ge 0.9$, and $R(t+2) \ge 0.9$. $\tau$ is the smallest eligible $t$ for which this condition is true.

## 6. Independent Variables
| Variable | Levels |
|---|---|
| **Algorithm** | PPO, SAC |
| **Treatment** | Adaptive vs Fixed |
| **Environment** | `gridworld`, `navigation_2d`, `traffic_signal`, `drone_disturbed` |
| **Shift magnitude** | Mild, Moderate, Severe |

## 7. Dependent Variables
* **Primary**: Horizon-truncated recovery time ($T_H$) (lower is better).
* **Secondary**: Final post-adaptation return, success rate.

## 8. Controlled Variables
* Pre-registered training timestep budget.
* Fixed checkpoint selection (final training checkpoint).
* Shared evaluation seeds for paired testing.

## 9. Baselines
* **Fixed PPO/SAC baseline**: Trained on nominal distribution, evaluated under shifted distribution with a frozen policy.

## 10. Evaluation Environments
Environments verified in the ARL registry:
* `gridworld` (Action space: Discrete)
* `navigation_2d` (Action space: Continuous)
* `traffic_signal` (Action space: Discrete)
* `drone_disturbed` (Action space: Continuous)

## 11. Training Distribution
All agents train exclusively on the nominal configuration of the environment as defined in the training configs. Test seeds and shift parameters must not leak into the training phase. Post-hoc scenario selection is strictly prohibited.

## 12. Distribution Shifts
Shift parameters target verified configuration endpoints in the repository:
* **`gridworld`** (`configs/gridworld_ppo.yaml`) — `num_obstacles`: nominal = 2, mild = 4, moderate = 6, severe = 8.
* **`navigation_2d`** (`configs/navigation.yaml`) — `lidar_noise`: *[FUTURE PROTOCOL PARAMETER — NOT CURRENTLY IMPLEMENTED IN ARL]*.
* **`traffic_signal`** (`configs/traffic_ppo.yaml`) — `arrival_rates` (tuple applied to all lanes): nominal = (0.3, 0.3, 0.3, 0.3), mild = (0.5, 0.5, 0.5, 0.5), moderate = (0.7, 0.7, 0.7, 0.7), severe = (1.0, 1.0, 1.0, 1.0).
* **`drone_disturbed`** (`configs/drone_disturbed_ppo.yaml`, `configs/drone_distribution_shift.yaml`) — `wind_speed` (m/s) and `gust_sigma`: nominal = (0.5, 0.15), mild = (2.0, 0.3), moderate = (4.0, 0.6), severe = (8.0, 1.2).

## 13. Evaluation Procedure
The evaluation separates cleanly into distinct phases:
1. **Pre-shift measurement**: Episodes on nominal parameters.
2. **Shift introduction**: Environment parameters are updated.
3. **Adaptation period**: Agent interacts with the shifted environment (online updates enabled for Adaptive treatment, policy frozen for Baseline).
4. **Post-adaptation measurement**: Final measurement of adapted performance.

## 14. Seed Protocol
* **Training seeds**: 10 independent experimental replicates. This seed count represents a pre-registered computational budget, not a powered sample size derived from a formal calculation.
* **Evaluation seeds**: For each training seed, a paired evaluation seed is used as a shared variance-control mechanism across treatments.
* Randomness for NumPy, PyTorch, and environment initialization must be seeded deterministically.

## 15. Budgets
* **Training budget**: The exact pre-registered `total_timesteps` from the archived config for that environment × algorithm cell (e.g., 60,000 from `drone_distribution_shift.yaml`). The same value is used for Adaptive and Fixed within a cell.
* **Adaptation budget**: Both conditions receive the exact same number of environment interactions ($H = 15$ post-shift episodes). The Adaptive treatment performs additional online computation; this compute overhead is an explicit part of the intervention. The baseline remains frozen.

## 16. Checkpoint Selection
The deterministic pre-declared rule is the **final training checkpoint** at the end of the training budget. Post-shift evaluation performance is never used to select checkpoints. The same selection rule applies to all conditions.

## 17. Primary Endpoint and Family
The single primary estimand is the paired difference in horizon-truncated recovery time ($T_H$) under the moderate shift condition.

**Primary Family**:
The primary family consists of exactly **6 statistical tests**, pairing specific pre-registered algorithms with environments. These are explicitly partitioned into currently executable cells and future extension cells:

*Current executable protocol cells* (4 tests):
1. `gridworld` × PPO
2. `traffic_signal` × PPO
3. `drone_disturbed` × PPO
4. `drone_disturbed` × SAC

*Future extension cells* (2 tests):
5. `navigation_2d` × PPO
6. `navigation_2d` × SAC

Because `lidar_noise` in `navigation_2d` is a future parameter, the tests involving it are pre-registered but not currently executable. The primary family becomes completely executable only after the required environment change is implemented in a future issue.

## 18. Statistical Analysis Plan
### 18.1 Experimental Unit
The independent experimental unit is the **independent training run / seed**. Evaluation episodes are repeated observations nested inside that experimental unit.
### 18.2 Primary Test
An **exact paired permutation (sign-flip) test**. Since $n=10$, we enumerate all $2^{10} = 1024$ possible sign assignments of the paired differences $D_i$. The test statistic is the mean paired difference. The test is **one-sided** consistent with H1 ($\mu_D < 0$).
### 18.3 Effect Size
**Cohen's $d_z$** for paired samples: mean($D_i$) / SD($D_i$). A negative $D_i$ favors the Adaptive treatment.
### 18.4 Confidence Interval
A simpler parametric paired t-interval (95%) on the mean difference is preferred for $n=10$, computed over the independent training seeds (never individual episodes).
### 18.5 Multiple Comparisons
**Correction**: Holm-Bonferroni step-down procedure applied to the explicit Primary Family of 6 tests to control FWER at $\alpha = 0.05$. Secondary exploratory tests remain uncorrected.

## 19. Failure / Invalid / Censored Runs
Categorization:
* **Planned**: 10 training seeds per condition.
* **Completed**: Runs that successfully evaluate without crashing.
* **Failed**: Training divergence, NaNs, or execution crashes.
* **Invalid**: (None. No-degradation runs are now valid and assign $\tau=0$).
* **Right_censored**: Non-recovered by horizon $H$ (handled directly in the primary endpoint as $T_H = H$).

Failed runs do not enter the primary estimand but are reported.
**Sensitivity Analysis**: A pre-registered deterministic sensitivity analysis assigns the worst-case convention $T_H = H$ to any failed treatment run, checking for hidden attrition bias.

## 20. Reporting Requirements
The protocol aligns strictly with the existing ARL repository artifact: the `ShiftBenchmarkReport` JSON schema.
The artifact must contain:
* `experiment_name`, `environment_name`, `algorithm_name`
* `training_provenance` and `policy_fingerprint`
* Detailed `scenario_results` containing per-seed metric records and effective environment parameters.
This protocol requires no external files (like a parallel `manifest.json`) outside the established `ShiftBenchmarkReport` schema.

## 21. Success Criteria
Statistical evidence requires $p < 0.05$ (Holm-adjusted, one-sided) on the primary endpoint. Practical significance is assessed independently via $d_z$.

## 22. Threats to Validity
| Threat | Why it matters | Mitigation | Remaining limitation |
|---|---|---|---|
| **Stochastic training** | High variance obscures effects | Paired evaluation seeds | Small computational budget (10 seeds) limits precision |
| **Algorithm confounding** | Treatment might just be a better base algorithm | Compare Adaptive vs Fixed holding base algorithm identical | Base hyperparameters may favor one condition |
| **Future online implementation** | Treatment is not executable yet | Protocol pre-registered before implementation | Actual adaptation behavior is untested |
| **Shift realism** | Toy shifts don't reflect real-world | Target documented ARL physical parameters | Simulation-to-reality gap remains |
| **Adaptation compute** | Adaptive agent uses more compute post-shift | Declare compute overhead as part of intervention | Deployment latency constraints unmeasured |
| **Censoring** | Non-recovery skews the true continuous mean | Assign explicit max horizon $T_H = H$ for test | Test becomes a truncated rank/mean test |
| **Failure attrition** | Discarding crashes could favor unstable treatments | Deterministic worst-case sensitivity analysis | Cannot recover true behavior of crashed runs |
| **Multiple testing** | Inflated false positives | Holm-Bonferroni correction on exact 6-test primary family | Secondary exploratory tests remain uncorrected |

## 23. Reproducibility Checklist
* Repository SHA recorded in `ShiftBenchmarkReport`.
* YAML configuration files archived.
* All random seeds logged.
* Test seeds proven disjoint from train seeds via existing runner checks.

## 24. Open Research Questions
* How does the adaptation compute overhead scale with observation dimensionality?
* Can representation learning improve the sample efficiency of the online adaptation phase?
