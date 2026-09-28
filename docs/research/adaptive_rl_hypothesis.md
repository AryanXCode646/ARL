# AdaptiveRL Hypothesis and Evaluation Protocol

## 1. Purpose
This document specifies an auditable, pre-specified experimental protocol for evaluating a proposed adaptive capability for RL agents.

> [!WARNING]
> **Implementation Status**
> The AdaptiveRL treatment described in this document (online shift adaptation) is a pre-registered experimental specification for a **future** AdaptiveRL implementation. This document does not imply that the complete treatment is currently implemented in the repository. The existing benchmark infrastructure enforces a strictly frozen policy during evaluation.

## 2. Research Question
Does an agent equipped with a future online environment-shift adaptation treatment recover performance faster after a distribution shift than a matched fixed-policy baseline (PPO or SAC) trained on the same nominal distribution?

## 3. Hypotheses
**Primary endpoint:** Pre-registered recovery time under the moderate shift condition for each environment independently.

### H0 — Null Hypothesis
The AdaptiveRL treatment has no difference from the matched fixed baseline on the primary endpoint. (The mean paired difference in recovery time is zero).

### H1 — Alternative Hypothesis
The AdaptiveRL treatment has a lower recovery time than the matched fixed baseline on the primary endpoint.

## 4. Treatment Definition
* **Algorithm**: The base learning algorithm (e.g., PPO or SAC).
* **Treatment**: Adaptive vs Fixed.

Where supported, we compare Adaptive PPO vs Fixed PPO, and Adaptive SAC vs Fixed SAC. The base algorithm and hyperparameters remain identical; the intervention is strictly the presence of the online adaptation mechanism and its associated compute overhead during the post-shift phase.

## 5. Operational Definition of Adaptation
**Time axis**: **Completed post-shift episode index**. All recovery definitions are evaluated entirely using this single canonical axis. A second researcher must be able to reconstruct the same recovery time from raw episode records.

* **Pre-shift performance ($P_{pre}$)**: Mean episodic return over the 20 pre-shift evaluation episodes.
* **Immediate post-shift performance ($P_0$)**: Mean return over the first 5 evaluation episodes after shift introduction.
* **Performance at episode $t$ ($P(t)$)**: Mean return over a **causal trailing window** of 5 evaluation episodes ending at index $t$. The first valid measurement occurs at $t=5$. A future episode is never used to declare recovery at an earlier time.
* **Recovery metric ($R(t)$)**: 
  $$ R(t) = \frac{P(t) - P_0}{P_{pre} - P_0} $$
  *Edge-case handling*: If $P_{pre} \le P_0$ (no post-shift degradation), the recovery time is undefined (invalid metric). If $P(t) < P_0$, $R(t)$ can be negative. If $P(t) > P_{pre}$, $R(t) > 1$.
* **Recovery time ($\tau$)**: The first episode index $t$ where $R(t) \ge 0.9$ for at least 3 consecutive episodes (persistence rule).
* **Non-recovery and Censoring**: If the condition $R(t) \ge 0.9$ is not met by the maximum evaluation horizon $H$, the run is designated as **right-censored**. For the primary permutation test, right-censored values are imputed as the maximum horizon $H$.

## 6. Independent Variables
| Variable | Levels |
|---|---|
| **Algorithm** | PPO, SAC |
| **Treatment** | Adaptive vs Fixed |
| **Environment** | `gridworld`, `navigation_2d`, `traffic_signal`, `drone_disturbed` |
| **Shift magnitude** | Mild, Moderate, Severe |

## 7. Dependent Variables
* **Primary**: Recovery time ($\tau$) (lower is better).
* **Secondary**: Final post-adaptation return, success rate.

## 8. Controlled Variables
* Identical training timestep budget.
* Fixed checkpoint selection (final training checkpoint).
* Shared evaluation seeds for paired testing.

## 9. Baselines
* **Fixed PPO/SAC baseline**: Trained on nominal distribution, evaluated under shifted distribution with a frozen policy.

## 10. Evaluation Environments
Environments verified in the ARL registry:
* `gridworld` (Action space: Discrete, Algorithm: PPO)
* `navigation_2d` (Action space: Continuous, Algorithm: PPO/SAC)
* `traffic_signal` (Action space: Discrete, Algorithm: PPO)
* `drone_disturbed` (Action space: Continuous, Algorithm: PPO/SAC)

## 11. Training Distribution
All agents train exclusively on the nominal configuration of the environment as defined in the training configs. The training parameters ($P_{train}$) are strictly separated from the shift parameters ($P_{shift}$). Test seeds and shift parameters must not leak into the training phase. Post-hoc scenario selection is strictly prohibited.

## 12. Distribution Shifts
Shift parameters target verified configuration endpoints.
* **`gridworld`** — `num_obstacles`: nominal = 2, mild = 4, moderate = 6, severe = 8.
* **`navigation_2d`** — `lidar_noise`: *[FUTURE PROTOCOL PARAMETER — NOT CURRENTLY IMPLEMENTED IN ARL]*.
* **`traffic_signal`** — `arrival_rates` (tuple applied to all lanes): nominal = (0.3, 0.3, 0.3, 0.3), mild = (0.5, 0.5, 0.5, 0.5), moderate = (0.7, 0.7, 0.7, 0.7), severe = (1.0, 1.0, 1.0, 1.0).
* **`drone_disturbed`** — `wind_speed` (m/s) and `gust_sigma`: nominal = (0.5, 0.15), mild = (2.0, 0.3), moderate = (4.0, 0.6), severe = (8.0, 1.2).

## 13. Evaluation Procedure
The evaluation separates cleanly into distinct phases:
1. **Pre-shift measurement**: 20 episodes on nominal parameters.
2. **Shift introduction**: Environment parameters are updated.
3. **Adaptation period**: Agent interacts with the shifted environment (online updates enabled for Adaptive treatment, policy frozen for Baseline).
4. **Post-adaptation measurement**: Final measurement of adapted performance.

## 14. Seed Protocol
* **Training seeds**: 10 independent experimental replicates. This seed count represents a pre-registered computational budget, not a powered sample size from a formal power analysis.
* **Evaluation seeds**: For each training seed, a paired evaluation seed is used to control environment variance across treatments. 
* Randomness for NumPy, PyTorch, and environment initialization must be seeded deterministically.

## 15. Training / Adaptation / Evaluation Budget
* **Training interactions**: Fixed budget (e.g., 200,000 steps).
* **Adaptation budget**: The Adaptive treatment incurs online computation overhead during the post-shift episodes. This extra computation is an explicit part of the intervention; the baseline receives no such computation. Both methods are evaluated on the exact same budget of environment interactions.

## 16. Checkpoint Selection
Checkpoint selection uses a deterministic pre-declared rule: the **final training checkpoint** at the end of the training budget. Post-shift evaluation performance is never used to select checkpoints. The same selection rule applies to all conditions.

## 17. Primary Endpoint
The single primary endpoint is the **recovery time under the moderate shift condition**, analyzed within each environment. Cross-environment aggregation of raw recovery times is not performed due to scale differences.

## 18. Secondary Endpoints
* Final post-adaptation return.
* Recovery time under mild and severe shifts.

## 19. Statistical Analysis Plan
### 19.1 Experimental Unit
The independent experimental unit is the **independent training run / seed**. Evaluation episodes are repeated observations nested inside that experimental unit.
### 19.2 Estimand
The paired difference in recovery time between the Adaptive treatment and the Fixed baseline for a given training seed.
### 19.3 Primary Test
A **Monte Carlo paired permutation (sign-flip) test** (10,000 sign-flips) at the seed level. The test statistic is the mean paired difference. The test is two-sided.
### 19.4 Effect Size
**Cohen's $d_z$** for paired samples (mean of paired differences divided by the standard deviation of paired differences).
### 19.5 Confidence Interval
95% Bias-Corrected and Accelerated (BCa) bootstrap interval, resampling the paired seed differences (5,000 resamples). Resampling strictly occurs at the seed level, not the episode level.
### 19.6 Multiple Comparisons
**Family**: The set of 4 environments evaluated at the moderate shift level.
**Correction**: Holm-Bonferroni step-down procedure applied to this primary family of 4 tests to control the Family-Wise Error Rate (FWER) at $\alpha = 0.05$.

## 20. Failure / Invalid / Censored Runs
Categorization:
* **Planned**: 10 training seeds per condition.
* **Completed**: Runs that successfully evaluate without crashing.
* **Failed**: Training divergence, NaNs, or execution crashes.
* **Invalid**: Metric undefined (e.g., $P_{pre} \le P_0$).
* **Censored**: Runs that do not recover within the adaptation horizon.

Failed and invalid runs are excluded from the primary test but reported. A pre-registered sensitivity analysis assigns worst-case values to failed treatment runs to check for hidden attrition bias.

## 21. Reporting Requirements
The protocol aligns with the existing ARL `ShiftBenchmarkReport` JSON schema.
The benchmark artifact must contain:
* `experiment_name`, `environment_name`, `algorithm_name`
* `training_provenance` and `policy_fingerprint`
* Detailed `scenario_results` containing per-seed metric records and environment parameters.
Do not assume external reporting infrastructure (e.g., parallel `manifest.json`) exists unless implemented in the `ShiftBenchmarkReport`.

## 22. Success Criteria
Statistical evidence requires $p < 0.05$ (Holm-adjusted) on the primary endpoint. Practical significance is assessed independently by evaluating if the effect size $d_z$ is meaningful for the deployment context.

## 23. Negative and Inconclusive Results
Transparently reported. If completion rate is < 80% due to failures, the result is considered inconclusive.

## 24. Threats to Validity
| Threat | Why it matters | Mitigation | Remaining limitation |
|---|---|---|---|
| **Stochastic training** | High variance obscures effects | Paired evaluation seeds | Limited computational budget (10 seeds) limits precision |
| **Algorithm confounding** | Treatment might just be a better base algorithm | Compare Adaptive PPO vs Fixed PPO | Base hyperparameters may favor one condition |
| **Shift realism** | Toy shifts don't reflect real-world | Target documented physical parameters | Simulation-to-reality gap remains |
| **Adaptation compute** | Adaptive agent uses more compute post-shift | Declare overhead as part of intervention | Deployment latency constraints unmeasured |
| **Cross-environment** | Aggregating raw recovery times is invalid | Analyze environments independently | Lack of a unified global metric |
| **Censoring** | Non-recovery skews the mean | Impute max horizon $H$ for test | Test becomes conservative |
| **Multiple testing** | Inflated false positives | Holm-Bonferroni correction on primary family | Secondary exploratory tests remain uncorrected |

## 25. Reproducibility Checklist
* Repository SHA recorded in report.
* YAML configuration files archived.
* All random seeds logged.
* Test seeds proven disjoint from train seeds.
* `ShiftBenchmarkReport` JSON output uploaded.

## 26. Open Research Questions
* How does the adaptation overhead scale with observation dimensionality?
* Can representation learning improve the sample efficiency of the online adaptation phase?
