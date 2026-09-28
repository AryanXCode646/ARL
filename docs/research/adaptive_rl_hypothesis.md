# AdaptiveRL Hypothesis and Evaluation Protocol

## 1. Purpose
This document specifies a reproducible, statistically sound experimental protocol for evaluating the claimed adaptive capability of AdaptiveRL agents. It is intended solely as a research specification; no code changes are required.

## 2. Research Question
*Does an AdaptiveRL agent equipped with an online environment‑shift adaptation mechanism recover performance faster after a distribution shift than a non‑adaptive baseline (PPO or SAC) trained on the nominal distribution?*

## 3. Hypothesis
### H0 — Null Hypothesis
After a distribution shift, the adaptive treatment does **not** achieve a statistically significant reduction in recovery time (or increase in normalized recovery) compared with the fixed‑policy baseline under the same training and evaluation budget.

### H1 — Alternative Hypothesis
After a distribution shift, the adaptive treatment achieves a statistically significant **faster** recovery (lower recovery time) and/or higher normalized recovery than the fixed‑policy baseline.

## 4. Operational Definition of Adaptation
* **Pre‑shift performance (P_pre)** – Mean episodic return over the last 20 evaluation episodes **before** the shift.
* **Immediate post‑shift performance (P₀)** – Mean return over the first 5 evaluation episodes **after** the shift.
* **Performance at step *t* (P(t))** – Mean return over a moving window of 5 evaluation episodes centered at timestep *t* after the shift.
* **Recovery metric (R(t))** – Normalized recovery:
```
R(t) = (P(t) - P₀) / (P_pre - P₀)
```
Values range from 0 (no recovery) to 1 (full return to pre‑shift level).
* **Recovery time (τ)** – First timestep where R(t) ≥ 0.9.
* **Final post‑shift performance (P_final)** – Mean return over the last 20 evaluation episodes after a fixed horizon of 10 000 environment steps post‑shift.

## 5. Variables
### 5.1 Independent Variables
| Variable | Levels |
|---|---|
| **Treatment** (primary) | AdaptiveRL (online shift adaptation) vs Fixed‑policy (PPO / SAC) |
| **Environment** (secondary) | GridWorld, ContinuousNavigation, TrafficSignal, Drone3D |
| **Shift type** | Obstacle‑density, Traffic‑arrival‑rate, Wind‑disturbance, Dynamics‑parameter change |
| **Shift magnitude** | Mild, Moderate, Severe (defined per‑environment in §6) |
| **Random seed** | Integer seed controlling environment generation, policy initialization, and any stochastic components |
### 5.2 Dependent Variables
| Metric | Direction (higher is better) |
|---|---|
| **Recovery time (τ)** | lower |
| **Normalized recovery AUC** (integral of R(t) over the adaptation horizon) | higher |
| **Final post‑shift return (P_final)** | higher |
| **Mean episodic return (pre‑shift)** – used only for normalization |
| **Success rate** – proportion of episodes reaching the goal |
### 5.3 Controlled Variables
* Training timestep budget: 200 000 steps for all agents.
* Evaluation episode budget: 40 deterministic episodes per seed (20 pre‑shift, 20 post‑shift).
* Network architecture, optimizer, learning‑rate, and SB3 hyper‑parameters are identical between adaptive and baseline agents.
* Checkpoint selection: Fixed checkpoint at 200 000 steps (no early‑stop based on validation).
* Seeding policy (see §9).

## 6. Experimental Conditions
### 6.1 Training Distribution
All agents are trained on the **nominal** version of each environment (baseline parameters as defined in `configs/*.yaml`).
### 6.2 Distribution Shifts
For each environment we define three shift magnitudes:
* **GridWorld** – obstacle count: nominal = 2, mild = 4, moderate = 6, severe = 8.
* **ContinuousNavigation** – LiDAR noise σ: nominal = 0.01, mild = 0.05, moderate = 0.10, severe = 0.20.
* **TrafficSignal** – arrival‑rate λ per lane: nominal = 0.3, mild = 0.5, moderate = 0.7, severe = 1.0.
* **Drone3D** – wind‑force magnitude: nominal = 0 m/s, mild = 0.5 m/s, moderate = 1.0 m/s, severe = 2.0 m/s.
The shift is applied **after** the pre‑shift evaluation phase; the agent then continues interacting with the altered environment.
### 6.3 Adaptation Window
Agents are allowed the full post‑shift evaluation horizon (10 000 steps) to adapt. No additional training budget is granted beyond the original 200 000 steps.

## 7. Baselines
* **Fixed PPO baseline** – PPO agent trained on the nominal distribution, evaluated under the shifted distribution without any adaptation logic.
* **Fixed SAC baseline** – SAC agent (where the environment has continuous actions) trained nominally, evaluated under shift without adaptation.
* **Adaptive treatment** – Agent that incorporates the AdaptiveRL curriculum‑wrapper that detects distribution‑shift cues (e.g., sudden drop in reward) and updates its policy online using the same optimizer and network as the baseline.
The adaptive implementation is assumed to exist as a configurable wrapper (see `src/adaptive_rl/curriculum/`); the protocol does not require code changes.

## 8. Evaluation Environments
We evaluate *all* four environments because they each expose a distinct shift modality. For each environment we report metrics separately; cross‑environment aggregation is performed on the **normalized recovery AUC**.

## 9. Seed and Reproducibility Protocol
1. **Training seeds** – 10 independent seeds generated by the deterministic rule: `seed_i = 1000 + i` for i∈[0,9].
2. **Evaluation seeds** – For each training seed, a paired evaluation seed is derived as `eval_seed_i = seed_i + 5000`. The same pair is used for both adaptive and baseline agents to eliminate environment variance.
3. All seeds are recorded in the experiment manifest (`manifest.json`).
4. Randomness sources (NumPy, PyTorch, env RNG) are seeded with the same value.
5. Failed runs (crash, NaN metrics) are logged and excluded from the primary analysis but reported in the failure accounting table.

## 10. Training and Evaluation Budget
* **Training budget** – 200 000 environment steps per seed.
* **Evaluation budget** – 40 deterministic episodes (20 pre‑shift, 20 post‑shift) per seed.
* No additional environment interactions are granted to the adaptive agent beyond the evaluation budget; adaptation occurs *online* within the allocated episodes.

## 11. Checkpoint Selection
A single checkpoint saved at the final training timestep (200 000) is used for both adaptive and baseline agents. No validation‑set checkpoint selection is employed to avoid information leakage.

## 12. Statistical Analysis Plan
### 12.1 Experimental Unit
The unit of analysis is the **paired seed‑level outcome** (τ and normalized‑AUC) for each treatment within a given environment and shift magnitude.
### 12.2 Primary Comparison
We conduct a **two‑tailed paired permutation test** (10 000 permutations) comparing adaptive vs baseline recovery time across paired seeds. The test is performed separately for each environment‑shift severity; the pre‑registered primary endpoint is the **average recovery time across environments at the moderate shift level**.
### 12.3 Effect Size
Report **Cohen’s d** for the paired differences.
### 12.4 Confidence Intervals
Bootstrap (5 000 resamples) 95 % confidence intervals for the mean difference.
### 12.5 Multiple Comparisons
Apply the **Holm‑Bonferroni** correction to the family of tests (4 environments × 3 shift magnitudes = 12 comparisons). The pre‑registered primary endpoint (moderate shift) is tested without correction; secondary tests are corrected.
### 12.6 Significance Threshold
Family‑wise α = 0.05.

## 13. Failure and Invalid‑Run Handling
| Status | Definition |
|---|---|
| **Planned runs** | 10 training seeds × 2 treatments × 4 environments × 3 shift levels = 240 seed‑level experiments. |
| **Completed runs** | Runs that finish training, produce a valid checkpoint, and return a non‑NaN evaluation report. |
| **Failed runs** | Crashes, divergence, or NaN/Inf metrics during evaluation. |
| **Invalid runs** | Metric violations (e.g., negative episode length) or missing manifest entries. |
Failed runs are reported but excluded from the primary permutation test; a sensitivity analysis includes them as worst‑case values.

## 14. Reporting Requirements
Each experiment must output a JSON record containing:
* `experiment_id`
* `git_commit_sha`
* `environment`
* `algorithm`
* `treatment`
* `training_seed`
* `evaluation_seed`
* `shift_type` & `shift_magnitude`
* `pre_shift_return`
* `post_shift_return`
* `recovery_time`
* `normalized_recovery_auc`
* `final_return`
* `failure_status`
* Paths to per‑episode report files (`episodes.json`).
All records are aggregated into a top‑level `summary.json` for the full benchmark.

## 15. Pre‑Registered Success Criteria
H1 is considered supported **only if**:
1. The paired permutation test for the primary endpoint (moderate shift) yields **p < 0.05** (Holm‑adjusted where applicable).
2. The 95 % CI for the mean difference in recovery time does **not include zero** and the effect size (Cohen’s d) is ≥ 0.5 (medium).
Both statistical and practical thresholds must be satisfied.

## 16. Negative Results and Inconclusive Results
* **Supports H1** – criteria in §15 met.
* **Fails to reject H0** – p ≥ 0.05 or CI includes zero.
* **Inconclusive** – insufficient valid runs (< 80 % completion) or violation of pre‑registered analysis plan.
All outcomes are reported transparently.

## 17. Threats to Validity
| Threat | Mitigation |
|---|---|
| Stochastic training variance | Use ≥ 10 seeds and paired analysis. |
| Environment stochasticity | Deterministic evaluation seeds; seed‑pairing. |
| Limited seed count | Power analysis justifies 10 seeds; additional seeds can be added later. |
| Algorithm‑specific tuning bias | Identical hyper‑parameters for adaptive and baseline agents. |
| Reward‑scale differences across environments | Primary metric is *recovery time* (a temporal measure) and *normalized AUC* which are scale‑independent. |
| Distribution‑shift realism | Shifts correspond to documented configurable parameters in each environment. |
| Adaptive‑overhead confounding | Adaptation budget is limited to the same evaluation steps; no extra timesteps granted. |
| Checkpoint‑selection bias | Fixed checkpoint at training horizon. |
| Multiple‑testing inflation | Holm correction; primary endpoint pre‑registered. |
| External validity | Results are reported per‑environment; cross‑environment generalisation is assessed only via normalized AUC. |

## 18. Reproducibility Checklist
* Repository SHA recorded in manifest.
* Full YAML configuration files (training, evaluation, shift) archived.
* All random seeds listed.
* Training budget and checkpoint policy documented.
* Metric extraction scripts (`adaptive_rl/evaluation/`) unchanged.
* Artifact files (`summary.json`, per‑seed JSON) uploaded.

## 19. Open Research Questions
* How does the adaptive mechanism scale with higher‑dimensional observation spaces?
* Can meta‑learning improve adaptation speed beyond the simple online update used here?
* What is the impact of curriculum‑based pre‑training on post‑shift recovery?
* How do different adaptation horizons (short vs long) affect the trade‑off between sample efficiency and robustness?
