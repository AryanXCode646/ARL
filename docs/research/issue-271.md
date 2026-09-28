# Issue #271: preregistered Adaptive vs Fixed study

## Scope and preregistration

This execution targets only `drone_disturbed/ppo` under TEST-B. The normative
protocol remains [`adaptive_rl_hypothesis.md`](adaptive_rl_hypothesis.md); this
document does not change its hypothesis, thresholds, alpha, horizon, seeds, or
analysis plan. The treatment is specified by
[`TREATMENT_CARD.md`](TREATMENT_CARD.md).

## Before implementation: gap analysis

| Requirement | Before | Evidence |
|---|---|---|
| Train once on nominal parameters; share pre-shift and shock; fork and adapt only between episodes | PARTIAL | `src/adaptive_rl/benchmarking/adaptation_runner.py` implemented the Issue #265 lifecycle, but had no Issue #271 invariant record |
| Exact ten-seed schedule and per-block provenance | PARTIAL | `src/adaptive_rl/protocol/seeds.py` froze the schedule; `--training-seeds` still allowed a subset |
| Recovery endpoint and registered tests/sensitivities | EXISTS | `src/adaptive_rl/protocol/recovery.py`, `src/adaptive_rl/protocol/statistics.py`, `src/adaptive_rl/benchmarking/adaptation_statistics.py` |
| Immutable JSON/CSV plus complete checksummed manifest and clean-tree execution gate | PARTIAL | `src/adaptive_rl/benchmarking/adaptation_artifacts.py` refused JSON/CSV overwrite, but there was no manifest or clean-tree gate |
| Required mutation, integrity, and tidy study CSV checks | PARTIAL | Existing Issue #265 tests covered smoke execution; no manifest validation or invariant mutation tests |
| Full real ten-replicate execution and report | MISSING | `docs/research/issue-265.md` stated that no full run had been collected |

## Implementation

The `adaptive-rl benchmark adaptation --study prereg-v1 --run-id RUN_ID`
entrypoint runs all ten training seeds in preregistered order, rejects subsets
and smoke mode, requires a clean committed tree, and writes into an immutable
run directory. A repeated run ID is refused. The JSON stores the raw trajectories,
protocol analysis, seed schedule, outcomes, runtime invariants, and run status.
The CSV has one row per replicate and arm, with finite-horizon `T_H`, status,
per-episode return vectors, and seed vectors. `manifest.json` checksums every
file in the run directory; `validate_study_manifest()` detects missing or
modified files.

The fixed arm has a prediction-only interface. Exact equality of its initial,
per-episode, and final policy fingerprints establishes a zero weight delta; the
artifact records this as `fixed_parameter_delta_l2: 0.0` and
`fixed_weight_update_count: 0`. Adaptive block logs contain their update seed,
visible episode prefix, finite loss metrics, state fingerprints, and parameter
delta. The runner recomputes invariant checks before writing a successful
replicate.

Right-censored recovery remains the preregistered finite endpoint `T_H = 15`.
Failed replicates are kept in the artifact and CSV; they are omitted pairwise
from the primary analysis and retained in both preregistered imputation bounds.
The statistical artifact includes standard error, t statistic, degrees of
freedom, one-sided p-value, 95% interval, Cohen's `d_z`, exact Wilcoxon and sign
tests, bootstrap interval, and failure-imputation bounds.

## Ambiguity resolutions

* The literal training seeds 31001–31010 identify the ten replicates. Each uses
  the existing SHA-256-derived `pre`, `post`, and `update` phase seeds; the frozen
  schedule fingerprint is recorded and rechecked.
* The shared shock window is the single execution of TEST-B post-shift episodes
  1–5, used by both arms before B5. The shared segment is represented once in
  the raw artifact and referenced by identical per-arm return-vector hashes.
* B5 through B14 use the exact completed post-shift episode prefix required by
  `protocol.adaptation.build_update_batch`. No block runs after episode 15.
* Censoring uses the preregistered finite-horizon value 15, not infinity. This
  differs from the issue prompt's parenthetical `T_H = inf`; the preregistration
  is the specified source of truth.

## Dependency and repository status

No local `ExperimentManifest` or roadmap issue 4/5 implementation was found in
the current checkout, so the study uses the minimal manifest described above.
The checked-out base was `fix/pr-259-review-hardening`, one commit ahead and
seven behind `origin/main`; implementation proceeds on
`feat/issue-271-preregistered-study`. GitHub issue/PR pages were unavailable to
the browsing environment, so the live status of PR #266 and roadmap issues 4/5
could not be independently verified. No dependency on unmerged code is used.

## Execution record

Execution status: **PENDING**. Do not interpret smoke tests as study results.

The intended single execution command is:

```bash
.venv/bin/adaptive-rl benchmark adaptation \
  --config configs/drone_distribution_shift.yaml \
  --output-dir artifacts/issue271 \
  --study prereg-v1 \
  --run-id issue271-prereg-v1-20260929-01
```

The runner enables Torch deterministic algorithms in warn-only mode and cuDNN
deterministic settings for the study, while recording that cross-hardware and
cross-library bitwise reproducibility is not claimed. Full run timing, host
details, commit SHA, status, artifact paths, and artifact digests will be added
after the execution attempt.
