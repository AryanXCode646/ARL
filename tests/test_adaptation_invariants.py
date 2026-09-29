"""Mutation checks for prereg-v1 runtime invariant auditing."""

from __future__ import annotations

from copy import deepcopy

import pytest

from adaptive_rl.benchmarking.adaptation_runner import (
    _audit_replicate_invariants,
    _restore_replicate,
    _return_vector_fingerprint,
)
from adaptive_rl.protocol.constants import TRAINING_SEEDS
from adaptive_rl.protocol.seeds import frozen_schedule


@pytest.fixture
def valid_replicate():
    schedule = frozen_schedule()
    seed = TRAINING_SEEDS[0]
    fingerprint = "frozen"
    pre = [{"reward": 2.0} for _ in range(15)]
    shock = [{"reward": 0.0} for _ in range(5)]
    pre_hash = _return_vector_fingerprint(pre)
    shock_hash = _return_vector_fingerprint(shock)
    replicate = {
        "training_seed": seed,
        "shared_pre_shift_episodes": pre,
        "shared_shock_episodes": shock,
        "adaptive_episodes": [{"update_block": block} for block in range(5, 15)],
        "fixed_episodes": [
            {"policy_fingerprint_start": fingerprint, "policy_fingerprint_end": fingerprint}
            for _ in range(10)
        ],
        "update_blocks": [
            {
                "block_episode": block,
                "visible_episode_indices": list(range(1, block + 1)),
                "update_seed": schedule[seed]["update"][block - 5],
                "loss_metrics": {"train/loss": 1.0},
            }
            for block in range(5, 15)
        ],
        "seeds": schedule[seed],
        "arm_input_fingerprints": {
            arm: {"pre": pre_hash, "shock": shock_hash} for arm in ("adaptive", "fixed")
        },
        "fixed_weight_update_count": 0,
        "fixed_parameter_delta_l2": 0.0,
        "fixed_final_fingerprint": fingerprint,
        "frozen_fingerprint": fingerprint,
        "fork_fingerprint": fingerprint,
    }
    return replicate, schedule


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("pre", "pre_shift_identical_across_arms"),
        ("shock", "shock_returns_identical_across_arms"),
        ("fixed", "fixed_zero_updates"),
        ("leak", "no_future_data"),
        ("fork", "fork_fingerprint_identical"),
    ],
)
def test_each_preregistered_invariant_violation_fails_audit(
    valid_replicate, mutation, failed_check
):
    replicate, schedule = valid_replicate
    replicate = deepcopy(replicate)
    if mutation == "pre":
        replicate["arm_input_fingerprints"]["fixed"]["pre"] = "different"
    elif mutation == "shock":
        replicate["arm_input_fingerprints"]["adaptive"]["shock"] = "different"
    elif mutation == "fixed":
        replicate["fixed_weight_update_count"] = 1
    elif mutation == "leak":
        replicate["update_blocks"][0]["visible_episode_indices"].append(6)
    else:
        replicate["fork_fingerprint"] = "different"

    audit = _audit_replicate_invariants(replicate, schedule)
    assert audit[failed_check] is False
    assert audit["all_passed"] is False


def test_replicate_restore_reconstructs_terminal_state() -> None:
    restored = _restore_replicate(
        {
            "training_seed": 31001,
            "status": "failed",
            "failure_reason": "recorded failure",
        }
    )
    assert restored.training_seed == 31001
    assert restored.status == "failed"
    assert restored.failure_reason == "recorded failure"
