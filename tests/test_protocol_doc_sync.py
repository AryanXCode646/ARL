"""Document <-> code sync tests for the preregistered protocol document.

``docs/research/adaptive_rl_hypothesis.md`` is the normative protocol; the
``adaptive_rl.protocol`` package is its executable mirror. This test fails if
either side drifts:

* every frozen literal required by the document must appear verbatim (and be
  computed from the constants, not retyped),
* superseded/incorrect claims banned by the PR review must never reappear,
* a handful of core constants are value-locked against regression.
"""

from __future__ import annotations

from pathlib import Path

from adaptive_rl.protocol import (
    ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    HORIZON,
    K_PRE,
    MIN_DEGRADATION_SE_MULTIPLIER,
    MIN_VALID_N,
    N_UPDATE,
    PINNED_COMMIT,
    PLANNED_N,
    PRIMARY_CELLS,
    PROTOCOL_VERSION,
    SEED_VALUE_MAX,
    TRAINING_SEEDS,
    WINDOW,
    build_schedule,
    schedule_fingerprint,
)

DOC_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "research" / "adaptive_rl_hypothesis.md"
)


def _document() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_core_constants_are_value_locked() -> None:
    assert PROTOCOL_VERSION == "2.0"
    assert PINNED_COMMIT == "faefc5c8e4a39bbcc728d73b1c9855c8e9c5386f"
    assert ALPHA == 0.05
    assert HORIZON == 15
    assert K_PRE == 15
    assert N_UPDATE == 10
    assert MIN_VALID_N == 8
    assert PLANNED_N == 10 == len(TRAINING_SEEDS)
    assert BOOTSTRAP_REPS == 10000
    assert BOOTSTRAP_SEED == 168098
    assert MIN_DEGRADATION_SE_MULTIPLIER == 2.0
    assert SEED_VALUE_MAX == 0x7FFFFFFF
    assert len(PRIMARY_CELLS) == 6


def test_document_contains_frozen_literals() -> None:
    text = _document()
    seeds_literal = "[" + ", ".join(str(seed) for seed in TRAINING_SEEDS) + "]"
    fingerprint = schedule_fingerprint(build_schedule())
    required = [
        f'PROTOCOL_VERSION = "{PROTOCOL_VERSION}"',
        f"TRAINING_SEEDS = {seeds_literal}",
        PINNED_COMMIT,
        f"K_pre = {K_PRE}",
        f"H = {HORIZON}",
        f"N_update = {N_UPDATE}",
        f"WINDOW = {WINDOW}",
        f"MIN_VALID_N = {MIN_VALID_N}",
        f"PLANNED_N = {PLANNED_N}",
        f"ALPHA = {ALPHA}",
        f"BOOTSTRAP_REPS = {BOOTSTRAP_REPS}",
        f"BOOTSTRAP_SEED = {BOOTSTRAP_SEED}",
        f"MIN_DEGRADATION_SE_MULTIPLIER = {MIN_DEGRADATION_SE_MULTIPLIER}",
        "SEED_VALUE_MAX = 0x" + format(SEED_VALUE_MAX, "X"),
        "10.0 * (p_t - p0) >= 9.0 * degradation",
        "delta_min = 2 * SE(delta)",
        'int.from_bytes(digest[:4], byteorder="big", signed=False) & 0x7FFFFFFF',
        'f"{training_seed}|{phase}|{index}"',
        fingerprint,
        'derive_seed(31001, "pre", 1) = 1280372827',
        "TEST-B",
        "intersection-union",
        "recovered",
        "right_censored",
        "no_degradation",
        "degradation_below_resolution",
        "SUPPORTED",
        "NOT_SUPPORTED",
        "INCONCLUSIVE",
        "[5]*5 + [30]*10",
        "[0]*5 + [9]*10",
        "cudnn/deterministic-algorithm flags exist",
        "scenarios[]",
    ]
    for literal in required:
        assert literal in text, f"document is missing required literal: {literal!r}"


def test_document_pins_the_commit_sha() -> None:
    assert PINNED_COMMIT in _document()


def test_document_lists_every_primary_cell() -> None:
    text = _document()
    for cell in PRIMARY_CELLS:
        assert cell in text, f"document does not mention primary cell {cell}"


def test_banned_superseded_claims_never_reappear() -> None:
    """Guard against regressions to claims invalidated by the PR review."""
    text = _document()
    banned = [
        "267d267",  # superseded docs-only PR-branch pin
        "scenario_results",  # real field name is scenarios[]
        "hash(training_seed",  # Python hash() is process-salted
        "hash((training_seed",
        "= 1024",  # old hard-coded sign-flip enumeration size
        "2^{10}",  # ditto
        "t_{0.025, 9}",  # CI critical value hard-coded to full-sample n
        "t_{0.025,9}",  # ditto without space
        "exact paired permutation",  # rejected as primary test (§18.2)
        "future SAC extension",  # drone SAC config exists at the pin
        "Option B is chosen",  # stale editorial leftover
    ]
    for pattern in banned:
        assert pattern not in text, f"document contains banned superseded claim: {pattern!r}"


def test_document_scope_disclaims_are_present() -> None:
    """The document must never claim executability/validation it does not have."""
    text = _document()
    assert "NOT IMPLEMENTED" in text
    assert "Non-Claims" in text or "non-claims" in text
    assert "no result has been collected" in text
