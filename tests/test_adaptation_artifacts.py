"""Structured artifact serialization and no-overwrite guarantees."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from adaptive_rl.benchmarking.adaptation_artifacts import (
    read_replicate_checkpoint,
    sha256_file,
    validate_study_manifest,
    write_adaptation_artifacts,
    write_adaptive_vs_fixed_artifacts,
    write_replicate_checkpoint,
    write_study_manifest,
)


def _artifact():
    return {
        "schema_version": "1.0",
        "protocol_version": "2.0",
        "issue": "265",
        "experiment": {"algorithm": "ppo", "environment": "drone"},
        "replicates": [
            {
                "training_seed": 31001,
                "shared_pre_shift_episodes": [
                    {
                        "episode_index": 1,
                        "episode_seed": 123,
                        "reward": np.float64(2.0),
                        "length": 15,
                        "success": True,
                        "collision": False,
                        "terminated": True,
                        "truncated": False,
                        "transitions": [{"observation": np.asarray([0.1, 0.2])}],
                    }
                ],
                "shared_shock_episodes": [],
                "adaptive_episodes": [],
                "fixed_episodes": [],
            }
        ],
    }


def test_json_and_flattened_csv_are_written_without_opaque_objects(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(_artifact(), tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["replicates"][0]["shared_pre_shift_episodes"][0]["transitions"][0][
        "observation"
    ] == [0.1, 0.2]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["arm"] == "shared"
    assert rows[0]["phase"] == "pre"
    assert rows[0]["episode_seed"] == "123"


def test_existing_artifacts_are_never_overwritten(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(_artifact(), tmp_path)
    before_json = json_path.read_bytes()
    before_csv = csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_adaptation_artifacts(_artifact(), tmp_path)
    assert json_path.read_bytes() == before_json
    assert csv_path.read_bytes() == before_csv


def test_nonfinite_values_are_rejected_for_strict_json(tmp_path) -> None:
    data = _artifact()
    data["value"] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant"):
        write_adaptation_artifacts(data, tmp_path)


def test_manifest_hashes_all_artifacts_and_detects_tampering(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(
        _artifact(), tmp_path, stem="adaptive_vs_fixed"
    )
    training_artifact = tmp_path / "training" / "seed_31001" / "weights.zip"
    training_artifact.parent.mkdir(parents=True)
    training_artifact.write_bytes(b"real-artifact-bytes")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_study_manifest(
        json_path,
        csv_path,
        manifest_path,
        run_id="test-run",
        command="adaptive-rl benchmark adaptation --study prereg-v1 --run-id test-run",
    )
    assert manifest["artifacts"][json_path.name] == sha256_file(json_path)
    assert manifest["artifacts"][csv_path.name] == sha256_file(csv_path)
    assert manifest["artifacts"]["training/seed_31001/weights.zip"] == sha256_file(
        training_artifact
    )
    validate_study_manifest(manifest_path)
    with pytest.raises(FileExistsError):
        write_study_manifest(
            json_path,
            csv_path,
            manifest_path,
            run_id="test-run",
            command="adaptive-rl benchmark adaptation --study prereg-v1 --run-id test-run",
        )
    training_artifact.write_bytes(b"tampered")
    assert manifest["artifacts"]["training/seed_31001/weights.zip"] != sha256_file(
        training_artifact
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_study_manifest(manifest_path)


def test_study_csv_has_one_row_per_arm_and_preserves_finite_censoring(tmp_path) -> None:
    data = {
        "replicates": [
            {
                "training_seed": 31001,
                "status": "completed",
                "shared_pre_shift_episodes": [{"reward": 10.0}],
                "shared_shock_episodes": [{"reward": 2.0}],
                "adaptive_episodes": [{"reward": 3.0}],
                "fixed_episodes": [{"reward": 2.5}],
                "adaptive_recovery": {"status": "right_censored", "T_H": 15},
                "fixed_recovery": {"status": "right_censored", "T_H": 15},
                "seeds": {"pre": [11], "post": [12, 13], "update": [14]},
            }
        ]
    }
    _, csv_path = write_adaptive_vs_fixed_artifacts(data, tmp_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["arm"] for row in rows} == {"adaptive", "fixed"}
    assert [row["T_H"] for row in rows] == ["15", "15"]
    assert [row["recovery_status"] for row in rows] == ["right_censored"] * 2
    assert json.loads(rows[0]["post_returns"]) == [2.0, 3.0]


def test_replicate_checkpoint_is_terminal_hashed_and_tamper_evident(tmp_path) -> None:
    checkpoint = tmp_path / "replicate_state" / "seed_31001.json"
    write_replicate_checkpoint(
        {"training_seed": 31001, "status": "failed", "failure_reason": "crash"},
        checkpoint,
    )
    assert read_replicate_checkpoint(checkpoint)["failure_reason"] == "crash"
    original = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        write_replicate_checkpoint(
            {"training_seed": 31001, "status": "failed", "failure_reason": "other"},
            checkpoint,
        )
    assert checkpoint.read_bytes() == original
    checkpoint.write_text('{"training_seed":31001,"status":"completed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_replicate_checkpoint(checkpoint)
