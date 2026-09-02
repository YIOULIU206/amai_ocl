"""Permanent integrity tests for the fresh Gate holdout protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agenticpay_ocl_v2.batch_experiment import _tactic_ids
from agenticpay_ocl_v2.datasets import SplitManifest, load_profiles


ROOT = Path(__file__).resolve().parents[3]
ATTACK_PATH = ROOT / "configs" / "adversarial_buyers.json"
BENIGN_PATH = ROOT / "integrations/agenticpay_ocl_v2/data/benign_buyers.json"
MANIFEST_PATH = ROOT / "integrations/agenticpay_ocl_v2/configs/split_manifest_gate_holdout_v1.json"
PROTOCOL_PATH = ROOT / "integrations/agenticpay_ocl_v2/configs/gate_holdout_protocol_v1.json"
INSTRUCTION_PATH = ROOT / "integrations/agenticpay_ocl_v2/configs/candidate_authoring_v1.md"
VALIDATION_MANIFEST_PATH = ROOT / "integrations/agenticpay_ocl_v2/configs/split_manifest_gate_holdout_validation_v1.json"


def _raw_profiles() -> tuple[dict[str, object], ...]:
    return tuple(
        item
        for path in (ATTACK_PATH, BENIGN_PATH)
        for item in json.loads(path.read_text(encoding="utf-8"))
    )


def _canonical_hash(
    profiles: tuple[dict[str, object], ...],
    profile_ids: tuple[str, ...],
) -> str:
    by_id = {str(item.get("profile_id")): item for item in profiles}
    payload = [by_id[profile_id] for profile_id in profile_ids]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_gate_holdout_profiles_and_manifest_are_frozen() -> None:
    profiles = load_profiles(ATTACK_PATH) + load_profiles(BENIGN_PATH)
    raw_profiles = _raw_profiles()
    manifest = SplitManifest.from_json(MANIFEST_PATH)
    manifest.validate(profiles)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    attack_ids = tuple(manifest.validation + manifest.evaluation)
    benign_ids = tuple(manifest.benign)

    assert attack_ids == tuple(
        f"privacy_phisher_holdout_v1_{index:03d}" for index in range(1, 7)
    )
    assert benign_ids == tuple(
        f"benign_buyer_holdout_v1_{index:03d}" for index in range(1, 7)
    )
    assert len(set(attack_ids)) == 6
    assert len(set(benign_ids)) == 6
    assert set(attack_ids).isdisjoint(benign_ids)
    assert manifest.derivation == tuple(
        f"privacy_phisher_{index:03d}" for index in range(1, 5)
    )
    assert manifest.validation == attack_ids[:2]
    assert manifest.evaluation == attack_ids[2:]
    assert manifest.benign[:2] == benign_ids[:2]
    assert manifest.benign[2:] == benign_ids[2:]
    all_splits = tuple(
        tuple(getattr(manifest, name))
        for name in ("derivation", "validation", "evaluation", "benign", "controls")
    )
    assert sum(len(items) for items in all_splits) == len(
        {profile_id for items in all_splits for profile_id in items}
    )
    assert _tactic_ids(manifest.validation, "privacy_phisher", 2, "validation") == manifest.validation
    assert _tactic_ids(manifest.evaluation, "privacy_phisher", 4, "evaluation") == manifest.evaluation
    assert _tactic_ids(manifest.evaluation, "privacy_phisher", 0, "evaluation") == ()
    assert _canonical_hash(raw_profiles, attack_ids) == protocol["attack_holdout_profiles_sha256"]
    assert _canonical_hash(raw_profiles, benign_ids) == protocol["benign_holdout_profiles_sha256"]
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == protocol["split_manifest_file_sha256"]
    assert protocol["validation_stage_evaluation_limit"] == 1
    assert protocol["validation_stage_evaluation_profile_count"] == 2
    assert protocol["validation_stage_evaluation_attack_ids"] == [
        "privacy_phisher_007"
    ]
    assert protocol["validation_stage_evaluation_benign_ids"] == [
        "benign_buyer_002"
    ]
    assert protocol["validation_stage_evaluation_is_contaminated_sentinel"] is True
    assert protocol["validation_stage_evaluation_metrics_are_holdout_evidence"] is False
    assert protocol["validation_stage_evaluation_must_not_update_bank"] is True
    assert protocol["final_stage_evaluation_limit"] == 4
    assert protocol["final_profiles_must_not_execute_during_validation_stage"] is True
    assert protocol["validation_manifest_file"] == str(VALIDATION_MANIFEST_PATH.relative_to(ROOT))
    assert hashlib.sha256(VALIDATION_MANIFEST_PATH.read_bytes()).hexdigest() == protocol["validation_manifest_file_sha256"]


def test_validation_only_manifest_excludes_final_test_profiles() -> None:
    profiles = load_profiles(ATTACK_PATH) + load_profiles(BENIGN_PATH)
    manifest = SplitManifest.from_json(VALIDATION_MANIFEST_PATH)
    manifest.validate(profiles)
    assert manifest.evaluation == ("privacy_phisher_007",)
    assert _tactic_ids(manifest.evaluation, "privacy_phisher", 1, "evaluation") == (
        "privacy_phisher_007",
    )
    assert manifest.benign == (
        "benign_buyer_holdout_v1_001",
        "benign_buyer_holdout_v1_002",
        "benign_buyer_002",
    )
    assert manifest.benign[:2] == (
        "benign_buyer_holdout_v1_001",
        "benign_buyer_holdout_v1_002",
    )
    assert manifest.benign[2:] == ("benign_buyer_002",)
    assert len(manifest.evaluation) == 1
    assert len(manifest.benign[2:]) == 1
    assert not set(manifest.validation) & {
        f"privacy_phisher_holdout_v1_{index:03d}" for index in range(3, 7)
    }
    assert not set(manifest.benign) & {
        f"benign_buyer_holdout_v1_{index:03d}" for index in range(3, 7)
    }


def test_holdout_ids_are_not_in_visible_self_fields_or_repeated_descriptions() -> None:
    profiles = _raw_profiles()
    holdout_profiles = [
        item
        for item in profiles
        if str(item.get("profile_id", "")).startswith(
            ("privacy_phisher_holdout_v1_", "benign_buyer_holdout_v1_")
        )
    ]
    assert len(holdout_profiles) == 12
    assert len({str(item["profile_id"]) for item in holdout_profiles}) == 12
    assert len({str(item["description"]) for item in holdout_profiles}) == 12
    for item in holdout_profiles:
        profile_id = str(item["profile_id"])
        assert profile_id not in str(item.get("name", ""))
        assert profile_id not in str(item.get("description", ""))


def test_candidate_authoring_instruction_hash_is_unchanged() -> None:
    content = INSTRUCTION_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "eecfb82c80cdad88f3f9712ada5a9330959e63c1bbec83377a5bce885adc2b49"
    )
    assert hashlib.sha256(content.rstrip()).hexdigest() == (
        "a26e5e1204abe35ff3797e053715130c4073d1d13994eb7fa9e77d627e50729d"
    )
