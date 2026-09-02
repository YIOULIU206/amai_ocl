"""Deterministic admission gate for newly diagnosed constraint candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .learning import CandidateDiagnosis, LearningTrace
from .library import (
    ConstraintScope,
    ConstraintStatus,
    FrozenConstraintLibrary,
    SoftConstraint,
)
from .retrieval import normalize_text, tokens


class CandidateGateDecision(str, Enum):
    """Whether a candidate should proceed to expensive rollout validation."""

    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"


class CandidateGateReason(str, Enum):
    """Stable, machine-readable reason for a curation decision."""

    INVALID_CANDIDATE = "invalid_candidate"
    EVIDENCE_NOT_GROUNDED = "evidence_not_grounded"
    EXACT_DUPLICATE = "exact_duplicate"
    SEMANTIC_DUPLICATE = "semantic_duplicate"
    BANK_CONFLICT = "bank_conflict"
    OVERGENERALIZED = "overgeneralized"
    NOVEL_GROUNDED = "novel_grounded"


@dataclass(frozen=True, slots=True)
class CandidateGateResult:
    """Serializable curation result recorded before candidate validation."""

    gate_version: int
    candidate_id: str
    decision: CandidateGateDecision
    reason: CandidateGateReason
    evidence_grounded: bool
    matched_constraint_ids: tuple[str, ...] = ()
    max_similarity: float = 0.0
    label_source: str = "deterministic_rules_v1"


@dataclass(frozen=True, slots=True)
class CandidateCurationGate:
    """Rule-based pre-validation filter for constraint-bank growth.

    This gate intentionally produces weak labels rather than gold labels.  Its
    output can be logged in shadow mode before it is allowed to suppress paired
    rollout validation.
    """

    semantic_duplicate_threshold: float = 0.85
    minimum_general_source_episodes: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.semantic_duplicate_threshold <= 1.0:
            raise ValueError(
                "semantic_duplicate_threshold must be between zero and one"
            )
        if self.minimum_general_source_episodes <= 0:
            raise ValueError(
                "minimum_general_source_episodes must be greater than zero"
            )

    def evaluate(
        self,
        candidate: SoftConstraint,
        diagnosis: CandidateDiagnosis,
        trace: LearningTrace,
        parent_library: FrozenConstraintLibrary,
    ) -> CandidateGateResult:
        """Return a deterministic admission decision for one candidate."""

        candidate_valid = (
            candidate.status is ConstraintStatus.CANDIDATE
            and diagnosis.constraint == candidate
            and trace.split == "derivation"
            and trace.episode_id in candidate.source_episode_ids
            and any(
                step.step_id == diagnosis.earliest_detectable_step
                for step in trace.steps
            )
        )
        if not candidate_valid:
            return self._result(
                candidate,
                CandidateGateDecision.REJECT,
                CandidateGateReason.INVALID_CANDIDATE,
                evidence_grounded=False,
            )

        evidence_grounded = _evidence_is_grounded(diagnosis, trace)
        if not evidence_grounded:
            return self._result(
                candidate,
                CandidateGateDecision.REJECT,
                CandidateGateReason.EVIDENCE_NOT_GROUNDED,
                evidence_grounded=False,
            )

        comparable = tuple(
            constraint
            for constraint in parent_library.approved
            if _action_types_overlap(candidate, constraint)
        )
        exact_matches = tuple(
            constraint.constraint_id
            for constraint in comparable
            if _semantic_signature(constraint) == _semantic_signature(candidate)
        )
        if exact_matches:
            return self._result(
                candidate,
                CandidateGateDecision.REJECT,
                CandidateGateReason.EXACT_DUPLICATE,
                evidence_grounded=True,
                matched_constraint_ids=exact_matches,
                max_similarity=1.0,
            )

        similarities = tuple(
            (_constraint_similarity(candidate, constraint), constraint)
            for constraint in comparable
        )
        max_similarity = max(
            (similarity for similarity, _ in similarities),
            default=0.0,
        )
        near_matches = tuple(
            constraint
            for similarity, constraint in similarities
            if similarity >= self.semantic_duplicate_threshold
        )
        if near_matches:
            conflicting = tuple(
                constraint.constraint_id
                for constraint in near_matches
                if constraint.response is not candidate.response
            )
            if conflicting:
                return self._result(
                    candidate,
                    CandidateGateDecision.DEFER,
                    CandidateGateReason.BANK_CONFLICT,
                    evidence_grounded=True,
                    matched_constraint_ids=conflicting,
                    max_similarity=max_similarity,
                )
            return self._result(
                candidate,
                CandidateGateDecision.REJECT,
                CandidateGateReason.SEMANTIC_DUPLICATE,
                evidence_grounded=True,
                matched_constraint_ids=tuple(
                    constraint.constraint_id for constraint in near_matches
                ),
                max_similarity=max_similarity,
            )

        if (
            candidate.scope is ConstraintScope.GENERAL
            and len(set(candidate.source_episode_ids))
            < self.minimum_general_source_episodes
        ):
            return self._result(
                candidate,
                CandidateGateDecision.DEFER,
                CandidateGateReason.OVERGENERALIZED,
                evidence_grounded=True,
                max_similarity=max_similarity,
            )

        return self._result(
            candidate,
            CandidateGateDecision.ACCEPT,
            CandidateGateReason.NOVEL_GROUNDED,
            evidence_grounded=True,
            max_similarity=max_similarity,
        )

    @staticmethod
    def _result(
        candidate: SoftConstraint,
        decision: CandidateGateDecision,
        reason: CandidateGateReason,
        *,
        evidence_grounded: bool,
        matched_constraint_ids: tuple[str, ...] = (),
        max_similarity: float = 0.0,
    ) -> CandidateGateResult:
        return CandidateGateResult(
            gate_version=1,
            candidate_id=candidate.constraint_id,
            decision=decision,
            reason=reason,
            evidence_grounded=evidence_grounded,
            matched_constraint_ids=matched_constraint_ids,
            max_similarity=round(max_similarity, 6),
        )


def _visible_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            item
            for key in sorted(value)
            for item in _visible_strings(value[key])
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            item
            for value_item in value
            for item in _visible_strings(value_item)
        )
    return ()


def _evidence_is_grounded(
    diagnosis: CandidateDiagnosis,
    trace: LearningTrace,
) -> bool:
    proposed_action_text = tuple(
        normalize_text(value)
        for step in trace.steps
        for value in _visible_strings(step.proposed_action)
        if normalize_text(value)
    )
    return all(
        normalize_text(evidence)
        and any(
            normalize_text(evidence) in visible_text
            for visible_text in proposed_action_text
        )
        for evidence in diagnosis.visible_evidence
    )


def _semantic_signature(constraint: SoftConstraint) -> tuple[Any, ...]:
    return (
        tuple(sorted(constraint.action_types)),
        normalize_text(constraint.trigger_pattern),
        normalize_text(constraint.instruction),
        constraint.response.value,
        normalize_text(constraint.revision_guidance or ""),
    )


def _action_types_overlap(left: SoftConstraint, right: SoftConstraint) -> bool:
    left_types = set(left.action_types)
    right_types = set(right.action_types)
    return "*" in left_types or "*" in right_types or bool(left_types & right_types)


def _constraint_similarity(left: SoftConstraint, right: SoftConstraint) -> float:
    left_tokens = tokens(
        " ".join(
            (
                left.trigger_pattern,
                left.instruction,
                left.revision_guidance or "",
            )
        )
    )
    right_tokens = tokens(
        " ".join(
            (
                right.trigger_pattern,
                right.instruction,
                right.revision_guidance or "",
            )
        )
    )
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)
