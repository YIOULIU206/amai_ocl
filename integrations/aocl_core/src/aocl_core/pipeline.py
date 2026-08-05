"""One-call orchestration of the offline adaptive constraint loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .learning import (
    CandidateDiagnosis,
    CandidateValidator,
    ConstraintDiagnoser,
    LearningTrace,
    OutcomeLabel,
    PromotionPolicy,
    PromotionResult,
    ReplayCase,
    ValidationReport,
    promote_candidate,
)
from .versioning import LibraryVersion, VersionedLibraryStore


@dataclass(frozen=True, slots=True)
class AdaptiveLearningResult:
    diagnosis: CandidateDiagnosis
    validation: ValidationReport
    promotion: PromotionResult
    library_version: LibraryVersion | None


@dataclass(frozen=True, slots=True)
class AdaptiveLearningPipeline:
    diagnoser: ConstraintDiagnoser
    validator: CandidateValidator
    promotion_policy: PromotionPolicy
    library_store: VersionedLibraryStore

    def process_failure(
        self,
        *,
        trace: LearningTrace,
        label: OutcomeLabel,
        replay_cases: Sequence[ReplayCase],
        parent_version: LibraryVersion,
        child_version_id: str,
    ) -> AdaptiveLearningResult:
        diagnosis = self.diagnoser.diagnose(trace, label)
        validation = self.validator.validate(diagnosis.constraint, replay_cases)
        promotion = promote_candidate(
            diagnosis.constraint,
            validation,
            self.promotion_policy,
        )
        child = None
        if promotion.approved:
            child = self.library_store.promote(
                parent=parent_version,
                result=promotion,
                policy=self.promotion_policy,
                version_id=child_version_id,
            )
        return AdaptiveLearningResult(
            diagnosis=diagnosis,
            validation=validation,
            promotion=promotion,
            library_version=child,
        )
