"""Environment-independent diagnosis, replay validation, and promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .contracts import DecisionType, ObservableContext, ProposedAction
from .evaluators import ConstraintEvaluator
from .library import ConstraintStatus, FrozenConstraintLibrary, SoftConstraint
from .policies import natural_decision
from .retrieval import ConstraintRetriever


class LearningError(ValueError):
    """Raised when an offline learning artifact violates the protocol."""


@dataclass(frozen=True, slots=True)
class VisibleActionStep:
    step_id: int
    action_type: str
    observable_context: Mapping[str, Any]
    proposed_action: Mapping[str, Any]
    executed: bool
    visible_result: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise LearningError("step_id must be non-negative")
        if not self.action_type.strip():
            raise LearningError("action_type must not be empty")
        object.__setattr__(
            self,
            "observable_context",
            MappingProxyType(dict(self.observable_context)),
        )
        object.__setattr__(
            self,
            "proposed_action",
            MappingProxyType(dict(self.proposed_action)),
        )
        object.__setattr__(
            self,
            "visible_result",
            MappingProxyType(dict(self.visible_result)),
        )


@dataclass(frozen=True, slots=True)
class LearningTrace:
    episode_id: str
    scenario_id: str
    split: str
    steps: tuple[VisibleActionStep, ...]
    visible_outcome: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in {"derivation", "validation", "evaluation", "benign"}:
            raise LearningError(f"invalid trace split: {self.split}")
        if not self.steps:
            raise LearningError("learning trace must contain at least one action step")
        object.__setattr__(self, "visible_outcome", MappingProxyType(dict(self.visible_outcome)))


@dataclass(frozen=True, slots=True)
class OutcomeLabel:
    episode_id: str
    policy_failure: bool
    safe_handling: bool
    false_positive_intervention: bool
    task_progress: bool
    evidence_step_ids: tuple[int, ...] = ()
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class CandidateDiagnosis:
    constraint: SoftConstraint
    earliest_detectable_step: int
    visible_evidence: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if self.constraint.status is not ConstraintStatus.CANDIDATE:
            raise LearningError("diagnosis must contain a candidate constraint")
        if self.earliest_detectable_step < 0:
            raise LearningError("earliest_detectable_step must be non-negative")
        if not self.visible_evidence:
            raise LearningError("visible_evidence must not be empty")


class ConstraintDiagnoser(Protocol):
    def diagnose(
        self,
        trace: LearningTrace,
        label: OutcomeLabel,
    ) -> CandidateDiagnosis: ...


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class PromptedConstraintDiagnoser:
    """Strict-JSON Meta-Agent adapter with no dependency on a model provider."""

    generator: TextGenerator

    def diagnose(
        self,
        trace: LearningTrace,
        label: OutcomeLabel,
    ) -> CandidateDiagnosis:
        if trace.split != "derivation":
            raise LearningError("candidate generation is restricted to derivation traces")
        if trace.episode_id != label.episode_id:
            raise LearningError("trace and label episode IDs do not match")
        if not label.policy_failure:
            raise LearningError("candidate generation requires a labeled policy failure")
        prompt = self._build_prompt(trace, label)
        raw = self.generator.generate(prompt)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LearningError(f"diagnoser did not return strict JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LearningError("diagnoser output must be a JSON object")
        try:
            constraint = SoftConstraint.from_dict(
                {
                    "constraint_id": payload["constraint_id"],
                    "action_types": payload.get(
                        "action_types",
                        sorted({step.action_type for step in trace.steps}),
                    ),
                    "tactic_type": payload["tactic_type"],
                    "trigger_pattern": payload["trigger_pattern"],
                    "keywords": payload.get("keywords", []),
                    "instruction": payload["instruction"],
                    "response": payload["response"],
                    "status": "candidate",
                    "source_episode_ids": [trace.episode_id],
                    "metadata": {"scenario_id": trace.scenario_id},
                }
            )
            return CandidateDiagnosis(
                constraint=constraint,
                earliest_detectable_step=int(payload["earliest_detectable_step"]),
                visible_evidence=tuple(payload["visible_evidence"]),
                rationale=str(payload.get("rationale", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LearningError(f"invalid diagnoser output: {exc}") from exc

    @staticmethod
    def _build_prompt(trace: LearningTrace, label: OutcomeLabel) -> str:
        steps = [
            {
                "step_id": step.step_id,
                "action_type": step.action_type,
                "observable_context": dict(step.observable_context),
                "proposed_action": dict(step.proposed_action),
                "executed": step.executed,
                "visible_result": dict(step.visible_result),
            }
            for step in trace.steps
        ]
        payload = {
            "episode_id": trace.episode_id,
            "scenario_id": trace.scenario_id,
            "steps": steps,
            "outcome_label": {
                "policy_failure": label.policy_failure,
                "evidence_step_ids": label.evidence_step_ids,
                "rationale": label.rationale,
            },
        }
        return (
            "Diagnose the earliest observable cause of this policy failure. "
            "Return one strict JSON object with keys constraint_id, action_types, "
            "tactic_type, trigger_pattern, keywords, instruction, response "
            "(warn|revise|block|escalate), earliest_detectable_step, "
            "visible_evidence, and rationale. Generalize beyond exact wording and "
            "do not use hidden state or profile labels.\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )


@dataclass(frozen=True, slots=True)
class ReplayCase:
    case_id: str
    split: str
    action: ProposedAction
    context: ObservableContext
    should_intervene: bool
    source_episode_id: str | None = None

    def __post_init__(self) -> None:
        if self.split not in {"validation", "benign"}:
            raise LearningError("replay cases must come from validation or benign splits")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    candidate_id: str
    cases: int
    positive_cases: int
    negative_cases: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    recall: float
    precision: float
    false_positive_rate: float
    case_results: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CandidateValidator:
    retriever: ConstraintRetriever
    evaluator: ConstraintEvaluator

    def validate(
        self,
        candidate: SoftConstraint,
        cases: Sequence[ReplayCase],
    ) -> ValidationReport:
        if candidate.status is not ConstraintStatus.CANDIDATE:
            raise LearningError("only candidate constraints may be validated")
        if not cases:
            raise LearningError("validation requires replay cases")
        positive_cases = sum(case.should_intervene for case in cases)
        negative_cases = len(cases) - positive_cases
        if positive_cases == 0 or negative_cases == 0:
            raise LearningError("validation requires both positive and negative cases")
        source_ids = set(candidate.source_episode_ids)
        if any(case.source_episode_id in source_ids for case in cases if case.source_episode_id):
            raise LearningError("source episodes cannot validate their own candidate")

        approved_for_replay = candidate.approved_copy(metadata={"validation_only": True})
        library = FrozenConstraintLibrary((approved_for_replay,))
        tp = fp = tn = fn = 0
        details: list[Mapping[str, Any]] = []
        for case in cases:
            retrieved = tuple(self.retriever.retrieve(case.action, case.context, library))
            checks = tuple(self.evaluator.evaluate(case.action, case.context, retrieved))
            decisions = [natural_decision(check) for check in checks]
            intervened = any(decision is not DecisionType.APPROVE for decision in decisions)
            if case.should_intervene and intervened:
                tp += 1
            elif case.should_intervene:
                fn += 1
            elif intervened:
                fp += 1
            else:
                tn += 1
            details.append(
                MappingProxyType(
                    {
                        "case_id": case.case_id,
                        "expected_intervention": case.should_intervene,
                        "intervened": intervened,
                        "retrieved": bool(retrieved),
                        "check_ids": tuple(check.check_id for check in checks),
                    }
                )
            )
        recall = tp / positive_cases
        precision = tp / max(1, tp + fp)
        false_positive_rate = fp / negative_cases
        return ValidationReport(
            candidate_id=candidate.constraint_id,
            cases=len(cases),
            positive_cases=positive_cases,
            negative_cases=negative_cases,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            recall=recall,
            precision=precision,
            false_positive_rate=false_positive_rate,
            case_results=tuple(details),
        )


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    minimum_positive_cases: int = 1
    minimum_negative_cases: int = 1
    minimum_recall: float = 0.5
    minimum_precision: float = 0.5
    maximum_false_positive_rate: float = 0.1

    def reasons(self, report: ValidationReport) -> tuple[str, ...]:
        failures: list[str] = []
        if report.positive_cases < self.minimum_positive_cases:
            failures.append("insufficient positive cases")
        if report.negative_cases < self.minimum_negative_cases:
            failures.append("insufficient negative cases")
        if report.recall < self.minimum_recall:
            failures.append("recall below threshold")
        if report.precision < self.minimum_precision:
            failures.append("precision below threshold")
        if report.false_positive_rate > self.maximum_false_positive_rate:
            failures.append("false-positive rate above threshold")
        return tuple(failures)

    def approves(self, report: ValidationReport) -> bool:
        return not self.reasons(report)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    approved: bool
    constraint: SoftConstraint
    reasons: tuple[str, ...]
    report: ValidationReport


def promote_candidate(
    candidate: SoftConstraint,
    report: ValidationReport,
    policy: PromotionPolicy,
) -> PromotionResult:
    if candidate.constraint_id != report.candidate_id:
        raise LearningError("candidate and validation report IDs do not match")
    reasons = policy.reasons(report)
    approved = not reasons
    if approved:
        constraint = candidate.approved_copy(
            metadata={
                "validation_cases": report.cases,
                "validation_recall": report.recall,
                "validation_precision": report.precision,
                "validation_false_positive_rate": report.false_positive_rate,
            }
        )
    else:
        constraint = SoftConstraint(
            constraint_id=candidate.constraint_id,
            action_types=candidate.action_types,
            tactic_type=candidate.tactic_type,
            trigger_pattern=candidate.trigger_pattern,
            keywords=candidate.keywords,
            instruction=candidate.instruction,
            response=candidate.response,
            status=ConstraintStatus.REJECTED,
            source_episode_ids=candidate.source_episode_ids,
            metadata={**dict(candidate.metadata), "rejection_reasons": reasons},
        )
    return PromotionResult(
        approved=approved,
        constraint=constraint,
        reasons=reasons,
        report=report,
    )
