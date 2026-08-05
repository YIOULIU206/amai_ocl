"""Online control runtime; evaluates proposals but never executes them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .audit import AuditEvent, AuditSink, InMemoryAuditSink
from .contracts import (
    CheckLevel,
    CheckResult,
    ControlDecision,
    DecisionType,
    ObservableContext,
    ObservedOutcome,
    ProposedAction,
)
from .evaluators import ConstraintEvaluator
from .library import FrozenConstraintLibrary
from .policies import ControlMode, aggregate_checks
from .retrieval import ConstraintRetriever
from .validators import Validator


class AuditFailure(RuntimeError):
    """Raised when strict audit recording fails."""


@dataclass(slots=True)
class IntegrationOCLRuntime:
    mode: ControlMode = ControlMode.BLOCKING
    validators: Sequence[Validator] = ()
    constraint_library: FrozenConstraintLibrary | None = None
    retriever: ConstraintRetriever | None = None
    constraint_evaluator: ConstraintEvaluator | None = None
    audit_sink: AuditSink = field(default_factory=InMemoryAuditSink)
    strict_audit: bool = False
    _evaluated_action_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _audit_failures: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ControlMode):
            self.mode = ControlMode(self.mode)
        if (self.retriever is None) != (self.constraint_evaluator is None):
            raise ValueError("retriever and constraint_evaluator must be configured together")

    @property
    def audit_failures(self) -> tuple[str, ...]:
        return tuple(self._audit_failures)

    def evaluate(
        self,
        *,
        action: ProposedAction,
        context: ObservableContext,
    ) -> ControlDecision:
        if action.actor_id != context.actor_id:
            raise ValueError("action.actor_id must match context.actor_id")
        common = {
            "episode_id": context.episode_id,
            "step_id": context.step_id,
            "action_id": action.action_id,
            "action_type": action.action_type,
        }
        self._record(AuditEvent(event_type="evaluation_started", **common))
        self._evaluated_action_ids.add(action.action_id)

        if self.mode is ControlMode.DISABLED:
            decision = ControlDecision(
                decision=DecisionType.APPROVE,
                checks=(),
                metadata={"mode": self.mode.value, "library_digest": None},
            )
            self._emit_decision(decision, common)
            return decision

        checks: list[CheckResult] = []
        for validator in self.validators:
            validator_name = type(validator).__name__
            try:
                results = tuple(validator.validate(action, context))
                if any(not isinstance(item, CheckResult) for item in results):
                    raise TypeError("validator returned a non-CheckResult value")
            except Exception as exc:
                results = (
                    CheckResult(
                        check_id=f"validator_failure:{validator_name}",
                        passed=False,
                        level=CheckLevel.CRITICAL,
                        reason=f"Validator failed: {type(exc).__name__}: {exc}",
                        source="infrastructure",
                        recommended_decision=DecisionType.ESCALATE,
                    ),
                )
            checks.extend(results)
            self._record(
                AuditEvent(
                    event_type="validator_completed",
                    metadata={
                        "validator": validator_name,
                        "checks": [self._check_metadata(item) for item in results],
                    },
                    **common,
                )
            )

        library_digest: str | None = None
        retrieved = ()
        if self.constraint_library is not None:
            library_digest = self.constraint_library.digest
        if (
            self.constraint_library is not None
            and self.retriever is not None
            and self.constraint_evaluator is not None
        ):
            try:
                retrieved = tuple(
                    self.retriever.retrieve(action, context, self.constraint_library)
                )
                for item in retrieved:
                    self._record(
                        AuditEvent(
                            event_type="constraint_retrieved",
                            metadata={
                                "constraint_id": item.constraint.constraint_id,
                                "rank": item.rank,
                                "score": item.score,
                                "library_digest": library_digest,
                            },
                            **common,
                        )
                    )
                evaluated = tuple(
                    self.constraint_evaluator.evaluate(action, context, retrieved)
                )
                if any(not isinstance(item, CheckResult) for item in evaluated):
                    raise TypeError("constraint evaluator returned a non-CheckResult value")
            except Exception as exc:
                evaluated = (
                    CheckResult(
                        check_id="constraint_pipeline_failure",
                        passed=False,
                        level=CheckLevel.CRITICAL,
                        reason=f"Constraint pipeline failed: {type(exc).__name__}: {exc}",
                        source="infrastructure",
                        recommended_decision=DecisionType.ESCALATE,
                    ),
                )
                self._record(
                    AuditEvent(
                        event_type="constraint_pipeline_failed",
                        metadata={"error": evaluated[0].reason},
                        **common,
                    )
                )
            checks.extend(evaluated)
            for item in evaluated:
                if not item.passed and item.source != "infrastructure":
                    self._record(
                        AuditEvent(
                            event_type="constraint_activated",
                            metadata=self._check_metadata(item),
                            **common,
                        )
                    )

        final = aggregate_checks(checks, mode=self.mode)
        failed_reasons = [item.reason for item in checks if not item.passed and item.reason]
        decision = ControlDecision(
            decision=final,
            checks=tuple(checks),
            message="; ".join(failed_reasons) or None,
            metadata={
                "mode": self.mode.value,
                "library_digest": library_digest,
                "retrieved_count": len(retrieved),
            },
        )
        self._emit_decision(decision, common)
        return decision

    def observe(self, outcome: ObservedOutcome) -> None:
        if outcome.action_id not in self._evaluated_action_ids:
            raise ValueError("cannot observe an action that was not evaluated")
        self._record(
            AuditEvent(
                event_type="outcome_observed",
                action_id=outcome.action_id,
                metadata={
                    "executed": outcome.executed,
                    "status": outcome.status,
                    "visible_result": dict(outcome.visible_result),
                },
            )
        )

    def _emit_decision(self, decision: ControlDecision, common: dict[str, object]) -> None:
        self._record(
            AuditEvent(
                event_type="decision_emitted",
                metadata={
                    "decision": decision.decision.value,
                    "mode": self.mode.value,
                    "checks": [self._check_metadata(item) for item in decision.checks],
                    **dict(decision.metadata),
                },
                **common,
            )
        )

    @staticmethod
    def _check_metadata(check: CheckResult) -> dict[str, object]:
        return {
            "check_id": check.check_id,
            "passed": check.passed,
            "level": check.level.value,
            "reason": check.reason,
            "source": check.source,
            "recommended_decision": (
                check.recommended_decision.value if check.recommended_decision else None
            ),
            "metadata": dict(check.metadata),
        }

    def _record(self, event: AuditEvent) -> None:
        try:
            self.audit_sink.record(event)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._audit_failures.append(message)
            if self.strict_audit:
                raise AuditFailure(message) from exc
