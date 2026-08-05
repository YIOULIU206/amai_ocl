"""Bridge CoffeeBench operational tools into the shared A-OCL core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from aocl_core.contracts import (
    CheckLevel,
    CheckResult,
    ControlDecision,
    DecisionType,
    ObservableContext,
    ObservedOutcome,
    ProposedAction,
)
from aocl_core.runtime import IntegrationOCLRuntime

from .contracts import ValidationResult, ValidationStatus
from .json_utils import jsonable


@dataclass(frozen=True, slots=True)
class CoffeeBenchActionDisposition:
    action_id: str
    decision: ControlDecision
    execute: bool


@dataclass(slots=True)
class CoffeeBenchCoreAdapter:
    """Normalize one focal BusinessApp tool call without executing it."""

    runtime: IntegrationOCLRuntime
    episode_id: str
    focal_agent_id: str
    _sequence: int = field(default=0, init=False, repr=False)

    def evaluate_tool(
        self,
        *,
        day: int,
        action_name: str,
        action_input: Mapping[str, Any],
        visible_state: Mapping[str, Any],
        validation: ValidationResult,
    ) -> CoffeeBenchActionDisposition:
        self._sequence += 1
        action_id = (
            f"{self.episode_id}:{day}:{self.focal_agent_id}:"
            f"{self._sequence}:{action_name}"
        )
        action = ProposedAction(
            action_id=action_id,
            actor_id=self.focal_agent_id,
            action_type=f"coffeebench.{action_name}",
            payload=jsonable(dict(action_input)),
            metadata={"host": "coffeebench", "day": day},
        )
        context = ObservableContext(
            episode_id=self.episode_id,
            step_id=self._sequence,
            actor_id=self.focal_agent_id,
            visible_state=jsonable(dict(visible_state)),
            metadata={"host": "coffeebench", "day": day},
        )
        decision = self.runtime.evaluate(
            action=action,
            context=context,
            host_checks=_validation_checks(validation),
        )
        return CoffeeBenchActionDisposition(
            action_id=action_id,
            decision=decision,
            execute=decision.decision in {DecisionType.APPROVE, DecisionType.WARN},
        )

    def observe(
        self,
        disposition: CoffeeBenchActionDisposition,
        *,
        executed: bool,
        status: str,
        visible_result: Mapping[str, Any] | None = None,
    ) -> None:
        self.runtime.observe(
            ObservedOutcome(
                action_id=disposition.action_id,
                executed=executed,
                status=status,
                visible_result=jsonable(dict(visible_result or {})),
                metadata={"host": "coffeebench"},
            )
        )


def _validation_checks(validation: ValidationResult) -> tuple[CheckResult, ...]:
    if validation.status is ValidationStatus.NOT_RUN:
        return ()
    checks: list[CheckResult] = []
    for index, error in enumerate(validation.errors, start=1):
        checks.append(
            CheckResult(
                check_id=f"coffeebench.hard.error.{index}",
                passed=False,
                level=CheckLevel.ERROR,
                reason=error,
                source="coffeebench_validator",
                recommended_decision=DecisionType.BLOCK,
                metadata=validation.metadata,
            )
        )
    for index, warning in enumerate(validation.warnings, start=1):
        checks.append(
            CheckResult(
                check_id=f"coffeebench.hard.warning.{index}",
                passed=False,
                level=CheckLevel.WARNING,
                reason=warning,
                source="coffeebench_validator",
                recommended_decision=DecisionType.WARN,
                metadata=validation.metadata,
            )
        )
    if not checks:
        checks.append(
            CheckResult(
                check_id="coffeebench.hard.pass",
                passed=True,
                source="coffeebench_validator",
                metadata=validation.metadata,
            )
        )
    return tuple(checks)
