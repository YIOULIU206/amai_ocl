"""Generic validators; AgenticPay business rules are adapter-owned."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .contracts import (
    CheckLevel,
    CheckResult,
    DecisionType,
    ObservableContext,
    ProposedAction,
)


class Validator(Protocol):
    def validate(
        self,
        action: ProposedAction,
        context: ObservableContext,
    ) -> Sequence[CheckResult]: ...


@dataclass(frozen=True, slots=True)
class CallableValidator:
    function: Callable[[ProposedAction, ObservableContext], Sequence[CheckResult]]

    def validate(
        self,
        action: ProposedAction,
        context: ObservableContext,
    ) -> Sequence[CheckResult]:
        return self.function(action, context)


@dataclass(frozen=True, slots=True)
class RequiredPayloadKeysValidator:
    keys: tuple[str, ...]
    check_id: str = "required_payload_keys"

    def validate(
        self,
        action: ProposedAction,
        context: ObservableContext,
    ) -> Sequence[CheckResult]:
        del context
        missing = tuple(key for key in self.keys if key not in action.payload)
        return (
            CheckResult(
                check_id=self.check_id,
                passed=not missing,
                level=CheckLevel.ERROR if missing else CheckLevel.INFO,
                reason=(f"Missing payload keys: {', '.join(missing)}" if missing else ""),
                recommended_decision=(DecisionType.BLOCK if missing else None),
                metadata={"missing_keys": missing},
            ),
        )
