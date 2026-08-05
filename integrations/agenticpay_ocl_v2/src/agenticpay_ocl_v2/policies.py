"""Control modes and deterministic decision aggregation."""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from .contracts import CheckLevel, CheckResult, DecisionType


class ControlMode(str, Enum):
    DISABLED = "disabled"
    AUDIT = "audit"
    WARNING = "warning"
    BLOCKING = "blocking"


_RANK = {
    DecisionType.APPROVE: 0,
    DecisionType.WARN: 1,
    DecisionType.REVISE: 2,
    DecisionType.BLOCK: 3,
    DecisionType.ESCALATE: 4,
}


def natural_decision(check: CheckResult) -> DecisionType:
    if check.passed:
        return DecisionType.APPROVE
    if check.recommended_decision is not None:
        return check.recommended_decision
    if check.level is CheckLevel.CRITICAL:
        return DecisionType.ESCALATE
    if check.level is CheckLevel.ERROR:
        return DecisionType.BLOCK
    if check.level is CheckLevel.WARNING:
        return DecisionType.WARN
    return DecisionType.APPROVE


def aggregate_checks(
    checks: Sequence[CheckResult],
    *,
    mode: ControlMode,
) -> DecisionType:
    if mode in {ControlMode.DISABLED, ControlMode.AUDIT}:
        return DecisionType.APPROVE
    decisions = [natural_decision(check) for check in checks]
    strongest = max(decisions, key=_RANK.get, default=DecisionType.APPROVE)
    if mode is ControlMode.WARNING and strongest is not DecisionType.APPROVE:
        return DecisionType.WARN
    return strongest
