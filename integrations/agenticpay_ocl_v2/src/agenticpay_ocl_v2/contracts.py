"""Stable online contracts shared by runtimes and host adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ContractError(ValueError):
    """Raised when observable runtime input violates a public contract."""


class CheckLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DecisionType(str, Enum):
    APPROVE = "approve"
    WARN = "warn"
    REVISE = "revise"
    BLOCK = "block"
    ESCALATE = "escalate"


def _frozen_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} keys must be strings")
    return MappingProxyType(dict(value))


def _required_text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ProposedAction:
    action_id: str
    actor_id: str
    action_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    visible_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _required_text(self.action_id, name="action_id"))
        object.__setattr__(self, "actor_id", _required_text(self.actor_id, name="actor_id"))
        object.__setattr__(self, "action_type", _required_text(self.action_type, name="action_type"))
        if self.visible_text is not None and not isinstance(self.visible_text, str):
            raise ContractError("visible_text must be a string or None")
        object.__setattr__(self, "payload", _frozen_mapping(self.payload, name="payload"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, name="metadata"))


@dataclass(frozen=True, slots=True)
class ObservableContext:
    episode_id: str
    step_id: int
    actor_id: str
    dialogue: tuple[Mapping[str, Any], ...] = ()
    visible_state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _required_text(self.episode_id, name="episode_id"))
        object.__setattr__(self, "actor_id", _required_text(self.actor_id, name="actor_id"))
        if not isinstance(self.step_id, int) or self.step_id < 0:
            raise ContractError("step_id must be a non-negative integer")
        if not isinstance(self.dialogue, tuple):
            raise ContractError("dialogue must be a tuple")
        object.__setattr__(
            self,
            "dialogue",
            tuple(_frozen_mapping(item, name="dialogue item") for item in self.dialogue),
        )
        object.__setattr__(
            self,
            "visible_state",
            _frozen_mapping(self.visible_state, name="visible_state"),
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, name="metadata"))


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    passed: bool
    level: CheckLevel = CheckLevel.INFO
    reason: str = ""
    source: str = "validator"
    recommended_decision: DecisionType | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", _required_text(self.check_id, name="check_id"))
        if not isinstance(self.passed, bool):
            raise ContractError("passed must be boolean")
        if not isinstance(self.level, CheckLevel):
            object.__setattr__(self, "level", CheckLevel(self.level))
        if self.recommended_decision is not None and not isinstance(
            self.recommended_decision, DecisionType
        ):
            object.__setattr__(
                self,
                "recommended_decision",
                DecisionType(self.recommended_decision),
            )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, name="metadata"))


@dataclass(frozen=True, slots=True)
class ControlDecision:
    decision: DecisionType
    checks: tuple[CheckResult, ...]
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, DecisionType):
            object.__setattr__(self, "decision", DecisionType(self.decision))
        if not isinstance(self.checks, tuple):
            raise ContractError("checks must be a tuple")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, name="metadata"))


@dataclass(frozen=True, slots=True)
class ObservedOutcome:
    action_id: str
    executed: bool
    status: str
    visible_result: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _required_text(self.action_id, name="action_id"))
        if not isinstance(self.executed, bool):
            raise ContractError("executed must be boolean")
        object.__setattr__(self, "status", _required_text(self.status, name="status"))
        object.__setattr__(
            self,
            "visible_result",
            _frozen_mapping(self.visible_result, name="visible_result"),
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, name="metadata"))
