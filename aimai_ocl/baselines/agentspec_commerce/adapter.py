"""Seller-turn adapter for the independent AgentSpec-Commerce baseline.

Execution flow:

    initial seller draft
        -> AgentSpec declarative runtime
        -> CONTINUE: execute the seller-authored draft unchanged
        -> SELF_REFLECT: ask the same seller LLM to revise exactly once
        -> second violation: return None, making the seller turn a no-op

This adapter does not use ToolGuard, EGI control, deterministic repair,
price clamping, escalation, or replanning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from aimai_ocl.adapters import raw_action_from_text
from aimai_ocl.baselines.agentspec_commerce.retry_policy import (
    AgentSpecRetryPolicy,
)
from aimai_ocl.baselines.agentspec_commerce.runtime import (
    AgentSpecCommerceRuntime,
    EnforcementDecision,
    RuntimeVerdict,
)
from aimai_ocl.schemas import ActionRole, RawAction


_PLATFORM_ONLY_KEYS = frozenset(
    {
        "buyer_max_price",
        "buyer_maximum_price",
        "buyer_budget",
        "buyer_budget_cap",
        "budget_cap",
        "buyer_willingness_to_pay",
    }
)


@dataclass(frozen=True)
class AgentSpecAttempt:
    """Result of checking one seller-authored proposal."""

    attempt: int
    verdict: RuntimeVerdict
    runtime_sec: float


@dataclass
class SellerTurnOutcome:
    """Final outcome of one AgentSpec-controlled seller turn."""

    text: str | None
    attempts: list[AgentSpecAttempt] = field(default_factory=list)
    self_reflection_calls: int = 0
    selected_attempt: int | None = None
    no_op: bool = False

    @property
    def decisions(self) -> list[str]:
        return [
            attempt.verdict.decision.value
            for attempt in self.attempts
        ]


class AgentSpecCommerceAdapter:
    """Apply AgentSpec-Commerce rules before seller execution."""

    def __init__(
        self,
        *,
        runtime: AgentSpecCommerceRuntime | None = None,
        retry_policy: AgentSpecRetryPolicy | None = None,
        parse_action: Callable[..., RawAction] = raw_action_from_text,
    ) -> None:
        self._runtime = (
            runtime if runtime is not None else AgentSpecCommerceRuntime()
        )
        self._retry_policy = (
            retry_policy
            if retry_policy is not None
            else AgentSpecRetryPolicy()
        )
        self._parse_action = parse_action

    @property
    def runtime(self) -> AgentSpecCommerceRuntime:
        return self._runtime

    @property
    def retry_policy(self) -> AgentSpecRetryPolicy:
        return self._retry_policy

    def process_seller_turn(
        self,
        *,
        text: str | None,
        seller_history: Sequence[Any] | None,
        observation: Mapping[str, Any] | None,
        buyer_max_price: float | None,
        seller_min_price: float | None,
        actor_id: str,
        round_id: int,
        seller_respond: Callable[..., Any] | None = None,
    ) -> SellerTurnOutcome:
        """Check one seller turn and optionally request one self-reflection.

        The returned text is always byte-identical to either the initial seller
        draft or the seller-authored revision. This adapter never rewrites or
        clamps seller text.
        """
        candidate = _normalize_seller_text(text)

        if candidate is None:
            return SellerTurnOutcome(
                text=None,
                attempts=[],
                self_reflection_calls=0,
                selected_attempt=None,
                no_op=True,
            )

        attempts: list[AgentSpecAttempt] = []
        extra_generations = 0
        attempt_index = 0

        while True:
            raw_action = self._parse_action(
                actor_id,
                ActionRole.SELLER,
                candidate,
            )

            started = perf_counter()
            verdict = self._runtime.evaluate(
                action=raw_action,
                buyer_max_price=buyer_max_price,
                seller_min_price=seller_min_price,
            )
            runtime_sec = perf_counter() - started

            attempts.append(
                AgentSpecAttempt(
                    attempt=attempt_index,
                    verdict=verdict,
                    runtime_sec=runtime_sec,
                )
            )

            if verdict.allowed:
                return SellerTurnOutcome(
                    text=candidate,
                    attempts=attempts,
                    self_reflection_calls=extra_generations,
                    selected_attempt=attempt_index,
                    no_op=False,
                )

            if verdict.decision != EnforcementDecision.SELF_REFLECT:
                return SellerTurnOutcome(
                    text=None,
                    attempts=attempts,
                    self_reflection_calls=extra_generations,
                    selected_attempt=None,
                    no_op=True,
                )

            can_reflect = (
                seller_respond is not None
                and self._retry_policy.allows_self_reflection(
                    extra_generations
                )
            )

            if not can_reflect:
                return SellerTurnOutcome(
                    text=None,
                    attempts=attempts,
                    self_reflection_calls=extra_generations,
                    selected_attempt=None,
                    no_op=True,
                )

            reflection_history = (
                self._retry_policy.build_reflection_history(
                    seller_history,
                    verdict.reason,
                    buyer_max_price=buyer_max_price,
                    round_id=round_id,
                )
            )

            revision = _normalize_seller_text(
                seller_respond(
                    conversation_history=reflection_history,
                    current_state=seller_visible_state(observation),
                )
            )

            extra_generations += 1

            if revision is None:
                return SellerTurnOutcome(
                    text=None,
                    attempts=attempts,
                    self_reflection_calls=extra_generations,
                    selected_attempt=None,
                    no_op=True,
                )

            attempt_index += 1
            candidate = revision


def seller_visible_state(
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return observation data with platform-only buyer values removed."""

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: clean(item)
                for key, item in value.items()
                if str(key).lower() not in _PLATFORM_ONLY_KEYS
            }

        if isinstance(value, list):
            return [clean(item) for item in value]

        if isinstance(value, tuple):
            return tuple(clean(item) for item in value)

        return value

    cleaned = clean(observation or {})
    return cleaned if isinstance(cleaned, dict) else {}


def _normalize_seller_text(value: Any) -> str | None:
    """Reject empty/non-string outputs without modifying valid seller text."""
    if not isinstance(value, str):
        return None

    if not value.strip():
        return None

    return value
