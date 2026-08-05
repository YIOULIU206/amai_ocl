"""Independent AgenticPay episode runner using the V2 action boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .agenticpay_adapter import AgenticPayOCLAdapter, HostActionDisposition


@dataclass(frozen=True, slots=True)
class AgenticPayProposalRecord:
    action_id: str
    proposed_text: str
    decision: str
    executed: bool
    message: str | None


@dataclass(frozen=True, slots=True)
class AgenticPayTurnRecord:
    round_id: int
    buyer_visible_text: str | None
    proposals: tuple[AgenticPayProposalRecord, ...]
    seller_executed_text: str | None


@dataclass(frozen=True, slots=True)
class AgenticPayRunResult:
    episode_id: str
    turns: tuple[AgenticPayTurnRecord, ...]
    final_info: Mapping[str, Any]


def _normalize(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def _default_make_env(env_id: str, **kwargs: Any) -> Any:
    import agenticpay  # imported only when the real host is requested

    return agenticpay.make(env_id, **kwargs)


def run_agenticpay_episode(
    *,
    episode_id: str,
    env_id: str,
    buyer_agent: Any,
    seller_agent: Any,
    ocl_adapter: AgenticPayOCLAdapter,
    reset_kwargs: Mapping[str, Any],
    env_kwargs: Mapping[str, Any] | None = None,
    maximum_revision_attempts: int = 1,
    make_env: Callable[..., Any] | None = None,
) -> AgenticPayRunResult:
    """Run an episode without importing or delegating to legacy ``aimai_ocl``."""

    if maximum_revision_attempts < 0:
        raise ValueError("maximum_revision_attempts must be non-negative")
    factory = make_env or _default_make_env
    configuration = dict(env_kwargs or {})
    configuration["buyer_agent"] = buyer_agent
    configuration["seller_agent"] = seller_agent
    env = factory(env_id, **configuration)
    turns: list[AgenticPayTurnRecord] = []
    final_info: Mapping[str, Any] = {}
    try:
        observation, _ = env.reset(**dict(reset_kwargs))
        done = False
        while not done:
            round_id = int(observation.get("current_round", len(turns)))
            history = list(observation.get("conversation_history", []))
            buyer_raw = buyer_agent.respond(
                conversation_history=history,
                current_state=observation,
            )
            buyer_text = _normalize(buyer_raw)
            seller_history = list(history)
            if buyer_text is not None:
                seller_history.append(
                    {"role": "buyer", "content": buyer_text, "round": round_id}
                )

            proposals: list[tuple[HostActionDisposition, str]] = []
            observed_action_ids: set[str] = set()
            seller_text: str | None = None
            revision_feedback: str | None = None
            for attempt in range(maximum_revision_attempts + 1):
                generation_history = list(seller_history)
                if revision_feedback:
                    generation_history.append(
                        {
                            "role": "platform",
                            "content": (
                                "Revise the proposal to satisfy this control feedback: "
                                + revision_feedback
                            ),
                            "round": round_id,
                        }
                    )
                seller_raw = seller_agent.respond(
                    conversation_history=generation_history,
                    current_state=observation,
                )
                proposal = _normalize(seller_raw)
                if proposal is None:
                    break
                disposition = ocl_adapter.evaluate_seller_text(
                    episode_id=episode_id,
                    step_id=round_id,
                    actor_id=getattr(seller_agent, "name", "seller"),
                    seller_text=proposal,
                    dialogue=generation_history,
                    visible_state=observation,
                    attempt=attempt,
                )
                proposals.append((disposition, proposal))
                if disposition.execute:
                    seller_text = disposition.seller_text
                    break
                if disposition.requires_revision and attempt < maximum_revision_attempts:
                    ocl_adapter.observe(disposition, status="revise_requested")
                    observed_action_ids.add(disposition.action_id)
                    revision_feedback = disposition.decision.message or "proposal requires revision"
                    continue
                break

            observation, _, terminated, truncated, final_info = env.step(
                buyer_action=buyer_text,
                seller_action=seller_text,
            )
            done = bool(terminated or truncated)
            for disposition, proposal in proposals:
                if disposition.action_id not in observed_action_ids:
                    ocl_adapter.observe(
                        disposition,
                        status=(
                            str(final_info.get("status", "executed"))
                            if disposition.execute
                            else disposition.decision.decision.value
                        ),
                    )
            turns.append(
                AgenticPayTurnRecord(
                    round_id=round_id,
                    buyer_visible_text=buyer_text,
                    proposals=tuple(
                        AgenticPayProposalRecord(
                            action_id=disposition.action_id,
                            proposed_text=proposal,
                            decision=disposition.decision.decision.value,
                            executed=disposition.execute,
                            message=disposition.decision.message,
                        )
                        for disposition, proposal in proposals
                    ),
                    seller_executed_text=seller_text,
                )
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return AgenticPayRunResult(
        episode_id=episode_id,
        turns=tuple(turns),
        final_info=dict(final_info),
    )
