"""Conversion and JSONL export of observable AgenticPay run traces."""

from __future__ import annotations

import json
from pathlib import Path

from .agenticpay_runner import AgenticPayRunResult
from aocl_core.json_utils import jsonable
from aocl_core.learning import LearningTrace, VisibleActionStep


def learning_trace_from_run(
    result: AgenticPayRunResult,
    *,
    profile_id: str,
    split: str,
    action_type: str = "commerce.respond",
) -> LearningTrace:
    steps: list[VisibleActionStep] = []
    for turn in result.turns:
        for proposal in turn.proposals:
            steps.append(
                VisibleActionStep(
                    step_id=len(steps),
                    action_type=action_type,
                    observable_context={
                        "round_id": turn.round_id,
                        "buyer_visible_text": turn.buyer_visible_text,
                    },
                    proposed_action={
                        "action_id": proposal.action_id,
                        "visible_text": proposal.proposed_text,
                    },
                    executed=proposal.executed,
                    visible_result={
                        "decision": proposal.decision,
                        "message": proposal.message,
                        "executed_text": (
                            turn.seller_executed_text if proposal.executed else None
                        ),
                    },
                )
            )
    return LearningTrace(
        episode_id=result.episode_id,
        scenario_id=profile_id,
        split=split,
        steps=tuple(steps),
        visible_outcome=dict(result.final_info),
    )


def append_learning_trace(path: str | Path, trace: LearningTrace) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(jsonable(trace), ensure_ascii=False, sort_keys=True) + "\n"
        )
