"""Conversion and JSONL export of observable AgenticPay run traces."""

from __future__ import annotations

import json
from pathlib import Path

from .agenticpay_runner import AgenticPayRunResult
from .json_utils import jsonable
from .learning import LearningTrace, VisibleTurn


def learning_trace_from_run(
    result: AgenticPayRunResult,
    *,
    profile_id: str,
    split: str,
    action_type: str = "commerce.respond",
) -> LearningTrace:
    turns: list[VisibleTurn] = []
    for turn in result.turns:
        latest_proposal = turn.proposals[-1].proposed_text if turn.proposals else None
        turns.append(
            VisibleTurn(
                round_id=turn.round_id,
                buyer_visible_text=turn.buyer_visible_text,
                seller_proposed_text=latest_proposal,
                seller_executed_text=turn.seller_executed_text,
            )
        )
    return LearningTrace(
        episode_id=result.episode_id,
        profile_id=profile_id,
        split=split,
        action_type=action_type,
        turns=tuple(turns),
        visible_outcome=dict(result.final_info),
    )


def append_learning_trace(path: str | Path, trace: LearningTrace) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(jsonable(trace), ensure_ascii=False, sort_keys=True) + "\n"
        )
