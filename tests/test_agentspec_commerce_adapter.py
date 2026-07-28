"""Unit tests for the independent AgentSpec-Commerce seller-turn adapter."""

from __future__ import annotations

import unittest
from typing import Any

from aimai_ocl.baselines.agentspec_commerce import (
    AgentSpecCommerceAdapter,
)


class RecordingSeller:
    """Seller stub that records calls and returns scripted replies."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def respond(
        self,
        conversation_history: Any,
        current_state: Any,
    ) -> Any:
        self.calls.append(
            {
                "conversation_history": conversation_history,
                "current_state": current_state,
            }
        )

        if not self.replies:
            return None

        return self.replies.pop(0)


class AgentSpecCommerceAdapterTests(unittest.TestCase):
    """Verify one-reflection and no-op behaviour."""

    def setUp(self) -> None:
        self.adapter = AgentSpecCommerceAdapter()

    def process(
        self,
        *,
        text: str | None,
        seller: RecordingSeller | None = None,
        seller_history: list[Any] | None = None,
        observation: dict[str, Any] | None = None,
    ):
        return self.adapter.process_seller_turn(
            text=text,
            seller_history=seller_history or [],
            observation=observation or {"current_round": 0},
            buyer_max_price=120.0,
            seller_min_price=90.0,
            actor_id="seller",
            round_id=0,
            seller_respond=(
                seller.respond
                if seller is not None
                else None
            ),
        )

    def test_safe_initial_draft_passes_without_reflection(self) -> None:
        seller = RecordingSeller(
            ["This reply must never be used."]
        )

        outcome = self.process(
            text="I can offer the jacket for $100.",
            seller=seller,
        )

        self.assertEqual(
            "I can offer the jacket for $100.",
            outcome.text,
        )
        self.assertEqual(["continue"], outcome.decisions)
        self.assertEqual(0, outcome.self_reflection_calls)
        self.assertEqual(0, outcome.selected_attempt)
        self.assertFalse(outcome.no_op)
        self.assertEqual([], seller.calls)

    def test_blocked_draft_gets_one_compliant_revision(self) -> None:
        seller = RecordingSeller(
            ["I can offer the jacket for $100."]
        )

        original_history = [
            {
                "role": "buyer",
                "content": "I can pay $100.",
                "round": 0,
            }
        ]

        outcome = self.process(
            text="I can offer the jacket for $130.",
            seller=seller,
            seller_history=original_history,
            observation={
                "current_round": 0,
                "buyer_max_price": 120.0,
                "buyer_budget_cap": 120.0,
                "public_product": "Winter Jacket",
            },
        )

        self.assertEqual(
            "I can offer the jacket for $100.",
            outcome.text,
        )
        self.assertEqual(
            ["llm_self_reflect", "continue"],
            outcome.decisions,
        )
        self.assertEqual(1, outcome.self_reflection_calls)
        self.assertEqual(1, outcome.selected_attempt)
        self.assertFalse(outcome.no_op)
        self.assertEqual(1, len(seller.calls))

        seller_state = seller.calls[0]["current_state"]
        self.assertNotIn("buyer_max_price", seller_state)
        self.assertNotIn("buyer_budget_cap", seller_state)

        reflection_history = seller.calls[0]["conversation_history"]
        self.assertEqual(2, len(reflection_history))
        self.assertEqual("platform", reflection_history[-1]["role"])

        feedback = reflection_history[-1]["content"].lower()
        self.assertNotIn("buyer_max_price", feedback)
        self.assertNotIn("buyer maximum price", feedback)

        # The caller-owned history must not be modified.
        self.assertEqual(1, len(original_history))

    def test_second_violation_becomes_no_op(self) -> None:
        seller = RecordingSeller(
            ["I still require $140."]
        )

        outcome = self.process(
            text="I can offer the jacket for $130.",
            seller=seller,
        )

        self.assertIsNone(outcome.text)
        self.assertEqual(
            ["llm_self_reflect", "llm_self_reflect"],
            outcome.decisions,
        )
        self.assertEqual(1, outcome.self_reflection_calls)
        self.assertIsNone(outcome.selected_attempt)
        self.assertTrue(outcome.no_op)
        self.assertEqual(1, len(seller.calls))

    def test_no_revision_callable_becomes_no_op(self) -> None:
        outcome = self.process(
            text="I can offer the jacket for $130.",
            seller=None,
        )

        self.assertIsNone(outcome.text)
        self.assertEqual(
            ["llm_self_reflect"],
            outcome.decisions,
        )
        self.assertEqual(0, outcome.self_reflection_calls)
        self.assertTrue(outcome.no_op)

    def test_privacy_revision_can_pass(self) -> None:
        seller = RecordingSeller(
            ["Could you confirm the jacket size instead?"]
        )

        outcome = self.process(
            text="Please provide your credit card number.",
            seller=seller,
        )

        self.assertEqual(
            "Could you confirm the jacket size instead?",
            outcome.text,
        )
        self.assertEqual(
            ["llm_self_reflect", "continue"],
            outcome.decisions,
        )
        self.assertEqual(1, outcome.self_reflection_calls)

    def test_empty_initial_output_is_no_op(self) -> None:
        outcome = self.process(
            text="   ",
        )

        self.assertIsNone(outcome.text)
        self.assertEqual([], outcome.attempts)
        self.assertTrue(outcome.no_op)


if __name__ == "__main__":
    unittest.main()
