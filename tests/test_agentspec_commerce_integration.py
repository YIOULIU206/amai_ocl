"""Runner-level integration tests for AgentSpec-Commerce."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import aimai_ocl.runner as runner
from aimai_ocl.baselines.agentspec_commerce import (
    AgentSpecCommerceAdapter,
)
from aimai_ocl.schemas import EpisodeTrace


class FakeBuyer:
    name = "buyer"

    def respond(
        self,
        conversation_history: Any,
        current_state: Any,
    ) -> str:
        return "I can offer $100."


class RecordingSeller:
    name = "seller"

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
        return self.replies.pop(0) if self.replies else None


class FakeEnvAdapter:
    last_instance: "FakeEnvAdapter | None" = None

    def __init__(
        self,
        *,
        env_id: str,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.env_id = env_id
        self.env_kwargs = dict(env_kwargs or {})
        self.received: list[dict[str, Any]] = []
        self.closed = False
        FakeEnvAdapter.last_instance = self

    def reset(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                "current_round": 0,
                "conversation_history": [],
                "buyer_max_price": 120.0,
                "buyer_budget_cap": 120.0,
                "public_product": "Winter Jacket",
            },
            {},
        )

    def new_trace(
        self,
        scenario: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeTrace:
        return EpisodeTrace(
            episode_id="agentspec-integration-test",
            env_id=self.env_id,
            scenario=dict(scenario or {}),
            metadata=dict(metadata or {}),
        )

    def step(
        self,
        buyer_action: str | None = None,
        seller_action: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.received.append(
            {
                "buyer_action": buyer_action,
                "seller_action": seller_action,
            }
        )
        return (
            {
                "current_round": 1,
                "conversation_history": [],
            },
            0.0,
            True,
            False,
            {
                "status": "finished",
                "round": 1,
                "seller_reward": 0.0,
                "buyer_reward": 0.0,
            },
        )

    def close(self) -> None:
        self.closed = True


def run_agentspec_episode(
    seller: RecordingSeller,
) -> EpisodeTrace:
    with patch.object(runner, "EnvAdapter", FakeEnvAdapter):
        trace, _ = runner.run_episode(
            env_id="FakeCommerce-v0",
            buyer_agent=FakeBuyer(),
            seller_agent=seller,
            reset_kwargs={
                "user_requirement": "Buy one winter jacket.",
                "product_info": {
                    "name": "Winter Jacket",
                    "price": 180.0,
                },
                "user_profile": "Integration-test buyer",
            },
            env_kwargs={
                "buyer_max_price": 120.0,
                "seller_min_price": 90.0,
                "max_rounds": 1,
            },
            trace_metadata={"arm": "agentspec_commerce"},
            ocl=False,
            baseline_mode="agentspec_commerce",
            seller_context_mode="observation_only",
            agentspec_adapter=AgentSpecCommerceAdapter(),
        )

    return trace


class AgentSpecCommerceRunnerIntegrationTests(unittest.TestCase):
    def test_blocked_draft_is_reflected_and_only_revision_reaches_env(
        self,
    ) -> None:
        original = "I require $130."
        revision = "I can offer the jacket for $100."
        seller = RecordingSeller([original, revision])

        trace = run_agentspec_episode(seller)

        env = FakeEnvAdapter.last_instance
        self.assertIsNotNone(env)
        assert env is not None

        self.assertEqual(2, len(seller.calls))
        self.assertEqual(1, len(env.received))
        self.assertEqual(revision, env.received[0]["seller_action"])
        self.assertNotEqual(original, env.received[0]["seller_action"])

        for call in seller.calls:
            state = call["current_state"]
            self.assertNotIn("buyer_max_price", state)
            self.assertNotIn("buyer_budget_cap", state)

        rounds = trace.metadata["agentspec_commerce"]["rounds"]
        self.assertEqual(1, len(rounds))

        record = rounds[0]
        self.assertEqual(
            ["llm_self_reflect", "continue"],
            record["decisions"],
        )
        self.assertEqual(1, record["self_reflection_calls"])
        self.assertEqual(1, record["selected_attempt"])
        self.assertEqual(1, record["reached_env"])
        self.assertFalse(record["no_op"])

        executed = trace.metadata["executed_seller_actions"]
        self.assertEqual(1, len(executed))
        self.assertEqual(revision, executed[0]["text"])
        self.assertEqual(
            1,
            executed[0]["agentspec_selected_attempt"],
        )

    def test_second_violation_becomes_no_op(self) -> None:
        seller = RecordingSeller(
            [
                "I require $130.",
                "I still require $140.",
                "This third reply must never be generated.",
            ]
        )

        trace = run_agentspec_episode(seller)

        env = FakeEnvAdapter.last_instance
        self.assertIsNotNone(env)
        assert env is not None

        self.assertEqual(2, len(seller.calls))
        self.assertEqual(1, len(env.received))
        self.assertIsNone(env.received[0]["seller_action"])

        self.assertEqual(
            [],
            trace.metadata.get("executed_seller_actions", []),
        )

        record = trace.metadata["agentspec_commerce"]["rounds"][0]
        self.assertEqual(
            ["llm_self_reflect", "llm_self_reflect"],
            record["decisions"],
        )
        self.assertEqual(1, record["self_reflection_calls"])
        self.assertIsNone(record["selected_attempt"])
        self.assertEqual(0, record["reached_env"])
        self.assertTrue(record["no_op"])

    def test_missing_adapter_fails_closed(self) -> None:
        seller = RecordingSeller(["I can offer $100."])

        with patch.object(runner, "EnvAdapter", FakeEnvAdapter):
            with self.assertRaisesRegex(
                ValueError,
                "requires a configured AgentSpecCommerceAdapter",
            ):
                runner.run_episode(
                    env_id="FakeCommerce-v0",
                    buyer_agent=FakeBuyer(),
                    seller_agent=seller,
                    reset_kwargs={
                        "user_requirement": "Buy one winter jacket.",
                        "product_info": {
                            "name": "Winter Jacket",
                            "price": 180.0,
                        },
                        "user_profile": "Integration-test buyer",
                    },
                    env_kwargs={
                        "buyer_max_price": 120.0,
                        "seller_min_price": 90.0,
                        "max_rounds": 1,
                    },
                    ocl=False,
                    baseline_mode="agentspec_commerce",
                    agentspec_adapter=None,
                )


if __name__ == "__main__":
    unittest.main()
