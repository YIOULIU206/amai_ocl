"""Runner-level integration tests for ToolGuard-Commerce."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent

for directory in (REPO_ROOT, TEST_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import aimai_ocl.runner as runner
from aimai_ocl.baselines.toolguard_commerce.adapter import (
    ToolGuardCommerceAdapter,
)
from aimai_ocl.baselines.toolguard_commerce.event_logger import (
    ToolGuardEventLogger,
)
from aimai_ocl.baselines.toolguard_commerce.retry_policy import RetryPolicy
from aimai_ocl.schemas import EpisodeTrace
from toolguard_commerce_fixtures import RecordingSeller, ScriptedGuard


class FakeBuyer:
    """Buyer stub that produces one deterministic offer."""

    name = "buyer"

    def respond(
        self,
        conversation_history: Any,
        current_state: Any,
    ) -> str:
        return "I can offer $100."


class FakeEnvAdapter:
    """Environment stub that records exactly what reaches step()."""

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

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                "current_round": 0,
                "conversation_history": [],
                "public_product": "Winter Jacket",
            },
            {},
        )

    def new_trace(
        self,
        *,
        scenario: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeTrace:
        return EpisodeTrace(
            episode_id="runner-integration-test",
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


def make_toolguard_adapter(
    guard: Any,
    *,
    episode_key: str,
) -> ToolGuardCommerceAdapter:
    logger = ToolGuardEventLogger(
        retry_budget=1,
        buyer_max_price_visibility="platform_visible",
        arm="toolguard_commerce",
        episode_key=episode_key,
    )

    return ToolGuardCommerceAdapter(
        guard_runtime=guard,
        retry_policy=RetryPolicy(budget=1),
        logger=logger,
        arm="toolguard_commerce",
        episode_key=episode_key,
    )


def run_toolguard_episode(
    *,
    seller: RecordingSeller,
    guard: ScriptedGuard,
    episode_key: str,
) -> EpisodeTrace:
    toolguard_adapter = make_toolguard_adapter(
        guard,
        episode_key=episode_key,
    )

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
            trace_metadata={
                "arm": "toolguard_commerce",
                "episode_key": episode_key,
            },
            ocl=False,
            baseline_mode="toolguard_commerce",
            seller_context_mode="observation_only",
            toolguard_adapter=toolguard_adapter,
            toolguard_buyer_max_price_visibility="platform_visible",
        )

    return trace


class ToolGuardCommerceRunnerIntegrationTests(unittest.TestCase):
    def test_blocked_draft_is_retried_and_only_revision_reaches_env(
        self,
    ) -> None:
        unsafe_text = "My offer is $130."
        compliant_revision = "I can offer the jacket for $110."

        seller = RecordingSeller(
            [
                unsafe_text,
                compliant_revision,
            ]
        )
        guard = ScriptedGuard(
            [
                (
                    "Proposed price 130.00 exceeds "
                    "buyer_max_price 120.00."
                ),
                None,
            ]
        )

        trace = run_toolguard_episode(
            seller=seller,
            guard=guard,
            episode_key="blocked-then-allowed",
        )

        env = FakeEnvAdapter.last_instance
        self.assertIsNotNone(env)
        assert env is not None

        # One initial generation and exactly one retry.
        self.assertEqual(2, len(seller.calls))
        self.assertEqual(2, len(guard.calls))

        # The unsafe original draft never reaches the environment.
        self.assertEqual(1, len(env.received))
        self.assertNotEqual(
            unsafe_text,
            env.received[0]["seller_action"],
        )
        self.assertEqual(
            compliant_revision,
            env.received[0]["seller_action"],
        )

        executed = trace.metadata["executed_seller_actions"]
        self.assertEqual(1, len(executed))
        self.assertEqual(compliant_revision, executed[0]["text"])
        self.assertIn("proposal_id", executed[0])

        payload = trace.metadata["toolguard_commerce"]
        proposals = payload["proposals"]

        self.assertEqual(2, len(proposals))

        # Original unsafe proposal was blocked and intercepted.
        self.assertEqual(1, proposals[0]["blocked"])
        self.assertEqual(1, proposals[0]["intercepted"])
        self.assertEqual(0, proposals[0]["reached_env"])
        self.assertEqual(1, proposals[0]["retry_attempted"])

        # Only the seller-authored revision reached the environment.
        self.assertEqual(0, proposals[1]["blocked"])
        self.assertEqual(1, proposals[1]["revision_passed"])
        self.assertEqual(1, proposals[1]["reached_env"])
        self.assertEqual(
            proposals[1]["proposal_id"],
            executed[0]["proposal_id"],
        )

        # The buyer budget cap never reaches either seller generation.
        for call in seller.calls:
            seller_state = call["current_state"]
            self.assertNotIn("buyer_max_price", seller_state)

    def test_second_block_becomes_no_op_without_third_generation(
        self,
    ) -> None:
        seller = RecordingSeller(
            [
                "My offer is $130.",
                "I still require $140.",
                "This third generation must never be used.",
            ]
        )
        guard = ScriptedGuard(
            [
                (
                    "Proposed price 130.00 exceeds "
                    "buyer_max_price 120.00."
                ),
                (
                    "Proposed price 140.00 exceeds "
                    "buyer_max_price 120.00."
                ),
            ]
        )

        trace = run_toolguard_episode(
            seller=seller,
            guard=guard,
            episode_key="blocked-twice",
        )

        env = FakeEnvAdapter.last_instance
        self.assertIsNotNone(env)
        assert env is not None

        # Initial generation plus one retry only.
        self.assertEqual(2, len(seller.calls))
        self.assertEqual(2, len(guard.calls))

        # Second block produces a seller no-op.
        self.assertEqual(1, len(env.received))
        self.assertIsNone(env.received[0]["seller_action"])
        self.assertEqual(
            [],
            trace.metadata.get("executed_seller_actions", []),
        )

        payload = trace.metadata["toolguard_commerce"]
        proposals = payload["proposals"]

        self.assertEqual(2, len(proposals))
        self.assertEqual(1, proposals[0]["blocked"])
        self.assertEqual(1, proposals[1]["blocked"])
        self.assertEqual(0, proposals[0]["reached_env"])
        self.assertEqual(0, proposals[1]["reached_env"])
        self.assertEqual(1, proposals[1]["retry_exhausted"])
        self.assertEqual(2, payload["totals"]["intercepted"])


if __name__ == "__main__":
    unittest.main()