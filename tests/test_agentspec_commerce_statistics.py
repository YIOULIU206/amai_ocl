"""Statistics tests for AgentSpec-Commerce."""

from __future__ import annotations

import unittest

from aimai_ocl.schemas import EpisodeTrace
from aimai_ocl.statistics import (
    collect_agentspec_stats,
    summarize_records,
)


def make_trace(
    metadata: dict | None = None,
) -> EpisodeTrace:
    return EpisodeTrace(
        episode_id="agentspec-statistics-test",
        env_id="FakeCommerce-v0",
        metadata=dict(metadata or {}),
    )


class AgentSpecCommerceStatisticsTests(unittest.TestCase):
    def test_collect_agentspec_stats(self) -> None:
        trace = make_trace(
            {
                "agentspec_commerce": {
                    "baseline": "agentspec_commerce",
                    "rounds": [
                        {
                            "decisions": [
                                "llm_self_reflect",
                                "continue",
                            ],
                            "self_reflection_calls": 1,
                            "selected_attempt": 1,
                            "no_op": False,
                            "reached_env": 1,
                            "attempts": [
                                {
                                    "attempt": 0,
                                    "allowed": False,
                                    "runtime_sec": 0.01,
                                },
                                {
                                    "attempt": 1,
                                    "allowed": True,
                                    "runtime_sec": 0.02,
                                },
                            ],
                        },
                        {
                            "decisions": ["continue"],
                            "self_reflection_calls": 0,
                            "selected_attempt": 0,
                            "no_op": False,
                            "reached_env": 1,
                            "attempts": [
                                {
                                    "attempt": 0,
                                    "allowed": True,
                                    "runtime_sec": 0.005,
                                }
                            ],
                        },
                        {
                            "decisions": [
                                "llm_self_reflect",
                                "llm_self_reflect",
                            ],
                            "self_reflection_calls": 1,
                            "selected_attempt": None,
                            "no_op": True,
                            "reached_env": 0,
                            "attempts": [
                                {
                                    "attempt": 0,
                                    "allowed": False,
                                    "runtime_sec": 0.01,
                                },
                                {
                                    "attempt": 1,
                                    "allowed": False,
                                    "runtime_sec": 0.015,
                                },
                            ],
                        },
                    ],
                }
            }
        )

        result = collect_agentspec_stats(trace)

        self.assertEqual(3, result["agentspec_proposal_count"])
        self.assertEqual(2, result["agentspec_detected_count"])
        self.assertEqual(2, result["agentspec_intercepted_count"])
        self.assertEqual(
            2,
            result["agentspec_self_reflection_calls"],
        )
        self.assertEqual(
            1,
            result["agentspec_revision_passed_count"],
        )
        self.assertEqual(
            1,
            result["agentspec_retry_exhausted_count"],
        )
        self.assertEqual(
            2,
            result["agentspec_candidate_selected_count"],
        )
        self.assertEqual(1, result["agentspec_no_op_count"])
        self.assertEqual(
            2,
            result["agentspec_reached_env_count"],
        )
        self.assertAlmostEqual(
            0.06,
            result["agentspec_policy_runtime_sec"],
        )
        self.assertAlmostEqual(
            2 / 3,
            result["agentspec_intercept_rate"],
        )
        self.assertAlmostEqual(
            0.5,
            result["agentspec_revision_pass_rate"],
        )

    def test_missing_metadata_returns_zero_metrics(self) -> None:
        result = collect_agentspec_stats(make_trace())

        self.assertEqual(0, result["agentspec_proposal_count"])
        self.assertEqual(0, result["agentspec_detected_count"])
        self.assertEqual(0, result["agentspec_no_op_count"])
        self.assertEqual(0.0, result["agentspec_intercept_rate"])
        self.assertEqual(
            0.0,
            result["agentspec_revision_pass_rate"],
        )

    def test_summarize_records_aggregates_agentspec_metrics(
        self,
    ) -> None:
        records = [
            {
                "arm": "agentspec_commerce",
                "success": 1,
                "has_violation": 0,
                "has_executed_violation": 0,
                "valid_success": 1,
                "unsafe_success": 0,
                "round": 2,
                "seller_reward": 1.0,
                "latency_sec": 1.5,
                "audit_events": 0,
                "agentspec_proposal_count": 2,
                "agentspec_detected_count": 1,
                "agentspec_intercepted_count": 1,
                "agentspec_self_reflection_calls": 1,
                "agentspec_revision_passed_count": 1,
                "agentspec_retry_exhausted_count": 0,
                "agentspec_candidate_selected_count": 2,
                "agentspec_no_op_count": 0,
                "agentspec_reached_env_count": 2,
                "agentspec_policy_runtime_sec": 0.03,
            },
            {
                "arm": "agentspec_commerce",
                "success": 0,
                "has_violation": 0,
                "has_executed_violation": 0,
                "valid_success": 0,
                "unsafe_success": 0,
                "round": 1,
                "seller_reward": 0.0,
                "latency_sec": 1.0,
                "audit_events": 0,
                "agentspec_proposal_count": 1,
                "agentspec_detected_count": 1,
                "agentspec_intercepted_count": 1,
                "agentspec_self_reflection_calls": 1,
                "agentspec_revision_passed_count": 0,
                "agentspec_retry_exhausted_count": 1,
                "agentspec_candidate_selected_count": 0,
                "agentspec_no_op_count": 1,
                "agentspec_reached_env_count": 0,
                "agentspec_policy_runtime_sec": 0.03,
            },
        ]

        summaries = summarize_records(records)

        self.assertEqual(1, len(summaries))
        summary = summaries[0]

        self.assertEqual(
            3,
            summary["total_agentspec_proposals"],
        )
        self.assertEqual(
            2,
            summary["total_agentspec_detected"],
        )
        self.assertEqual(
            2,
            summary["total_agentspec_intercepted"],
        )
        self.assertEqual(
            2,
            summary["total_agentspec_self_reflections"],
        )
        self.assertEqual(
            1,
            summary["total_agentspec_revision_passed"],
        )
        self.assertEqual(
            1,
            summary["total_agentspec_retry_exhausted"],
        )
        self.assertEqual(
            2,
            summary["total_agentspec_candidate_selected"],
        )
        self.assertEqual(
            1,
            summary["total_agentspec_no_op"],
        )
        self.assertEqual(
            2,
            summary["total_agentspec_reached_env"],
        )
        self.assertAlmostEqual(
            0.06,
            summary["total_agentspec_policy_runtime_sec"],
        )
        self.assertAlmostEqual(
            2 / 3,
            summary["agentspec_intercept_rate"],
        )
        self.assertAlmostEqual(
            0.5,
            summary["agentspec_revision_pass_rate"],
        )
        self.assertAlmostEqual(
            0.02,
            summary["avg_agentspec_policy_runtime_sec"],
        )


if __name__ == "__main__":
    unittest.main()
