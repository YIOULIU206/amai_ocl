# AgentSpec-Commerce Matched Five-Profile Pilot

This directory preserves the matched five-profile comparison used to
validate the AgentSpec-Commerce baseline integration.

## Scope

- Profiles: 0–4
- Seeds: 42–46
- Model: Qwen-plus through DashScope
- Comparison: EGI, ToolGuard-Commerce, and AgentSpec-Commerce
- Evaluation type: matched descriptive pilot

## Main result

| Method | Valid success | Executed violation | Average rounds | Average latency |
|---|---:|---:|---:|---:|
| EGI | 5/5 (100%) | 0/5 (0%) | 4.40 | 109.81 s |
| AgentSpec-Commerce | 4/5 (80%) | 0/5 (0%) | 6.00 | 172.82 s |
| ToolGuard-Commerce | 3/5 (60%) | 0/5 (0%) | 7.40 | 496.92 s |

In this matched pilot, EGI achieved higher valid success than both
external baselines while maintaining the same zero executed-violation
rate. EGI also required fewer negotiation rounds and lower wall-clock
latency.

This is preliminary descriptive evidence and is not presented as a
statistical-significance claim. Seller reward is retained in the data
but is not used as the primary advantage claim.
