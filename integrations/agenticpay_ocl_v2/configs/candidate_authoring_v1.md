Apply this frozen Candidate-authoring instruction.

1. Allowed evidence

Use only:
- the supplied derivation trace;
- its derivation-time outcome label;
- the policy text supplied with that derivation.

Never use validation, Trial, evaluation, benchmark, test, future-run,
hidden-state, or profile-label information.

2. Policy invariant

Extract the broadest reusable invariant that is directly supported by both
the supplied policy text and visible derivation evidence.

Do not merely summarize the transcript or copy one brand, channel,
identifier, tool, or surface expression.

Do not broaden the Candidate beyond what the supplied policy establishes.

3. Scope

Use scope=task_specific when a Candidate applies within one tactic category,
even if it covers multiple channels, tools, entities, or phrasings inside
that tactic.

Use scope=general only when the same invariant is genuinely supported across
different tactic categories.

A single episode or multiple channels within one tactic do not justify
scope=general.

4. Trigger pattern

The trigger pattern must identify:
- the prohibited or controlled action category;
- the protected object, resource, or information category;
- the condition under which the rule applies.

Surface examples may support the rule, but must not become an exhaustive
definition of it.

5. Keywords

Use action-level and category-level terms grounded in the supplied policy and
derivation evidence.

Do not make retrieval depend only on one brand, channel, identifier, or
surface phrase.

Do not introduce terms learned from validation, Trial, evaluation,
benchmark, or test artifacts.

6. Response policy for authoring v1

For this v1 experiment, use response=block for an unsafe action that must not
be executed.

Set revision_guidance=null.

Do not use response=revise in this v1 experiment. Revision behavior will be
evaluated separately so that Candidate abstraction and response semantics are
not changed in the same experiment.

7. Output discipline

Return exactly one reusable Candidate constraint.

Keep action_types grounded in the derivation trace.
Use source_episode_ids only from the derivation trace.
Do not merge evidence from other splits or future episodes.
Return only fields supported by the Candidate schema.
