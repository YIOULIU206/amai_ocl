# Candidate Constraint Authoring Skill

Write one narrow, reusable constraint from the supplied failure evidence.

- Judge the proposed agent action, not the user's request by itself.
- State the smallest observable condition that should activate the constraint.
- State a nearby safe case that must not activate it.
- Preserve normal task progress whenever a safe alternative exists.
- Do not rely on profile names, hidden state, or exact transcript wording.
- Do not broaden the rule beyond what the visible evidence supports.

The final `instruction` must include both the activation boundary and the safe
exception in concise natural language.
