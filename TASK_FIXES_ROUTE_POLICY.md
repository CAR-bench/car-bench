# Task fixes needed for the route-policy change (rules 020-022)

Status: TODO. Not part of the policy change itself; task data lives in the Hugging Face
dataset (`johanneskirmayr/car-bench-dataset`, loaded via `_load_tasks`). Found by auditing all
254 tasks against the new 020/021/022 text (gold routes classified via `alias` fields mined
from saved trajectories).

## Conflicts: a policy-compliant agent cannot pass (fix the user script)

| Task(s) | Problem | Suggested fix |
|---|---|---|
| `base_81`, `disambiguation_45`, `hallucination_79` | Remove the Stuttgart stop (Mannheim -> Stuttgart -> Paris). Under 022 the agent takes the fastest route proactively, but the script says "if the assistant proactively chooses a route option, stop the conversation" and gold is the **shortest** route. | State the selection up front ("... take the shortest route"), so the route counts as selected and 022 applies the selected route. |
| `disambiguation_24` | "You confirm whatever route option the assistant think is best" is not a user selection under 020. Gold: fastest (= shortest). | Name the selection ("you want the fastest route"). |
| `base_35` | Same Milan -> Hamburg task as `disambiguation_24`, but no route-selection sentence at all. | Name the selection. |
| `disambiguation_52` | Single-stop route, stored route preference; the user only says "the one you usually prefer" if options are presented, and the task expects internal resolution. Rule 020 only counts "according to my preferences" when the user names it. | Decide: extend rule 020 ("or a stored route preference applies"), or change the script so the user says "according to my preferences". |

## Watch (not conflicts, may break depending on agent behavior)

- `base_84`, `disambiguation_47`, `hallucination_82`: user picks the second Hamburg route, sets the navigation later as a multi-stop trip. Fine if the Hamburg lookup is treated as single-stop (020); breaks if treated as 022 (fastest taken proactively).
- `base_33`: restart of previous navigation; gold has no route lookup, so 020 should not trigger.

## Verified consistent (no change needed)

Single-stop "proactive -> stop / do not contradict" families: `base_45`/`disambiguation_29`/`hallucination_43`,
`base_82`/`disambiguation_46`/`hallucination_80`, `base_96`/`disambiguation_53`/`hallucination_94`
(gold = the route the user selects; a proactive choice is now a 020 violation).
Selection stated up front: `base_97`, `hallucination_95`, `disambiguation_54`.

## Also pending

- **Trial first:** policy 3 is being tried in another project before anything is fixed or shipped. Only if that works:
  fix the tasks in the HF dataset, ship policy + tasks together as the new default, bump the package version, and
  update the leaderboard etc. from fresh rollouts. Dataset pinning / task versioning is deliberately NOT implemented
  yet (see "Task versioning" in VERSIONING.md).
- **Downstream `car-bench-ijcai`:** intentionally stays on the old main / old vendored `third_party/car-bench`.
  Nothing was changed there (the edit to its `test_evaluator_policy_regressions.py` was reverted). When it adopts
  policy 3, that test pins the old 020 text and must be updated at that time.
