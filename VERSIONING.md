# Versioning

## Policy version (implemented)

`POLICY_VERSION` in `car_bench/versions.py` covers the policy the agent is given (`wiki.md`) and the
judge's gating/enhancement text (`reward_calculators.enhance_policy_line_with_context`). Bump it on every
change to either; `tests/test_versions.py` pins a hash per version and fails otherwise.

Every run result records `info.policy_version` (the version the agent saw) and `info.policy_fingerprint`
(hash of `wiki.md`). `reward_info.policy_version` records the version a score was *computed* with, so a
re-scored trajectory is distinguishable from a fresh rollout.

| Policy | Period | Notes |
|---|---|---|
| 1 | before 2026-07-27 | rule 020 not graded |
| 2 | 2026-07-27 (`987d7d2`) to policy 3 | rule 020 graded with the stacking text |
| 3 | route-policy rewrite (branch `route-policy-020-022`, being trialled) | rules 020-022 clarified (single-stop: user selects; multi-stop: fastest by default), id-based gating |

Result files written before versioning have no version fields: treat them as policy 1 or 2 by file date.

## Task versioning (NOT implemented, notes only)

Tasks and mock data (routes, weather, ...) are loaded from the HF dataset `johanneskirmayr/car-bench-dataset`
unpinned (only a `main` branch, no tags; last commit `1fcf24ad`, 2026-02-12). For the eventual release:

- Pin the dataset commit in code (`load_dataset(..., revision=...)` and `snapshot_download(..., revision=...)`),
  tag it on HF (e.g. `tasks-v2`), and record `tasks_version` / `dataset_revision` in results next to the policy fields.
- Ship policy and tasks as one pair (e.g. `policy-3.tasks-2`): fix the tasks listed in `TASK_FIXES_ROUTE_POLICY.md`,
  then bump both, bump the package version, and publish an updated leaderboard from fresh rollouts.
- Rule of thumb: policy-only changes can be re-scored (`r_policy` only, published as "re-scored"); task changes need
  fresh rollouts; headline numbers always need fresh rollouts because the agent prompt contains the wiki.
