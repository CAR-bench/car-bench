import unittest

from car_bench.envs.car_voice_assistant.wiki import (
    WIKI_LLM_POL_LINES,
    WIKI_LLM_POL_RULES,
    policy_rule_id,
)
from car_bench.envs.reward_calculators import enhance_policy_line_with_context

ROUTE_LOOKUP = "get_routes_from_start_to_destination"
# rule id -> (actions that make it applicable, a phrase only its enhancement adds)
GATED_RULES = {
    "004": ({"send_email"}, "The tools that require confirmation are"),
    "007": ({"open_close_window"}, "this policy is one-way"),
    "008": ({"open_close_sunroof"}, None),
    "012": ({"set_climate_temperature"}, None),
    "020": ({ROUTE_LOOKUP}, "single-stop route whose route the user has not selected"),
    "021": ({ROUTE_LOOKUP}, "presented in detail or set in the navigation system"),
    "022": ({ROUTE_LOOKUP}, "Mark it not applicable for single-stop routes"),
}
SELECTED = "The user has selected a route when they named it, in this or an earlier turn"


def _enhance(rule_id, actions, split=True):
    return enhance_policy_line_with_context(
        WIKI_LLM_POL_RULES[rule_id],
        performed_action_names=actions,
        difference_is_more_than_3_degrees=split,
    )


class PolicyGatingTest(unittest.TestCase):
    def test_every_graded_line_has_exactly_one_unique_id(self) -> None:
        ids = [policy_rule_id(line) for line in WIKI_LLM_POL_LINES]
        self.assertNotIn(None, ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {"002", "004", "007", "008", "012", "020", "021", "022"})
        self.assertEqual(set(WIKI_LLM_POL_RULES), set(ids))

    def test_gated_rules_are_skipped_without_their_trigger(self) -> None:
        for rule_id in GATED_RULES:
            with self.subTest(rule=rule_id):
                self.assertIsNone(_enhance(rule_id, {"get_weather"}, split=False))

    def test_gated_rules_are_graded_with_their_trigger(self) -> None:
        for rule_id, (actions, marker) in GATED_RULES.items():
            with self.subTest(rule=rule_id):
                enhanced = _enhance(rule_id, actions)
                self.assertIsNotNone(enhanced)
                self.assertTrue(enhanced.startswith(WIKI_LLM_POL_RULES[rule_id]))
                if marker is not None:
                    self.assertIn(marker, enhanced)

    def test_ungated_rule_is_returned_bare(self) -> None:
        self.assertEqual(_enhance("002", set(), split=False), WIKI_LLM_POL_RULES["002"])

    def test_rule_020_covers_single_stop_routes_only(self) -> None:
        line = _enhance("020", {ROUTE_LOOKUP})
        self.assertIn("single-stop route", line)
        self.assertIn("Multi-stop routes are not covered by this rule", line)
        self.assertIn(SELECTED, line)

    def test_rule_022_covers_multi_stop_routes_and_route_changes(self) -> None:
        line = _enhance("022", {ROUTE_LOOKUP})
        self.assertIn("change to an active route with intermediate stops", line)
        self.assertIn("no count of alternatives is required", line)
        self.assertIn(SELECTED, line)

    def test_rule_021_covers_routes_set_in_the_navigation_system(self) -> None:
        self.assertIn("or set in the navigation system", _enhance("021", {ROUTE_LOOKUP}))

    def test_rule_020_enhancement_does_not_override_route_selection(self) -> None:
        enhanced = _enhance("020", {ROUTE_LOOKUP})
        self.assertNotIn("Even with prior authorization", enhanced)
        self.assertIn("after the user selected the route", enhanced)
        self.assertIn("before the user selected it does not follow the policy", enhanced)


if __name__ == "__main__":
    unittest.main()
