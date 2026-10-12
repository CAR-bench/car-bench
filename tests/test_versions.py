import hashlib
import inspect
import unittest

from car_bench import versions
from car_bench.envs.car_voice_assistant import wiki
from car_bench.envs.reward_calculators import enhance_policy_line_with_context
from car_bench.types import RewardInfo

# POLICY_VERSION -> hash of wiki.md + the judge enhancement code. If this test fails you edited
# the policy or the judge text: bump POLICY_VERSION in car_bench/versions.py and add a new pin.
PINNED_POLICIES = {
    "3": "bccb3ae5654e",
}


def _policy_hash() -> str:
    with open(versions._WIKI_PATH, "rb") as f:
        wiki_bytes = f.read()
    code = inspect.getsource(enhance_policy_line_with_context).encode()
    return hashlib.sha256(wiki_bytes + code).hexdigest()[:12]


class VersionsTest(unittest.TestCase):
    def test_policy_text_change_requires_version_bump(self) -> None:
        self.assertIn(versions.POLICY_VERSION, PINNED_POLICIES)
        self.assertEqual(
            _policy_hash(),
            PINNED_POLICIES[versions.POLICY_VERSION],
            "wiki.md or enhance_policy_line_with_context changed without a POLICY_VERSION bump",
        )

    def test_wiki_reexports_policy_version(self) -> None:
        self.assertEqual(wiki.POLICY_VERSION, versions.POLICY_VERSION)

    def test_version_info_has_policy_version_and_fingerprint(self) -> None:
        info = versions.version_info()
        self.assertEqual(set(info), {"policy_version", "policy_fingerprint"})
        self.assertEqual(info["policy_version"], versions.POLICY_VERSION)
        self.assertEqual(len(info["policy_fingerprint"]), 12)

    def test_reward_info_carries_policy_version(self) -> None:
        self.assertIn("policy_version", RewardInfo.model_fields)


if __name__ == "__main__":
    unittest.main()
