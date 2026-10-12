"""Policy version (see VERSIONING.md).

Deliberately a leaf module (no imports from the rest of the package) so it can be used from
anywhere without creating import cycles.

POLICY_VERSION covers the policy the agent is given (wiki.md) and the judge's gating/enhancement
text (reward_calculators.enhance_policy_line_with_context). Bump it on every change to either;
tests/test_versions.py fails otherwise.
"""
import hashlib
import os

POLICY_VERSION = "3"

_WIKI_PATH = os.path.join(os.path.dirname(__file__), "envs", "car_voice_assistant", "wiki.md")


def policy_fingerprint() -> str:
    """Short hash of the policy text, to detect an edit that did not bump POLICY_VERSION."""
    with open(_WIKI_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def version_info() -> dict:
    """Version metadata recorded in every run result."""
    return {
        "policy_version": POLICY_VERSION,
        "policy_fingerprint": policy_fingerprint(),
    }
