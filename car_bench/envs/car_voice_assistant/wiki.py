import os
import re

FOLDER_PATH = os.path.dirname(__file__)

with open(os.path.join(FOLDER_PATH, "wiki.md"), "r") as f:
    WIKI_RAW = f.read()
    WIKI = WIKI_RAW
    WIKI = WIKI.replace("INS:", "")
    WIKI = WIKI.replace("AUT-POL:", "")
    WIKI = WIKI.replace("LLM-POL:", "")
    WIKI_LLM_POL_LINES = []
    for line in WIKI_RAW.split("\n\n"):
        if line.startswith("- "):
            line = line[2:]
            if any([llm_policy_tag in line for llm_policy_tag in ["LLM-POL"]]):
                line.replace("LLM-POL", "")
                WIKI_LLM_POL_LINES.append(line.strip())
    WIKI_AUT_POL_LINES = []
    for line in WIKI_RAW.split("\n\n"):
        if line.startswith("- "):
            line = line[2:]
            if any([aut_policy_tag in line for aut_policy_tag in ["AUT-POL"]]):
                line.replace("AUT-POL", "")
                WIKI_AUT_POL_LINES.append(line.strip())
    pass


# The graded lines keep their tag (the replace calls above do not assign), so a rule is
# identified by its id, not by its wording: rewording a rule cannot silently skip its gating.
_RULE_ID = re.compile(r"^LLM-POL:(\d{3}):")


def policy_rule_id(line):
    """The id of a graded LLM policy line ("020" for "LLM-POL:020:..."), or None."""
    match = _RULE_ID.match(line)
    return match.group(1) if match else None


WIKI_LLM_POL_RULES = {policy_rule_id(line): line for line in WIKI_LLM_POL_LINES}
