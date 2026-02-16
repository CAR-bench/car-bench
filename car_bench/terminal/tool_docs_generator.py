"""
Tool documentation generator for terminal agents.

Converts OpenAI function calling schemas into human-readable text
that can be included in system prompts for terminal-using agents.
"""

from typing import Any, Dict, List


def generate_tool_summary(tools_info: List[Dict[str, Any]]) -> str:
    """Generate a brief one-line-per-tool summary."""
    lines = []
    for tool in tools_info:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        # Truncate long descriptions
        if len(desc) > 100:
            desc = desc[:97] + "..."
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _format_parameter(
    name: str, prop: Dict[str, Any], required_params: List[str]
) -> str:
    """Format a single parameter for documentation."""
    ptype = prop.get("type", "string")
    desc = prop.get("description", "")
    req = "(required)" if name in required_params else "(optional)"
    parts = [f"  - {name} {req} [{ptype}]: {desc}"]

    enum_vals = prop.get("enum")
    if enum_vals:
        parts.append(f"    Options: {', '.join(str(v) for v in enum_vals)}")

    minimum = prop.get("minimum")
    maximum = prop.get("maximum")
    if minimum is not None or maximum is not None:
        range_parts = []
        if minimum is not None:
            range_parts.append(f"min={minimum}")
        if maximum is not None:
            range_parts.append(f"max={maximum}")
        parts.append(f"    Range: {', '.join(range_parts)}")

    multiple_of = prop.get("multipleOf")
    if multiple_of is not None:
        parts.append(f"    Step: {multiple_of}")

    # Handle array items
    if ptype == "array" and "items" in prop:
        items = prop["items"]
        items_desc = items.get("description", "")
        items_enum = items.get("enum")
        if items_enum:
            parts.append(
                f"    Array items: {items_desc} Options: {', '.join(str(v) for v in items_enum)}"
            )
        elif items_desc:
            parts.append(f"    Array items: {items_desc}")

    return "\n".join(parts)


def generate_tool_docs_detailed(tools_info: List[Dict[str, Any]]) -> str:
    """Generate detailed tool documentation with all parameters."""
    docs = []

    for tool in tools_info:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        lines = [f"### {name}", desc, ""]

        if properties:
            lines.append("Parameters:")
            for pname, pinfo in properties.items():
                lines.append(_format_parameter(pname, pinfo, required))
        else:
            lines.append("Parameters: (none)")

        lines.append("")
        docs.append("\n".join(lines))

    return "\n".join(docs)
