#!/usr/bin/env python3
"""
MCP Server for CAR-bench tools.

A standalone MCP (Model Context Protocol) server that acts as a thin proxy
to the CAR-bench HTTP Tool Server. Terminal-using agents (Claude Code, etc.)
connect to this via stdio transport to discover and call benchmark tools.

Usage:
    CAR_TOOL_SERVER_URL=http://localhost:8765 python -m car_bench.terminal.mcp_server

The server exposes:
- All benchmark tools (fetched dynamically from the HTTP tool server)
- A special 'send_message_to_user' tool for agent↔user interaction
"""

import json
import os
import sys
from typing import Any

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

SERVER_URL = os.environ.get("CAR_TOOL_SERVER_URL", "http://localhost:8765")

app = Server("car-bench-tools")


def _fetch_tools() -> list[dict[str, Any]]:
    """Fetch tool schemas from the HTTP tool server."""
    try:
        resp = requests.get(f"{SERVER_URL}/tools", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching tools from {SERVER_URL}: {e}", file=sys.stderr)
        return []


def _openai_schema_to_mcp_tool(tool_schema: dict[str, Any]) -> Tool:
    """Convert an OpenAI function calling schema to an MCP Tool."""
    func = tool_schema.get("function", {})
    return Tool(
        name=func.get("name", "unknown"),
        description=func.get("description", ""),
        inputSchema=func.get("parameters", {"type": "object", "properties": {}}),
    )


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools from the HTTP tool server."""
    tool_schemas = _fetch_tools()
    tools = [_openai_schema_to_mcp_tool(schema) for schema in tool_schemas]

    # Add the special user interaction tool
    tools.append(
        Tool(
            name="send_message_to_user",
            description=(
                "Send a message to the user (driver) and receive their response. "
                "Use this to communicate with the user, ask questions, confirm actions, "
                "or provide information. The user will respond to your message."
            ),
            inputSchema={
                "type": "object",
                "required": ["message"],
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the user/driver.",
                    }
                },
                "additionalProperties": False,
            },
        )
    )

    return tools


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Call a tool via the HTTP tool server."""

    # Handle the special user interaction tool
    if name == "send_message_to_user":
        message = arguments.get("message", "")
        try:
            resp = requests.post(
                f"{SERVER_URL}/user/message",
                json={"message": message},
                timeout=300,  # Long timeout: blocks until user responds
            )
            resp.raise_for_status()
            result = resp.json()
            user_response = result.get("user_response", "")
            return [TextContent(type="text", text=user_response)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error communicating with user: {e}")]

    # Regular tool call
    try:
        resp = requests.post(
            f"{SERVER_URL}/tool/{name}",
            json=arguments,
            timeout=60,
        )
        result_text = resp.text
        return [TextContent(type="text", text=result_text)]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
