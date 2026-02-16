"""
Terminal Agent for CAR-bench.

A generic agent that wraps terminal-using LLM agents (Claude Code, Devin,
OpenHands, etc.). Instead of making LLM API calls with tool schemas, this
agent launches a subprocess that interacts with tools via MCP or CLI commands.

The agent is agnostic to the specific terminal agent implementation. Different
agents can be invoked by changing the command and arguments.
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from car_bench.agents.base import Agent
from car_bench.types import AgentState


class TerminalAgent(Agent):
    def __init__(
        self,
        agent_command: str,
        agent_args: Optional[List[str]] = None,
        tools_info: Optional[List[Dict[str, Any]]] = None,
        wiki: str = "",
        timeout: int = 300,
    ):
        """
        Initialize the terminal agent.

        Args:
            agent_command: Command to launch the terminal agent
                          (e.g., "claude", "python my_agent.py")
            agent_args: Additional arguments for the agent command
            tools_info: Tool schemas (used for documentation generation)
            wiki: Wiki/policy information
            timeout: Maximum seconds per task
        """
        self.agent_command = agent_command
        self.agent_args = agent_args or []
        self.tools_info = tools_info or []
        self.wiki = wiki
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self._working_dir: Optional[str] = None

    def get_init_state(
        self, system_prompt: str, initial_observation: str
    ) -> AgentState:
        """
        Initialize agent state. For terminal agents, messages serve as an
        interaction log rather than an LLM conversation history.
        """
        return AgentState(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_observation},
            ]
        )

    def generate_next_message(
        self, state: AgentState, tools_info: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], AgentState]:
        """
        Not used for terminal agents. The TerminalOrchestrator manages
        execution directly via run_agent_process().
        """
        raise NotImplementedError(
            "TerminalAgent uses run_agent_process() via TerminalOrchestrator, "
            "not the standard generate_next_message() loop."
        )

    def build_agent_prompt(
        self,
        system_prompt: str,
        initial_observation: str,
        tool_docs: str,
        server_port: int,
    ) -> str:
        """
        Build the complete prompt/instructions for the terminal agent.

        Includes the system prompt (wiki/policy), tool documentation,
        and instructions on how to interact with the benchmark.

        Args:
            system_prompt: Domain policy/wiki
            initial_observation: Initial user message
            tool_docs: Human-readable tool documentation
            server_port: HTTP tool server port

        Returns:
            Complete prompt string
        """
        return f"""You are an in-car voice assistant. You must help the user (driver) with their request by using the available tools.

## Domain Policy
{system_prompt}

## Available Tools
You have access to car control and information tools via MCP (Model Context Protocol).
These tools are provided by the "car-bench-tools" MCP server.

The tools include vehicle controls (climate, lights, windows, sunroof), navigation,
weather, contacts, calendar, charging information, and more.

There is also a special tool: `send_message_to_user`
- Use this to communicate with the user/driver
- Input: {{"message": "your message"}}
- Returns: the user's response
- You MUST use this tool to talk to the user. Do NOT just print text.

## Tool Reference
{tool_docs}

## Important Instructions
1. Use the MCP tools to fulfill the user's request
2. Follow the domain policy above carefully
3. When you need to communicate with the user, use the `send_message_to_user` tool
4. When done, send a final message to the user with `send_message_to_user`
5. After the user confirms they're satisfied, you can exit

## Current User Request
The user (driver) says: "{initial_observation}"

Begin by analyzing the request and using the appropriate tools.
"""

    def setup_working_dir(self, server_port: int) -> str:
        """
        Create a temporary working directory with MCP server configuration.

        For Claude Code, this writes a .mcp.json file. Other agents may
        need different configuration files.

        Args:
            server_port: HTTP tool server port

        Returns:
            Path to the working directory
        """
        self._working_dir = tempfile.mkdtemp(prefix="car_bench_terminal_")

        # Write MCP server configuration for Claude Code
        mcp_config = {
            "mcpServers": {
                "car-bench-tools": {
                    "command": sys.executable,
                    "args": ["-m", "car_bench.terminal.mcp_server"],
                    "env": {
                        "CAR_TOOL_SERVER_URL": f"http://127.0.0.1:{server_port}",
                    },
                }
            }
        }

        mcp_config_path = os.path.join(self._working_dir, ".mcp.json")
        with open(mcp_config_path, "w") as f:
            json.dump(mcp_config, f, indent=2)

        # Also write a task prompt file that agents can read
        prompt_path = os.path.join(self._working_dir, "TASK.md")
        # This will be populated later by the orchestrator if needed

        return self._working_dir

    def run_agent_process(
        self,
        prompt: str,
        env_vars: Dict[str, str],
        working_dir: str,
    ) -> subprocess.Popen:
        """
        Launch the terminal agent subprocess.

        Args:
            prompt: Complete prompt for the agent
            env_vars: Environment variables to set
            working_dir: Working directory for the agent

        Returns:
            The subprocess.Popen object
        """
        full_env = {**os.environ, **env_vars}

        # Write prompt to a file in the working directory
        prompt_path = os.path.join(working_dir, "TASK.md")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        # Build command
        cmd = [self.agent_command] + self.agent_args

        print(f"Launching terminal agent: {' '.join(cmd)}")
        print(f"Working directory: {working_dir}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            cwd=working_dir,
            text=True,
        )

        # Feed prompt via stdin if the agent reads from it
        if self.process.stdin:
            try:
                self.process.stdin.write(prompt)
                self.process.stdin.flush()
                self.process.stdin.close()
            except BrokenPipeError:
                # Agent may not read stdin (e.g., reads from TASK.md instead)
                pass

        return self.process

    def stop_agent_process(self):
        """Terminate the agent process gracefully, then force-kill if needed."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
