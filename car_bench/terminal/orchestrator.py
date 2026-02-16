"""
Terminal Orchestrator for terminal-using agents.

Unlike AgentOrchestrator which does a turn-by-turn loop
(agent.generate_next_message -> parse tool_calls -> env.run_steps),
the TerminalOrchestrator:
1. Starts an HTTP tool server
2. Launches the terminal agent subprocess
3. The agent calls tools via MCP/HTTP autonomously
4. The tool server records all Actions and state hashes
5. User interaction is routed through the tool server
6. Messages are reconstructed from the interaction log for reward calculation
"""

import json
import subprocess
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from car_bench.envs.base import Env
from car_bench.envs.tool_manipulation import (
    check_hallucinated_removed_part,
    remove_tool_elements,
)
from car_bench.terminal.tool_docs_generator import generate_tool_docs_detailed
from car_bench.terminal.tool_server import ToolServer
from car_bench.types import (
    RESPOND_ACTION_NAME,
    USER_AS_A_TOOL_ACTION_NAMES,
    Action,
    AgentState,
    SolveResult,
)

if TYPE_CHECKING:
    from car_bench.agents.terminal_agent import TerminalAgent


class TerminalOrchestrator:
    """Orchestrates terminal agent ↔ environment interaction."""

    def __init__(
        self,
        agent: "TerminalAgent",
        remove_planning_tools: bool = True,
    ):
        self.agent = agent
        self.remove_planning_tools = remove_planning_tools

    def execute(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 40,
    ) -> SolveResult:
        """
        Execute the terminal agent against the environment.

        Args:
            env: Environment to solve
            task_index: Optional task index
            max_num_steps: Maximum number of steps (not directly enforced for
                          terminal agents, but the tool server can limit calls)

        Returns:
            SolveResult with reward, messages, info, and total_cost
        """
        # 1. Reset environment (same as standard path)
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()

        # 2. Prepare tools (same filtering as AgentOrchestrator)
        tools_info = env.tools_info
        if self.remove_planning_tools:
            tools_info = remove_tool_elements(
                tools_info, env.tools_info, ["planning_tool", "think"]
            )
        if env.task.removed_part:
            tools_info = remove_tool_elements(
                tools_info, env.tools_info, env.task.removed_part
            )

        # 3. Start HTTP tool server
        tool_server = ToolServer(
            tools_map=env.tools_map,
            data=env.data,
            tools_info=tools_info,
            env=env,
        )
        port = tool_server.start()
        print(f"Tool server started on port {port}")

        # 4. Build tool documentation and agent prompt
        system_prompt = env.wiki if env.wiki is not None else ""
        tool_docs = generate_tool_docs_detailed(tools_info)

        prompt = self.agent.build_agent_prompt(
            system_prompt=system_prompt,
            initial_observation=obs,
            tool_docs=tool_docs,
            server_port=port,
        )

        # 5. Set up working directory with MCP config
        working_dir = self.agent.setup_working_dir(port)

        # 6. Initialize agent state for logging
        state = self.agent.get_init_state(system_prompt, obs)

        # 7. Start user interaction loop in background thread
        user_thread = threading.Thread(
            target=self._user_interaction_loop,
            args=(tool_server, env, state),
            daemon=True,
        )
        user_thread.start()

        # 8. Launch agent subprocess
        env_vars = {
            "CAR_TOOL_SERVER_PORT": str(port),
            "CAR_TOOL_SERVER_URL": f"http://127.0.0.1:{port}",
        }

        process = self.agent.run_agent_process(
            prompt=prompt,
            env_vars=env_vars,
            working_dir=working_dir,
        )

        # 9. Wait for agent to finish (with timeout)
        try:
            process.wait(timeout=self.agent.timeout)
        except subprocess.TimeoutExpired:
            print("Agent timed out, terminating...")
            self.agent.stop_agent_process()

        # 10. Stop tool server
        tool_server.signal_done()
        user_thread.join(timeout=5)
        tool_server.stop()

        # 11. Collect recorded actions and state hashes
        actions = tool_server.get_actions()
        interaction_log = tool_server.get_interaction_log()

        # Inject into env for reward calculation
        env.actions = actions
        env.state_hashes = env.state_hashes  # Keep initial hashes from reset

        # Merge state hashes from tool server: the env already has [initial_hash]
        # from reset(); we need to add hashes recorded during tool execution.
        # The tool server records hashes via env._record_state_hash_if_needed()
        # which appends to env.state_hashes directly (via parent_context.run).

        # 12. Check for hallucinations in recorded tool calls
        if env.task.removed_part:
            for entry in interaction_log:
                if entry["type"] == "tool_call":
                    # Build a fake tool_calls structure for the checker
                    fake_tool_calls = [
                        {
                            "function": {
                                "name": entry["name"],
                                "arguments": json.dumps(entry["kwargs"]),
                            }
                        }
                    ]
                    hallucinated = check_hallucinated_removed_part(
                        env.task.removed_part,
                        fake_tool_calls,
                        env.task.task_type,
                    )
                    if hallucinated:
                        print(
                            f"Hallucination detected: agent used removed part "
                            f"{env.task.removed_part}"
                        )

        # 13. Reconstruct messages from interaction log
        messages = self._build_messages_from_log(
            state.messages, interaction_log
        )

        # 14. Ensure we have a respond action for reward calc
        if not actions or all(
            a.name not in USER_AS_A_TOOL_ACTION_NAMES for a in actions
        ):
            actions.append(
                Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={
                        "content": "(agent exited without responding to user)"
                    },
                )
            )
            env.actions = actions

        # 15. Calculate reward
        reward_res = env.calculate_reward(messages)
        reward = reward_res.reward
        info = {
            **info,
            **reward_res.info.model_dump(),
            "total_agent_cost": 0.0,  # Terminal agents have external cost tracking
            "total_llm_induced_latency_ms": 0.0,
            "average_llm_induced_latency_per_turn_ms": 0.0,
            "least_prompt_tokens": 0,
            "latest_prompt_tokens": 0,
        }

        if env.task.removed_part:
            info["removed_part"] = env.task.removed_part

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=0.0,
        )

    def _user_interaction_loop(
        self,
        tool_server: ToolServer,
        env: Env,
        state: AgentState,
    ):
        """
        Monitor the tool server for user messages and route through user sim.
        Runs in a background thread.
        """
        while True:
            # Block until agent sends a message to user
            agent_message = tool_server.get_pending_user_message()
            if agent_message is None:
                break  # Server stopped

            # Record state hash for the turn boundary
            # (mirroring how env.steps() records after user-facing actions)

            # Feed message to user simulator
            user_response = env.user.step(agent_message)

            # Check if conversation should end
            if "###STOP###" in user_response:
                # Send the STOP back so agent gets it, then signal done
                tool_server.set_user_response(user_response)
                tool_server.signal_done()
                break

            # Return user response to the agent
            tool_server.set_user_response(user_response)

            # Update turn counter
            state.turn_counter += 1

    def _build_messages_from_log(
        self,
        initial_messages: List[Dict[str, Any]],
        interaction_log: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert the tool server's interaction log into the standard message
        format expected by reward calculation (policy_evaluator.evaluate_aut(),
        calculate_policy_reward(), etc.).

        The format matches what AgentOrchestrator produces:
        - assistant messages with tool_calls array
        - tool response messages with tool_call_id
        - assistant text messages + user messages for conversations
        """
        messages = list(initial_messages)

        # Group consecutive tool calls into batches (one assistant message
        # can contain multiple tool_calls, as in the tool-calling path)
        current_tool_batch: List[Dict[str, Any]] = []

        for entry in interaction_log:
            if entry["type"] == "tool_call":
                current_tool_batch.append(entry)
            elif entry["type"] == "user_message":
                # Flush any pending tool calls first
                if current_tool_batch:
                    messages = self._flush_tool_batch(
                        messages, current_tool_batch
                    )
                    current_tool_batch = []

                # Add assistant message (what agent said to user)
                messages.append(
                    {
                        "role": "assistant",
                        "content": entry["agent_said"],
                    }
                )
                # Add user response
                messages.append(
                    {
                        "role": "user",
                        "content": entry["user_replied"],
                    }
                )

        # Flush remaining tool calls
        if current_tool_batch:
            messages = self._flush_tool_batch(messages, current_tool_batch)

        return messages

    def _flush_tool_batch(
        self,
        messages: List[Dict[str, Any]],
        tool_batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add a batch of tool calls as an assistant message with tool_calls
        array, followed by individual tool response messages.

        This matches the format that policy_evaluator.evaluate_aut() expects:
        - step["tool_calls"][i]["function"]["name"]
        - json.loads(step["tool_calls"][i]["function"]["arguments"])
        """
        # Build assistant message with tool_calls
        tool_calls = []
        for entry in tool_batch:
            tool_call_id = f"terminal_{entry['index']}"
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": entry["name"],
                        "arguments": json.dumps(entry["kwargs"]),
                    },
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            }
        )

        # Add tool response messages
        for i, entry in enumerate(tool_batch):
            tool_call_id = f"terminal_{entry['index']}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": entry["name"],
                    "content": entry["result"],
                }
            )

        return messages
