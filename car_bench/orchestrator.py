import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from car_bench.envs.base import Env
from car_bench.envs.tool_manipulation import (
    check_hallucinated_removed_part,
    remove_result_element,
    remove_tool_elements,
)
from car_bench.versions import version_info
from car_bench.types import (
    RESPOND_ACTION_NAME,
    USER_AS_A_TOOL_ACTION_NAMES,
    Action,
    AgentState,
    SolveResult,
)

if TYPE_CHECKING:
    from car_bench.agents.base import Agent


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    """Convert a single message to an action."""
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})


def message_to_actions(
    message: Dict[str, Any],
) -> List[Action]:
    """Convert a message to a list of actions."""
    actions = []
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
    ):
        for tool_call in message["tool_calls"]:
            if tool_call["function"] is not None:
                if tool_call["function"]["name"] == RESPOND_ACTION_NAME:
                    actions = [
                        Action(
                            name=tool_call["function"]["name"],
                            kwargs={
                                "content": tool_call["function"]["arguments"]["content"]
                            },
                        )
                    ]
                    return actions
                elif tool_call["function"]["name"] in USER_AS_A_TOOL_ACTION_NAMES:
                    actions = [
                        Action(
                            name=tool_call["function"]["name"],
                            kwargs={
                                "content": json.loads(
                                    tool_call["function"]["arguments"]
                                )["message_to_user"]
                            },
                        )
                    ]
                    return actions
                actions.append(
                    Action(
                        name=tool_call["function"]["name"],
                        kwargs=json.loads(tool_call["function"]["arguments"]),
                    )
                )
        return actions
    else:
        return [
            Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
        ]


class AgentOrchestrator:
    """Orchestrates the agent-environment interaction loop."""

    def __init__(
        self,
        agent: "Agent",
        remove_planning_tools: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            agent: The agent to orchestrate
            remove_planning_tools: Whether to remove planning/thinking tools
        """
        self.agent = agent
        self.remove_planning_tools = remove_planning_tools

    def execute(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 40
    ) -> SolveResult:
        """
        Execute the agent-environment interaction loop.

        Args:
            env: Environment to solve
            task_index: Optional task index
            max_num_steps: Maximum number of steps

        Returns:
            SolveResult with reward, messages, info, and total_cost
        """
        # Initialize environment
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        # Prepare tools: remove planning tools if configured, and handle test removals
        tools_info = env.tools_info
        if self.remove_planning_tools:
            tools_info = remove_tool_elements(
                tools_info, env.tools_info, ["planning_tool", "think"]
            )
        if env.task.removed_part:
            tools_info = remove_tool_elements(
                tools_info, env.tools_info, env.task.removed_part
            )

        # Let agent initialize its state
        system_prompt = env.wiki if env.wiki is not None else ""
        state = self.agent.get_init_state(system_prompt, obs)

        # Main interaction loop
        for step in range(max_num_steps):
            # Generate next message from agent
            next_message, state = self.agent.generate_next_message(state, tools_info)

            # Debug output
            if "reasoning_content" in next_message:
                print(f"🤖🐞💡 {next_message['reasoning_content']}")
            if next_message.get("tool_calls"):
                print(f"🤖🐞 {next_message['content']}")

            # Check if LLM hallucinated the removed part
            if env.task.removed_part:
                if (
                    "tool_calls" in next_message
                    and next_message["tool_calls"] is not None
                ):
                    hallucinated = check_hallucinated_removed_part(
                        env.task.removed_part,
                        next_message["tool_calls"],
                        env.task.task_type,
                    )
                    if hallucinated:
                        print(
                            f"🚨 Hallucination detected: LLM used removed part {env.task.removed_part}"
                        )

            # Convert message to actions
            actions = message_to_actions(next_message)

            # Execute actions in environment
            env_response = env.run_steps(actions, state.messages)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # Handle tool responses or user messages
            if (
                not actions[0].name in USER_AS_A_TOOL_ACTION_NAMES
                and not env_response.done
            ):
                # Add tool responses to messages
                for idx, tool_call in enumerate(next_message["tool_calls"]):
                    observation = env_response.observation[idx]

                    # Apply result removal if specified
                    if env.task.removed_part and any(
                        [
                            env.task.removed_part[i].split(".")[0] == "result"
                            for i in range(len(env.task.removed_part))
                        ]
                    ):
                        for i in range(len(env.task.removed_part)):
                            if (
                                env.task.removed_part[i].split(".")[0] == "result"
                                and tool_call["function"]["name"]
                                == env.task.removed_part[i].split(".")[1]
                            ):
                                observation_with_removed_part = remove_result_element(
                                    json.loads(observation),
                                    env.task.removed_part,
                                )
                                observation = json.dumps(observation_with_removed_part)

                    # Add tool response message
                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "content": observation,
                        }
                    )
            else:
                # This is a user interaction turn
                state = AgentState(
                    messages=state.messages,
                    total_cost=state.total_cost,
                    total_llm_induced_latency_ms=state.total_llm_induced_latency_ms,
                    turn_counter=state.turn_counter + 1,
                    least_prompt_tokens=state.least_prompt_tokens,
                    latest_prompt_tokens=state.latest_prompt_tokens,
                    total_prompt_tokens=state.total_prompt_tokens,
                    total_completion_tokens=state.total_completion_tokens,
                    start_time_s=state.start_time_s,
                )

                if (
                    actions[0].name != RESPOND_ACTION_NAME
                    and actions[0].name in USER_AS_A_TOOL_ACTION_NAMES
                ):
                    # Extract message from tool call for user interaction
                    next_message = {
                        "role": "assistant",
                        "content": json.loads(
                            next_message["tool_calls"][0]["function"]["arguments"]
                        )["message_to_user"],
                    }

                # Add user's response
                state.messages.append(
                    {"role": "user", "content": env_response.observation[0]}
                )

            if env_response.done:
                break

        # Compile final info with metrics
        info = {
            **info,
            "total_agent_cost": state.total_cost,
            "total_llm_induced_latency_ms": state.total_llm_induced_latency_ms,
            "average_llm_induced_latency_per_turn_ms": state.average_llm_induced_latency_per_turn_ms,
            "least_prompt_tokens": state.least_prompt_tokens,
            "latest_prompt_tokens": state.latest_prompt_tokens,
            "total_prompt_tokens": state.total_prompt_tokens,
            "total_completion_tokens": state.total_completion_tokens,
            "total_tokens": state.total_tokens,
            "elapsed_time_s": state.elapsed_time_s,
            "tokens_per_minute": state.tokens_per_minute,
            **version_info(),
        }

        # Add information about removed elements to info
        if env.task.removed_part:
            info["removed_part"] = env.task.removed_part

        return SolveResult(
            reward=reward, info=info, messages=state.messages, total_cost=state.total_cost
        )
