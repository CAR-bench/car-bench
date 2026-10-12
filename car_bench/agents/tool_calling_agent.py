import time
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from car_bench.agents.base import Agent
from car_bench.envs.base import Env
from car_bench.types import (
    AgentState,
    SolveResult,
)


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        thinking: bool = False,
        interleaved_thinking: bool = False,
        reasoning_effort: str = "low",
    ):
        self.tools_info = tools_info  # Not used directly anymore
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.thinking = thinking
        self.interleaved_thinking = interleaved_thinking
        self.reasoning_effort = reasoning_effort
        # Some newer Anthropic models (e.g. Sonnet 5) reject `temperature` outright;
        # once detected, stop sending it for the rest of this agent's calls.
        self._temperature_supported = True

    def get_init_state(
        self, system_prompt: str, initial_observation: str
    ) -> AgentState:
        """
        Get the initial state of the agent.

        Args:
            system_prompt: System prompt (domain policy/wiki)
            initial_observation: Initial user message/observation

        Returns:
            Initial AgentState with messages and metrics initialized
        """
        return AgentState(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_observation},
            ],
            start_time_s=time.time(),
        )

    def generate_next_message(
        self, state: AgentState, tools_info: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], AgentState]:
        """
        Generate the next message from the LLM.

        Args:
            state: Current agent state with messages and metrics
            tools_info: List of tool definitions (potentially modified for task)

        Returns:
            Tuple of (next_message, updated_state)
        """
        for i in range(4):
            try:
                # Configure prompt caching
                tools_info[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                state.messages[0]["cache_control"] = {"type": "ephemeral"}

                completion_kwargs = {
                    "model": self.model,
                    "custom_llm_provider": self.provider,
                    "tools": tools_info,
                }
                if self._temperature_supported:
                    completion_kwargs["temperature"] = self.temperature

                if self.thinking:
                    if self.reasoning_effort in [
                        "none",
                        "disable",
                        "low",
                        "medium",
                        "high",
                    ]:
                        completion_kwargs["reasoning_effort"] = self.reasoning_effort
                    else:
                        try:
                            thinking_budget = int(self.reasoning_effort)
                        except ValueError:
                            raise ValueError(
                                "reasoning_effort must be 'none', 'disable', 'low', 'medium', 'high', or an integer value"
                            )
                        if self.model == "claude-opus-4-6" and self.provider == "anthropic":
                            completion_kwargs["thinking"] = {
                                "type": "adaptive"
                            }
                        else:
                            completion_kwargs["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": thinking_budget,
                            }

                    if self.interleaved_thinking:
                        if self.provider == "bedrock":
                            completion_kwargs["anthropic_beta"] = [
                                "interleaved-thinking-2025-05-14"
                            ]
                        else:
                            completion_kwargs["extra_headers"] = {
                                "anthropic-beta": "interleaved-thinking-2025-05-14"
                            }

                res = completion(
                    messages=state.messages,
                    **completion_kwargs,
                )
                next_message = res.choices[0].message.model_dump()
                break
            except Exception as e:
                print(f"Error calling LLM: {e}.")
                if "temperature" in str(e) and "deprecated" in str(e):
                    self._temperature_supported = False
                    continue
                if i == 3:
                    raise e
                time.sleep(120)

        # Update metrics in state
        latest_prompt_tokens = res.usage.prompt_tokens
        completion_tokens = getattr(res.usage, "completion_tokens", 0) or 0
        least_prompt_tokens = state.least_prompt_tokens
        if latest_prompt_tokens < least_prompt_tokens and latest_prompt_tokens > 0:
            least_prompt_tokens = latest_prompt_tokens

        total_cost = state.total_cost + (
            res._hidden_params["response_cost"]
            if "response_cost" in res._hidden_params
            and res._hidden_params["response_cost"] is not None
            else 0.0
        )
        total_llm_induced_latency_ms = (
            state.total_llm_induced_latency_ms + res._response_ms
        )

        total_prompt_tokens = state.total_prompt_tokens + latest_prompt_tokens
        total_completion_tokens = state.total_completion_tokens + completion_tokens

        # Append next message to messages
        messages = state.messages + [next_message]

        # Create updated state
        updated_state = AgentState(
            messages=messages,
            total_cost=total_cost,
            total_llm_induced_latency_ms=total_llm_induced_latency_ms,
            turn_counter=state.turn_counter,
            least_prompt_tokens=least_prompt_tokens,
            latest_prompt_tokens=latest_prompt_tokens,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            start_time_s=state.start_time_s,
        )

        return next_message, updated_state

