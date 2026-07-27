import asyncio
import json
import random
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from car_bench.envs.policy_evaluator import (
    PolicyEvaluatorStrategy,
    load_policy_evaluator,
    policy_errors_during_runtime,
)
from car_bench.envs.reward_calculators import (
    calculate_end_conversation_reward,
    calculate_output_reward,
    calculate_policy_reward,
    calculate_state_based_reward,
    calculate_tool_execution_reward,
    calculate_tool_subset_reward,
)
from car_bench.envs.tool import Tool
from car_bench.envs.tool_execution_error_evaluator import (
    tool_execution_errors_during_runtime,
)
from car_bench.envs.user.user import UserStrategy, load_user
from car_bench.envs.user.user_end_conversation import end_conversation_failure
from car_bench.types import (
    RESPOND_ACTION_NAME,
    USER_AS_A_TOOL_ACTION_NAMES,
    Action,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RewardInfo,
    RewardResult,
    Task,
    TaskType,
)

ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]


def to_hashable(item: ToHashable) -> Hashable:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item


def consistent_hash(
    value: Hashable,
) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


def _record_tool_runtime_error(action_name: str, exc: Exception) -> None:
    try:
        errors = tool_execution_errors_during_runtime.get()
    except LookupError:
        return

    if isinstance(errors, list):
        errors.append(f"{action_name}: {type(exc).__name__}: {exc}")


class Env(object):
    def __init__(
        self,
        data_load_func: Callable[[], Dict[str, Any]],
        tools: List[Type[Tool]],
        tasks: List[Task],
        wiki: str,
        user_strategy: Union[str, UserStrategy],
        user_model: str,
        user_provider: Optional[str] = None,
        user_thinking: bool = False,
        task_index: Optional[int] = None,
        policy_evaluator_strategy: Optional[Union[str, PolicyEvaluatorStrategy]] = None,
        policy_evaluator_model: Optional[str] = None,
        policy_evaluator_provider: Optional[str] = None,
        evaluate_policy: Optional[bool] = False,
        score_tool_execution_errors: Optional[bool] = False,
        score_policy_errors: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.data_load_func = data_load_func
        self.data = data_load_func()
        self.tools_map: Dict[str, Type[Tool]] = {
            tool.get_info()["function"]["name"]: tool for tool in tools
        }
        self.tools_info = [tool.get_info() for tool in tools]
        self.terminate_tools = []
        self.tasks = tasks
        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks) - 1)
        self.task = tasks[self.task_index]
        self.wiki = wiki
        self.user = load_user(
            user_strategy=user_strategy,
            model=user_model,
            provider=user_provider,
            user_thinking=user_thinking,
        )
        self.policy_evaluator = load_policy_evaluator(
            policy_evaluator_strategy=policy_evaluator_strategy,
            model=policy_evaluator_model,
            provider=policy_evaluator_provider,
        )
        self.actions: List[Action] = []
        self.state_hashes: List[str] = []  # Track intermediate state hashes
        self.evaluate_policy = evaluate_policy
        self.score_tool_execution_errors = score_tool_execution_errors
        self.score_policy_errors = score_policy_errors

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        if task_index is None:
            task_index = random.randint(0, len(self.tasks))

        # Tools are shared as classes across environment instances. Give each tool a
        # chance to install fresh task-scoped state before the new run starts.
        for tool in set(self.tools_map.values()):
            tool.reset()

        self.task_index = task_index
        self.data = self.data_load_func()
        self.task = self.tasks[task_index]
        self.actions = []

        # Initialize state hash tracking with initial state
        from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
            context_state,
        )

        initial_state_hash = self.get_context_state_hash(context_state.get())
        self.state_hashes = [initial_state_hash]

        initial_observation = self.user.reset(
            persona=self.task.persona,
            instruction=self.task.instruction,
            task_type=self.task.task_type,
            removed_part=self.task.removed_part,
            disambiguation_element_internal=self.task.disambiguation_element_internal,
        )
        return EnvResetResponse(
            observation=initial_observation, info=EnvInfo(task=self.task, source="user")
        )

    def _record_state_hash_if_needed(self):
        """Record state hash after tool actions that might change context state."""
        from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
            context_state,
        )

        current_state_hash = self.get_context_state_hash(context_state.get())
        self.state_hashes.append(current_state_hash)

    def step(self, action: Action, messages: List[Dict]) -> EnvResponse:
        self.actions.append(action)

        info = EnvInfo(task=self.task)
        reward = 0
        done = False
        if action.name in USER_AS_A_TOOL_ACTION_NAMES:
            pass
        elif action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(
                    data=self.data, **action.kwargs
                )
            except Exception as e:
                _record_tool_runtime_error(action.name, e)
                observation = f"Error: {e}"
            info.source = action.name
        else:
            observation = f"Unknown action {action.name}"
            info.source = action.name

        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    async def _run_action(self, action: Action, tools_map, data):
        """Helper function to run a single action asynchronously."""
        if action.name in tools_map:
            try:
                observation = tools_map[action.name].invoke(data=data, **action.kwargs)
                source = action.name
                is_terminate = action.name in self.terminate_tools
            except Exception as e:
                _record_tool_runtime_error(action.name, e)
                observation = f"Error: {e}"
                source = action.name
                is_terminate = False
                if self.task.removed_part:
                    removed_parameters = [
                        removed_part
                        for removed_part in self.task.removed_part
                        if "." in removed_part and removed_part.split(".")[0] == action.name
                    ]
                    if removed_parameters:
                        observation = f"Error: {e}. Note that this tool argument is currently removed so that this tool can not be used. Do not try to call this tool with this argument as it will result in an error."
                # tool_names_removed_part = [removed_part.split('.')[0] for removed_part in self.task.removed_part] if self.task.removed_part else []
                # if action.name in tool_names_removed_part:
                #     observation = f"Error: {e}. Note that this tool argument is currently removed so that this tool can not be used. Do not try to call this tool with this paramter as it will result in an error."
        else:
            observation = f"Unknown action {action.name}"
            source = action.name
            is_terminate = False

        return {
            "observation": observation,
            "source": source,
            "is_terminate": is_terminate,
        }

    async def steps(self, actions: List[Action], messages: List[Dict]) -> EnvResponse:
        """Run multiple actions asynchronously and return a combined response."""
        self.actions.extend(actions)

        info = EnvInfo(task=self.task)
        reward = 0
        done = False
        observations = []

        # Handle USER_AS_A_TOOL_ACTION_NAMES separately if it's the first action
        new_user_message = None
        if actions and actions[0].name in USER_AS_A_TOOL_ACTION_NAMES:
            # Record state hash after tools of one turn are performed
            self._record_state_hash_if_needed()
            new_user_message = self.user.step(actions[0].kwargs["content"])
            observations.append(new_user_message)
            info.source = "user"
            done = "###STOP###" in new_user_message
            # Do not process any other action
            remaining_actions = None
        else:
            remaining_actions = actions

        # Process remaining actions asynchronously
        if remaining_actions:
            # DEBUG
            for action in remaining_actions:
                if not action.name == "planning_tool":
                    print(f"🔧 Running tool: {action}")
            # Create tasks for all remaining actions
            tasks = [
                self._run_action(action, self.tools_map, self.data)
                for action in remaining_actions
            ]

            # Gather results
            results = await asyncio.gather(*tasks)

            for i, result in enumerate(results):
                # DEBUG
                if result["source"] == "planning_tool":
                    try:
                        plan_result = json.loads(result["observation"])
                        from car_bench.envs.car_voice_assistant.tools.cross_domain.planning import (
                            pretty_print_plan,
                        )

                        formatted_plan = pretty_print_plan(
                            plan_result["result"]["plan_details"]
                        )
                        print(f"🔧📝 Plan: {formatted_plan}")
                    except Exception as e:
                        print(f"🔧 Error parsing plan: {e}")
                else:
                    print(f"🔧 Tool outputs: {result['observation']}")
                observations.append(result["observation"])
                if result["is_terminate"]:
                    done = True

            # If there were multiple sources, combine them
            if new_user_message is not None:
                info.source = "user+" + "+".join(result["source"] for result in results)
            else:
                info.source = "+".join(result["source"] for result in results)

        if done:
            if not actions[0].name in USER_AS_A_TOOL_ACTION_NAMES:
                # Extract tool_call_ids from the last assistant message to match with tool results
                # This is required by OpenAI/litellm API - tool responses must have tool_call_id
                tool_call_ids = []
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        tool_call_ids = [tc["id"] for tc in msg["tool_calls"]]
                        break
                
                for i, action in enumerate(remaining_actions):
                    tool_msg = {
                        "role": "tool",
                        "name": action.name,
                        "content": observations[i],
                    }
                    # Add tool_call_id if available (required by OpenAI API spec)
                    if i < len(tool_call_ids):
                        tool_msg["tool_call_id"] = tool_call_ids[i]
                    messages.append(tool_msg)
            reward_res = self.calculate_reward(messages)
            reward = reward_res.reward
            info.reward_info = reward_res
            info.user_cost = self.user.get_total_cost()

        return EnvResponse(
            observation=observations, reward=reward, done=done, info=info
        )

    def run_steps(self, actions: List[Action], messages: List[Dict]) -> EnvResponse:
        """Synchronous wrapper for the asynchronous steps method that works in threads."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        # Set it as the current event loop for this thread
        asyncio.set_event_loop(loop)
        try:
            # Run the async function in this loop
            return loop.run_until_complete(self.steps(actions, messages))
        finally:
            # Clean up
            loop.close()

    def get_data_hash(self) -> str:
        return consistent_hash(to_hashable(self.data))

    def get_context_state_hash(self, context_state) -> str:
        return consistent_hash(to_hashable(context_state))

    def calculate_reward(self, messages: List[Dict]) -> RewardResult:
        reward = 1.0
        performed_actions = self.actions.copy()

        # Get temperature difference for policy evaluation
        from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
            context_state,
        )
        climate_temperature_driver = context_state.get().climate_temperature_driver
        climate_temperature_passenger = (
            context_state.get().climate_temperature_passenger
        )
        difference_is_more_than_3_degrees = (
            abs(climate_temperature_driver - climate_temperature_passenger) > 3
        )

        # Get performed action names for policy evaluation
        performed_action_names = {action.name for action in performed_actions}

        # ==== State-based Reward ====
        (
            r_actions_final,
            r_actions_intermediate,
            r_actions,
            reward_delta,
        ) = calculate_state_based_reward(
            task=self.task,
            state_hashes=self.state_hashes,
            get_context_state_hash=self.get_context_state_hash,
            step_fn=self.step,
            tools_map=self.tools_map,
            messages=messages,
        )
        reward += reward_delta

        # ==== Tool Subset Evaluation Reward ====
        r_tool_subset, tool_subset_missing_tools, reward_delta = (
            calculate_tool_subset_reward(
                task=self.task, performed_actions=performed_actions
            )
        )
        reward += reward_delta

        # ==== Tool-Execution-Error-Calculations ====
        r_tool_execution, tool_execution_errors, reward_delta = (
            calculate_tool_execution_reward(
                score_tool_execution_errors=self.score_tool_execution_errors
            )
        )
        reward += reward_delta

        # ==== End-Conversation-Failure-Calculations ====
        r_user_end_conversation, end_conversation_keyword, reward_delta = (
            calculate_end_conversation_reward()
        )
        reward += reward_delta

        # ==== Output-string-based Reward ====
        r_outputs, outputs = calculate_output_reward()

        # ==== Policy-Error-Calculations ====
        r_policy, policy_llm_errors, policy_errors_aut, reward_delta = (
            calculate_policy_reward(
                task=self.task,
                evaluate_policy=self.evaluate_policy,
                score_policy_errors=self.score_policy_errors,
                policy_evaluator=self.policy_evaluator,
                messages=messages,
                performed_action_names=performed_action_names,
                difference_is_more_than_3_degrees=difference_is_more_than_3_degrees,
            )
        )
        reward += reward_delta

        # Ensure reward doesn't go below 0
        reward = max(0.0, reward)

        info = RewardInfo(
            r_actions=r_actions,
            r_actions_final=r_actions_final,
            r_actions_intermediate=r_actions_intermediate,
            r_tool_subset=r_tool_subset,
            tool_subset_missing_tools=tool_subset_missing_tools,
            r_tool_execution=r_tool_execution,
            tool_execution_errors=tool_execution_errors,
            r_policy=r_policy,
            policy_llm_errors=policy_llm_errors,
            policy_aut_errors=policy_errors_aut,
            r_user_end_conversation=r_user_end_conversation,
            end_conversation_keyword=end_conversation_keyword,
            r_outputs=r_outputs,
            outputs=outputs,
        )
        return RewardResult(reward=reward, info=info, actions=performed_actions)
