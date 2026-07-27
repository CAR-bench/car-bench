"""
Modular reward calculation functions for CAR-bench evaluation.

This module contains separate functions for each reward component,
making the reward calculation logic more readable and maintainable.
"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from car_bench.envs.policy_evaluator import policy_errors_during_runtime
from car_bench.envs.tool_execution_error_evaluator import (
    tool_execution_errors_during_runtime,
)
from car_bench.envs.user.user_end_conversation import end_conversation_failure
from car_bench.types import RESPOND_ACTION_NAME, USER_AS_A_TOOL_ACTION_NAMES, TaskType


def is_hallucination_task(task_type: TaskType) -> bool:
    """Check if the task is a hallucination task type."""
    return task_type in [
        TaskType.HALLUCINATION_MISSING_TOOL,
        TaskType.HALLUCINATION_MISSING_TOOL_PARAMETER,
        TaskType.HALLUCINATION_MISSING_TOOL_RESPONSE,
    ]


def calculate_state_based_reward(
    task,
    state_hashes: List[str],
    get_context_state_hash,
    step_fn,
    tools_map: Dict,
    messages: List[Dict],
) -> Tuple[Optional[bool], Optional[bool], Optional[bool], float]:
    """
    Calculate state-based reward (final and intermediate state correctness).

    Args:
        task: The task being evaluated
        state_hashes: List of state hashes from agent execution
        get_context_state_hash: Function to compute state hash
        step_fn: Function to execute actions
        tools_map: Dictionary of available tools
        messages: Conversation messages

    Returns:
        Tuple of (r_actions_final, r_actions_intermediate, r_actions, reward_delta)
    """
    if is_hallucination_task(task.task_type):
        return None, None, None, 0.0

    from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
        ContextState,
        context_state,
    )

    context_state_hash = get_context_state_hash(context_state.get())

    # Generate ground truth state hashes
    gt_context_state = ContextState()
    token = context_state.set(gt_context_state)
    context_state.get().update_state(**task.context_init_config)

    # Record initial state hash for ground truth
    gt_state_hashes = [get_context_state_hash(context_state.get())]

    for action in task.actions:
        step_fn(action, messages)
        # Record state hash after each ground truth action
        if action.name in tools_map and action.name not in USER_AS_A_TOOL_ACTION_NAMES:
            current_gt_hash = get_context_state_hash(context_state.get())
            gt_state_hashes.append(current_gt_hash)

    gt_vehicle_state_hash = get_context_state_hash(context_state.get())

    # Restore the vehicle state
    context_state.reset(token)

    # Check final state correctness
    r_actions_final = context_state_hash == gt_vehicle_state_hash
    reward_delta = 0.0 if not r_actions_final else 0.0

    # Check intermediate states correctness
    gt_state_hashes_set = set(gt_state_hashes)
    actual_state_hashes_set = set(state_hashes)
    r_actions_intermediate = actual_state_hashes_set.issubset(gt_state_hashes_set)
    if not r_actions_intermediate:
        reward_delta = -1.0

    if not r_actions_final:
        reward_delta = -1.0

    # Legacy field for backward compatibility
    r_actions = r_actions_final and r_actions_intermediate

    return r_actions_final, r_actions_intermediate, r_actions, reward_delta


def calculate_tool_subset_reward(
    task, performed_actions: List
) -> Tuple[Optional[float], Optional[List[str]], float]:
    """
    Calculate tool subset reward (coverage of required information-gathering tools).

    Args:
        task: The task being evaluated
        performed_actions: List of actions performed by the agent

    Returns:
        Tuple of (r_tool_subset, tool_subset_missing_tools, reward_delta)
    """
    if is_hallucination_task(task.task_type):
        return None, None, 0.0

    gt_action_names = {
        action.name for action in task.actions if action.name != RESPOND_ACTION_NAME
    }
    performed_action_names = {action.name for action in performed_actions}

    # Calculate missing tools from ground truth
    tool_subset_missing_tools = list(gt_action_names - performed_action_names)
    r_tool_subset = gt_action_names.issubset(performed_action_names)

    if not r_tool_subset:
        return 0.0, tool_subset_missing_tools, -1.0
    else:
        return 1.0, tool_subset_missing_tools, 0.0


def calculate_tool_execution_reward(
    score_tool_execution_errors: bool,
) -> Tuple[float, List, float]:
    """
    Calculate tool execution error reward.

    Args:
        score_tool_execution_errors: Whether to score tool execution errors

    Returns:
        Tuple of (r_tool_execution, tool_execution_errors, reward_delta)
    """
    tool_execution_errors = tool_execution_errors_during_runtime.get()

    if score_tool_execution_errors and len(tool_execution_errors) > 0:
        return 0.0, tool_execution_errors, -1.0
    else:
        return 1.0, tool_execution_errors, 0.0


def calculate_end_conversation_reward() -> Tuple[
    float, Optional[str], float
]:
    """
    Calculate end conversation reward (user satisfaction).

    Returns:
        Tuple of (r_user_end_conversation, end_conversation_keyword, reward_delta)
    """
    end_conversation_fail = end_conversation_failure.get()

    if len(end_conversation_fail) > 0:
        end_conversation_keyword = end_conversation_fail[0][
            "conversation_control_keyword"
        ]
        return 0.0, end_conversation_keyword, -1.0
    else:
        return 1.0, None, 0.0


def calculate_output_reward() -> Tuple[Optional[float], Dict]:
    """
    Calculate output-string-based reward.

    Returns:
        Tuple of (r_outputs, outputs)
    """
    # Placeholder for future implementation
    return None, {}


def filter_trajectory_messages(messages: List[Dict]) -> List[Dict]:
    """
    Filter trajectory messages to include only specified keys.

    Args:
        messages: List of conversation messages

    Returns:
        Filtered list of messages with core fields only
    """
    traj_messages = messages[1:] if messages[0]["role"] == "system" else messages
    traj_messages_core = []

    for msg in traj_messages:
        filtered_msg = {}
        for key in ["role", "content", "name", "tool_calls"]:
            if key in msg:
                if key == "content" and "tool_calls" in msg and msg["tool_calls"]:
                    filtered_msg[key] = ""
                else:
                    filtered_msg[key] = msg[key]
        traj_messages_core.append(filtered_msg)

    return traj_messages_core


def enhance_policy_line_with_context(
    line: str, performed_action_names: Set[str], difference_is_more_than_3_degrees: bool
) -> Optional[str]:
    """
    Enhance policy line with context-specific information or skip if not applicable.

    Args:
        line: Policy line to enhance
        performed_action_names: Set of action names performed by agent
        difference_is_more_than_3_degrees: Whether temperature difference exceeds 3 degrees

    Returns:
        Enhanced policy line or None if policy should be skipped
    """
    if "REQUIRES_CONFIRMATION" in line:
        if len(
            performed_action_names.intersection(
                {
                    "send_email",
                    "open_close_trunk_door",
                    "set_head_lights_high_beams",
                }
            )
        ) > 0:
            return (
                line
                + " The tools that require confirmation are: 'send_email', 'open_close_trunk_door', and 'set_head_lights_high_beams'."
            )
        return None

    elif (
        "windows are requested by the user to open more than 25% (absolute position) and AC is ON in that moment"
        in line
    ):
        if len(performed_action_names.intersection({"open_close_window"})) > 0:
            return (
                line
                + " Consider that you can only know the AC status if it was turned on **before** the user request to open the windows. Mark not applicable, if th AC was not requested to turn on before, also  mark not applicable if the windows are closed or requested to close. Also this policy is one-way: applies only if user requests to open the windows, turn on AC request are handled in another policy."
            )
        return None

    elif "In certain weather conditions, the vehicle control actions" in line:
        if (
            len(
                performed_action_names.intersection(
                    {"open_close_sunroof", "set_fog_lights"}
                )
            )
            == 0
        ):
            return None

    elif "user sets the temperature to a single seat zone" in line:
        if (
            len(performed_action_names.intersection({"set_climate_temperature"})) == 0
            or not difference_is_more_than_3_degrees
        ):
            return None

    elif "If you get multiple alternative routes from the tool" in line:
        if "get_routes_from_start_to_destination" not in performed_action_names:
            return None
        return (
            line
            + " Apply this policy only to route results that contain multiple "
            "alternatives. Prior explicit user authorization to start "
            "navigation counts as authorization, so do not require redundant "
            "confirmation. Even with prior authorization, the assistant must "
            "still tell the user about the fastest and shortest route (or that "
            "one route is both) and state the number of further alternatives "
            "in a user-facing response. Evaluate every route-tool result "
            "independently; a complete presentation for a later route query "
            "does not repair an omitted presentation for an earlier query."
        )

    elif (
        "route is presented in detail (fastest route, shortest route, or upon user detail request)"
        in line
    ):
        if (
            len(
                performed_action_names.intersection(
                    {"get_routes_from_start_to_destination"}
                )
            )
            > 0
        ):
            return (
                line
                + " For this policy, first reason which routes were presented in detail, then evaluate if these routes include a toll road - if true this would be included in the list of road types else it's not present, then evaluate the policy for the routes identified to be presented in detail."
            )
        return None

    elif (
        "user asks for a multi-stop route and does not specify the route selection"
        in line
    ):
        if (
            len(
                performed_action_names.intersection(
                    {"get_routes_from_start_to_destination"}
                )
            )
            == 0
        ):
            return None

    return line


def calculate_policy_llm_errors(
    policy_evaluator,
    messages: List[Dict],
    performed_action_names: Set[str],
    difference_is_more_than_3_degrees: bool,
) -> List[str]:
    """
    Calculate LLM-based policy errors.

    Args:
        policy_evaluator: Policy evaluator instance
        messages: Conversation messages
        performed_action_names: Set of action names performed
        difference_is_more_than_3_degrees: Whether temperature difference exceeds 3 degrees

    Returns:
        List of policy error reasoning strings
    """
    from car_bench.envs.car_voice_assistant.wiki import WIKI_LLM_POL_LINES

    traj_messages_core = filter_trajectory_messages(messages)
    pol_llm_evaluation_results = []

    for line in WIKI_LLM_POL_LINES:
        enhanced_line = enhance_policy_line_with_context(
            line, performed_action_names, difference_is_more_than_3_degrees
        )

        if enhanced_line is None:
            continue

        evaluation_result = policy_evaluator.evaluate_llm(
            policy=enhanced_line, trajectory=str(traj_messages_core)
        )
        pol_llm_evaluation_results.append(json.loads(evaluation_result))

    policy_llm_errors = [
        eval_result["reasoning"]
        for eval_result in pol_llm_evaluation_results
        if not eval_result["policy_followed"]
    ]

    return policy_llm_errors


def calculate_policy_reward(
    task,
    evaluate_policy: bool,
    score_policy_errors: bool,
    policy_evaluator,
    messages: List[Dict],
    performed_action_names: Set[str],
    difference_is_more_than_3_degrees: bool,
) -> Tuple[Optional[float], List[str], List, float]:
    """
    Calculate policy compliance reward (LLM-based and AUT-based).

    Args:
        task: The task being evaluated
        evaluate_policy: Whether to evaluate policy compliance
        score_policy_errors: Whether to score policy errors
        policy_evaluator: Policy evaluator instance
        messages: Conversation messages
        performed_action_names: Set of action names performed
        difference_is_more_than_3_degrees: Whether temperature difference exceeds 3 degrees

    Returns:
        Tuple of (r_policy, policy_llm_errors, policy_errors_aut, reward_delta)
    """
    if is_hallucination_task(task.task_type):
        return None, None, None, 0.0

    policy_llm_errors = []
    policy_errors_aut = []
    r_policy = None
    reward_delta = 0.0

    # LLM-based Policy Error Calculation
    if evaluate_policy:
        r_policy = 1.0
        policy_llm_errors = calculate_policy_llm_errors(
            policy_evaluator,
            messages,
            performed_action_names,
            difference_is_more_than_3_degrees,
        )
    else:
        r_policy = None

    # AUT-based Policy Error Calculation
    if evaluate_policy:
        traj_messages = (
            messages[1:] if messages[0]["role"] == "system" else messages
        )
        policy_evaluator.evaluate_aut(trajectory=traj_messages)
        policy_errors_aut = policy_errors_during_runtime.get()

    # Calculate reward penalty
    if score_policy_errors and (
        len(policy_errors_aut) > 0 or len(policy_llm_errors) > 0
    ):
        r_policy = 0.0
        reward_delta = -1.0

    return r_policy, policy_llm_errors, policy_errors_aut, reward_delta
