import argparse
import contextvars
import json
import threading
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import comb
from typing import Any, Callable, Dict, List, Optional

# Suppress Pydantic serialization warnings from litellm
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
)

from litellm import provider_list

from car_bench.agents.base import Agent
from car_bench.orchestrator import AgentOrchestrator
from car_bench.envs import get_env
from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
    ContextState,
    context_state,
)
from car_bench.envs.car_voice_assistant.context.fixed_context import (
    FixedContext,
    fixed_context,
)
from car_bench.envs.car_voice_assistant.tasks.task_config import TaskConfig, task_config
from car_bench.envs.policy_evaluator import (
    PolicyEvaluatorStrategy,
    policy_errors_during_runtime,
)
from car_bench.envs.tool_execution_error_evaluator import (
    tool_execution_errors_during_runtime,
)
from car_bench.envs.user.user import UserStrategy
from car_bench.envs.user.user_end_conversation import end_conversation_failure
from car_bench.types import EnvRunResult


def filter_non_standard_fields_from_tools(
    tools_info: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Filter out non-standard fields from tool function definitions.
    Keeps only 'type' and 'description' in properties, adding removed fields to description.
    """
    import copy

    # Standard OpenAI function calling schema fields that should be kept
    STANDARD_PROPERTY_FIELDS = {"type", "description"}
    # Standard schema fields that should be preserved
    PRESERVED_FIELDS = {
        "properties",
        "required",
        "additionalProperties",
        "anyOf",
        "enum",
        "items",
        "prefixItems",
    }

    def filter_object_recursively(obj):
        """Recursively filter objects to remove non-standard fields."""
        if not isinstance(obj, dict):
            return obj

        # Create a copy to avoid modifying the original during iteration
        filtered_obj = {}

        # First, copy all preserved fields and standard property fields
        for key, value in obj.items():
            if key in STANDARD_PROPERTY_FIELDS or key in PRESERVED_FIELDS:
                filtered_obj[key] = value

        # Handle properties recursively if this is an object type
        if filtered_obj.get("type") == "object":
            if "properties" in filtered_obj:
                if isinstance(filtered_obj["properties"], dict):
                    # Recursively filter each property
                    new_properties = {}
                    for prop_name, prop_data in filtered_obj["properties"].items():
                        if isinstance(prop_data, dict):
                            new_properties[prop_name] = filter_object_recursively(
                                prop_data
                            )
                        else:
                            # Non-dict property values are invalid - convert to dict with type
                            print(
                                f"Warning: Property '{prop_name}' has invalid non-dict value: {prop_data}"
                            )
                            new_properties[prop_name] = {
                                "type": "string",
                                "description": f"Invalid property value: {prop_data}",
                            }
                    filtered_obj["properties"] = new_properties
                else:
                    # Properties is not a dict - this is invalid, fix it
                    print(
                        f"Warning: 'properties' field is not a dict: {filtered_obj['properties']}"
                    )
                    filtered_obj["properties"] = {}
            elif "anyOf" not in filtered_obj:
                # Add empty properties to make it valid
                filtered_obj["properties"] = {}

        # Handle array types that need items or prefixItems
        if filtered_obj.get("type") == "array":
            if "items" not in filtered_obj and "prefixItems" not in filtered_obj:
                # Add default items to make it valid
                print(
                    f"Warning: Array type missing 'items' or 'prefixItems', adding default"
                )
                filtered_obj["items"] = {"type": "string", "description": "Array item"}

        # Collect non-standard fields that were removed
        non_standard_fields = {}
        for field_name, field_value in obj.items():
            if (
                field_name not in STANDARD_PROPERTY_FIELDS
                and field_name not in PRESERVED_FIELDS
            ):
                non_standard_fields[field_name] = field_value

        # Add non-standard fields to description if any exist
        if non_standard_fields:
            current_description = filtered_obj.get("description", "")
            non_standard_str = ", ".join(
                [f"{k}: {v}" for k, v in non_standard_fields.items()]
            )
            if current_description:
                filtered_obj["description"] = (
                    f"{current_description} ({non_standard_str})"
                )
            else:
                filtered_obj["description"] = f"({non_standard_str})"

        return filtered_obj

    filtered_tools = copy.deepcopy(tools_info)

    for tool in filtered_tools:
        if "function" in tool and "parameters" in tool["function"]:
            tool["function"]["parameters"] = filter_object_recursively(
                tool["function"]["parameters"]
            )

    return filtered_tools


def _init_context_state(isolated_env, idx):
    # load task config
    task = isolated_env.tasks[idx]
    default_task_config = TaskConfig()
    token_task_config = task_config.set(default_task_config)
    task_cfg = task_config.get()
    task_cfg.update_state(calendar_id=task.calendar_id)
    # load fixed and dynamic context and apply context init config
    default_context_state = ContextState()
    default_fixed_car_context = FixedContext()
    token_context_config = context_state.set(default_context_state)
    token_fixed_car_context = fixed_context.set(default_fixed_car_context)
    vehicle_ctx = context_state.get()
    fixed_ctx = fixed_context.get()
    fixed_ctx.update_state(**task.context_init_config)
    vehicle_ctx.update_state(**task.context_init_config)
    if (
        "{{placeholder_location_based_on_task_context_init_config}}"
        in isolated_env.wiki
    ):
        isolated_env.wiki = isolated_env.wiki.replace(
            "{{placeholder_location_based_on_task_context_init_config}}",
            fixed_ctx.current_location.model_dump_json(),
        )
    if (
        "{{placeholder_datetime_based_on_task_context_init_config}}"
        in isolated_env.wiki
    ):
        isolated_env.wiki = isolated_env.wiki.replace(
            "{{placeholder_datetime_based_on_task_context_init_config}}",
            fixed_ctx.current_datetime.model_dump_json(),
        )
    return token_context_config, token_fixed_car_context, token_task_config


def _reset_context_state(
    token_context_config, token_fixed_car_context, token_task_config
):
    context_state.reset(token_context_config)
    fixed_context.reset(token_fixed_car_context)
    task_config.reset(token_task_config)


def get_task_indices_to_run(
    env,
    args: argparse.Namespace,
    ckpt_path: str,
) -> List[int]:
    """
    Determine which task indices to run based on args.
    
    Args:
        env: The environment with tasks
        args: Command-line arguments containing task_id_filter, num_tasks, task_type, task_split
        ckpt_path: Checkpoint path for logging
        
    Returns:
        List of task indices to run
    """
    if args.task_id_filter and len(args.task_id_filter) > 0:
        # Filter by specific task IDs
        task_id_set = set(args.task_id_filter)
        available_task_ids = {task.task_id for task in env.tasks}
        
        # Check for task IDs not in the current split
        missing_task_ids = task_id_set - available_task_ids
        if missing_task_ids:
            print(f"⚠️  WARNING: The following task IDs are not in the '{args.task_type}_{args.task_split}' split: {sorted(missing_task_ids)}")
        
        # Get indices of tasks that are actually available
        found_task_ids = task_id_set & available_task_ids
        idxs_to_run = [i for i, task in enumerate(env.tasks) if task.task_id in found_task_ids]
        
        if len(idxs_to_run) == 0:
            raise ValueError(f"No tasks found matching task IDs: {args.task_id_filter}")
        
        actual_task_ids = [env.tasks[i].task_id for i in idxs_to_run]
        print(f"Running {len(idxs_to_run)} tasks with IDs: {actual_task_ids} (checkpoint path: {ckpt_path})")
    else:
        # Use first num_tasks from the filtered split
        num_tasks = len(env.tasks) if args.num_tasks == -1 else min(args.num_tasks, len(env.tasks))
        idxs_to_run = list(range(num_tasks))
        print(f"Running {num_tasks} tasks from {args.task_type}_{args.task_split} split (checkpoint path: {ckpt_path})")
    
    return idxs_to_run


def run(
    args: argparse.Namespace,
    ckpt_path: str,
    custom_agent_factory: Optional[Callable[[List[Dict[str, Any]], Any, argparse.Namespace], Agent]] = None,
) -> List[EnvRunResult]:
    # Pre-load mock data so all tasks have equal runtime
    from car_bench.envs.car_voice_assistant.mock_data import car_va_data_manager
    car_va_data_manager.initialize()

    print(f"Loading user with strategy: {args.user_strategy}")
    env = get_env(
        args.env,
        user_strategy=args.user_strategy,
        user_model=args.user_model,
        user_provider=args.user_model_provider,
        policy_evaluator_strategy=args.policy_evaluator_strategy,
        policy_evaluator_model=args.policy_evaluator_model,
        policy_evaluator_provider=args.policy_evaluator_model_provider,
        task_type=args.task_type,
        task_split=args.task_split,
        use_user_as_a_tool_tools=args.use_user_as_a_tool_tools,
        user_thinking=args.user_thinking,
    )
    
    idxs_to_run = get_task_indices_to_run(env, args, ckpt_path)
    
    results: List[EnvRunResult] = []
    lock = threading.Lock()
    
    for i in range(args.num_trials):
        idxs = idxs_to_run.copy()
        if args.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            isolated_env = get_env(
                args.env,
                user_strategy=args.user_strategy,
                user_model=args.user_model,
                policy_evaluator_strategy=args.policy_evaluator_strategy,
                policy_evaluator_model=args.policy_evaluator_model,
                task_type=args.task_type,
                task_split=args.task_split,
                user_provider=args.user_model_provider,
                policy_evaluator_provider=args.policy_evaluator_model_provider,
                user_thinking=args.user_thinking,
                task_index=idx,
                evaluate_policy=args.evaluate_policy,
                score_tool_execution_errors=args.score_tool_execution_errors,
                score_policy_errors=args.score_policy_errors,
            )

            # Create an isolated agent per task to avoid cross-thread mutation
            local_agent = agent_factory(
                tools_info=isolated_env.tools_info,
                wiki=isolated_env.wiki,
                args=args,
                custom_agent_factory=custom_agent_factory,
            )

            print(f"Running task {idx}")
            token_context_config, token_fixed_car_context, token_task_config = (
                _init_context_state(isolated_env, idx)
            )

            token_policy_errors = policy_errors_during_runtime.set([])
            token_tool_execution_errors = tool_execution_errors_during_runtime.set([])
            token_end_conversation_failure = end_conversation_failure.set([])
            try:
                # Select orchestrator based on agent strategy
                if args.agent_strategy == "terminal":
                    from car_bench.terminal.orchestrator import TerminalOrchestrator

                    orchestrator = TerminalOrchestrator(
                        local_agent,
                        remove_planning_tools=not args.planning_and_thinking_tool,
                    )
                else:
                    orchestrator = AgentOrchestrator(
                        local_agent,
                        remove_planning_tools=not args.planning_and_thinking_tool,
                    )
                res = orchestrator.execute(
                    env=isolated_env,
                    task_index=idx,
                )
                result = EnvRunResult(
                    task_index=idx,
                    task_id=isolated_env.tasks[idx].task_id,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_index=idx,
                    task_id=isolated_env.tasks[idx].task_id,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            finally:
                _reset_context_state(
                    token_context_config, token_fixed_car_context, token_task_config
                )
                policy_errors_during_runtime.reset(token_policy_errors)
                tool_execution_errors_during_runtime.reset(token_tool_execution_errors)
                end_conversation_failure.reset(token_end_conversation_failure)
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={result.task_id} (index={idx})",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                # Ensure the directory exists before writing
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], 
    wiki, 
    args: argparse.Namespace,
    custom_agent_factory: Optional[Callable[[List[Dict[str, Any]], Any, argparse.Namespace], Agent]] = None,
) -> Agent:
    """Create an agent instance based on args or custom factory.
    
    Args:
        tools_info: List of tool definitions
        wiki: Wiki/policy information
        args: Command-line arguments
        custom_agent_factory: Optional custom factory function that takes (tools_info, wiki, args) and returns an Agent.
                            For custom agents, write a Python script that calls run() with this parameter.
        
    Returns:
        Agent instance
    """
    # If custom factory provided, use it
    if custom_agent_factory is not None:
        return custom_agent_factory(tools_info, wiki, args)
    
    if args.agent_strategy == "tool-calling":
        # native tool calling
        from car_bench.agents.tool_calling_agent import ToolCallingAgent

        if args.remove_non_standard_fields_from_tools:
            tools_info = filter_non_standard_fields_from_tools(tools_info)
        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=args.model,
            provider=args.model_provider,
            temperature=args.temperature,
            thinking=args.thinking,
            interleaved_thinking=args.interleaved_thinking,
            reasoning_effort=args.reasoning_effort,
        )
    elif args.agent_strategy == "terminal":
        from car_bench.agents.terminal_agent import TerminalAgent

        if not args.terminal_agent_command:
            raise ValueError(
                "--terminal-agent-command is required when using --agent-strategy=terminal"
            )
        return TerminalAgent(
            agent_command=args.terminal_agent_command,
            agent_args=args.terminal_agent_args or [],
            tools_info=tools_info,
            wiki=wiki,
            timeout=args.terminal_agent_timeout,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {args.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_index: dict[int, int] = {}
    for result in results:
        if result.task_index not in c_per_task_index:
            c_per_task_index[result.task_index] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_index[result.task_index] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_index.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_index)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env",
        type=str,
        choices=["car_voice_assistant"],
        default="car_voice_assistant",
    )
    parser.add_argument("--model", type=str, help="The model to use for the agent")
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gemini-2.5-flash",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        default="gemini",
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--policy-evaluator-model",
        type=str,
        default="gemini-2.5-flash",
        help="The model to use for the policy evaluator",
    )
    parser.add_argument(
        "--policy-evaluator-model-provider",
        type=str,
        choices=provider_list,
        default="gemini",
        help="The model provider for the policy evaluator",
    )
    parser.add_argument(
        "--agent-strategy", type=str, default="tool-calling", choices=["tool-calling", "terminal"]
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="base",
        choices=["base", "hallucination", "disambiguation"],
        help="The type of tasks to run",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="The split of tasks to run (train or test)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=-1,
        help="Number of tasks to run from the filtered split. Use -1 to run all tasks (default)",
    )
    parser.add_argument(
        "--task-id-filter",
        type=str,
        nargs="+",
        help="(Optional) run only specific task IDs (e.g., base_0 base_2). Takes precedence over --num-tasks",
    )
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument(
        "--user-strategy",
        type=str,
        default="llm",
        choices=[item.value for item in UserStrategy],
    )
    parser.add_argument(
        "--policy-evaluator-strategy",
        type=str,
        default="llm",
        choices=[item.value for item in PolicyEvaluatorStrategy],
    )
    parser.add_argument(
        "--few-shot-displays-path",
        type=str,
        help="Path to a jsonlines file containing few shot displays",
    )
    parser.add_argument(
        "--evaluate-policy", type=bool, default=True, help="Whether to evaluate the policy"
    )
    parser.add_argument(
        "--score-tool-execution-errors",
        type=bool,
        default=True,
        help="Whether to score tool execution errors",
    )
    parser.add_argument(
        "--score-policy-errors",
        type=bool,
        default=True,
        help="Whether to score policy errors",
    )
    parser.add_argument(
        "--use-user-as-a-tool-tools",
        type=bool,
        default=False,
        help="Whether to use the user as a tool",
    )
    parser.add_argument(
        "--thinking", action="store_true", help="Whether to use thinking"
    )
    parser.add_argument(
        "--user-thinking",
        type=bool,
        default=True,
        help="Whether to use thinking for the user simulator",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="none",
        help="The reasoning effort to use for thinking",
    )
    parser.add_argument(
        "--interleaved-thinking",
        action="store_true",
        help="Whether to use interleaved thinking",
    )
    parser.add_argument(
        "--remove-non-standard-fields-from-tools",
        type=bool,
        default=False,
        help="Whether to remove non-standard fields from tools",
    )
    parser.add_argument(
        "--planning-and-thinking-tool",
        type=bool,
        default=True,
        help="Whether to use the planning and thinking tool",
    )
    # Terminal agent arguments
    parser.add_argument(
        "--terminal-agent-command",
        type=str,
        default=None,
        help="Command to launch the terminal agent (e.g., 'claude', 'python my_agent.py'). Required when --agent-strategy=terminal",
    )
    parser.add_argument(
        "--terminal-agent-args",
        type=str,
        nargs="*",
        default=[],
        help="Additional arguments for the terminal agent command",
    )
    parser.add_argument(
        "--terminal-agent-timeout",
        type=int,
        default=300,
        help="Maximum seconds per task for terminal agent (default: 300)",
    )
    args = parser.parse_args()

    print(args)
    random.seed(args.seed)

    time_str = datetime.now().strftime("%m%d%H%M%S")

    # Build filename components
    thinking_suffix = "-thinking" if args.thinking else ""
    interleaved_thinking_suffix = (
        "-interleaved-thinking" if args.interleaved_thinking else ""
    )
    reasoning_effort_suffix = (
        f"-reasoning-effort-{args.reasoning_effort}" if args.thinking else ""
    )
    user_thinking_suffix = "-user-thinking" if args.user_thinking else ""

    # Determine task count for filename
    task_count = len(args.task_id_filter) if args.task_id_filter else (args.num_tasks if args.num_tasks != -1 else "all")
    file_str = f"{args.log_dir}/{args.task_type}_{args.task_split}/{args.model.split('/')[-1]}{thinking_suffix}{interleaved_thinking_suffix}{reasoning_effort_suffix}-{args.temperature}_tasks_{task_count}_user-{args.user_model}{user_thinking_suffix}-{args.user_strategy}_{time_str}.json"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    results = run(
        args=args,
        ckpt_path=file_str,
    )

    display_metrics(results)

    with open(file_str, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {file_str}\n")


if __name__ == "__main__":
    main()
