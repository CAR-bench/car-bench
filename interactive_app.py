"""
Interactive Task UI for CAR-bench.

A web-based interface where a human can play through benchmark tasks,
taking the role of either the user (driver) or the agent (in-car assistant).
Uses the real orchestrator loop so evaluation works correctly.

Usage:
    pip install flask python-dotenv
    python interactive_app.py [--port 5000]
"""

import argparse
import contextvars
import json
import os
import re
import threading
import time
import traceback
import uuid
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, render_template, request

from car_bench.agents.base import Agent
from car_bench.agents.tool_calling_agent import ToolCallingAgent
from car_bench.envs import get_env
from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
    ContextState,
    context_state,
)
from car_bench.envs.car_voice_assistant.context.fixed_context import (
    FixedContext,
    fixed_context,
)
from car_bench.envs.car_voice_assistant.mock_data import car_va_data_manager
from car_bench.envs.car_voice_assistant.tasks.task_config import TaskConfig, task_config
from car_bench.envs.car_voice_assistant.wiki import WIKI_RAW
from car_bench.envs.policy_evaluator import policy_errors_during_runtime
from car_bench.envs.tool_execution_error_evaluator import (
    tool_execution_errors_during_runtime,
)
from car_bench.envs.tool_manipulation import remove_tool_elements
from car_bench.envs.user.user import BaseUserSimulationEnv
from car_bench.envs.user.user_end_conversation import end_conversation_failure
from car_bench.orchestrator import AgentOrchestrator, message_to_actions
from car_bench.types import (
    RESPOND_ACTION_NAME,
    USER_AS_A_TOOL_ACTION_NAMES,
    Action,
    AgentState,
    SolveResult,
    Task,
    TaskType,
)

app = Flask(__name__, template_folder="templates")

# ---------------------------------------------------------------------------
# Task loading with HuggingFace fallback to local result files
# ---------------------------------------------------------------------------

_RESULT_DIR_MAP = {
    "base": ["base_v2", "base"],
    "hallucination": ["hallucination_train_v2", "hallucination"],
    "disambiguation": ["disambiguation_v2", "disambiguation"],
}


def _natural_sort_key(s: str):
    """Sort key that handles embedded numbers naturally (base_2 before base_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def _load_tasks_from_hf(task_type: str, task_split: str) -> List[Task]:
    """Try loading tasks from HuggingFace dataset."""
    from car_bench.envs.car_voice_assistant.env import _load_tasks

    return _load_tasks(task_type, task_split)


def _load_tasks_from_results(task_type: str) -> List[Task]:
    """Fallback: extract unique tasks from local result files."""
    results_base = os.path.join(os.path.dirname(__file__), "results")
    tasks_by_id: Dict[str, Task] = {}

    dirs = _RESULT_DIR_MAP.get(task_type, [task_type])
    for d in dirs:
        dirpath = os.path.join(results_base, d)
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                for entry in data:
                    info = entry.get("info", {})
                    task_dict = info.get("task")
                    if not task_dict or not isinstance(task_dict, dict):
                        continue
                    if "task_id" not in task_dict:
                        top_id = entry.get("task_id")
                        if top_id is not None:
                            task_dict["task_id"] = f"{task_type}_{top_id}"
                        else:
                            continue
                    tid = task_dict["task_id"]
                    if tid in tasks_by_id:
                        continue
                    if "actions" in task_dict and task_dict["actions"]:
                        task_dict["actions"] = [
                            Action(**a) if isinstance(a, dict) else a
                            for a in task_dict["actions"]
                        ]
                    tasks_by_id[tid] = Task(**task_dict)
            except Exception:
                continue
        if tasks_by_id:
            break
    return sorted(tasks_by_id.values(), key=lambda t: _natural_sort_key(t.task_id))


def load_tasks(task_type: str, task_split: str) -> List[Task]:
    """Load tasks, falling back to local result files if HuggingFace is unavailable."""
    try:
        tasks = _load_tasks_from_hf(task_type, task_split)
        if tasks:
            return tasks
    except Exception as e:
        print(f"HuggingFace loading failed ({e}), falling back to local results...")
    tasks = _load_tasks_from_results(task_type)
    if tasks:
        print(f"Loaded {len(tasks)} {task_type} tasks from local result files")
    return tasks


# ---------------------------------------------------------------------------
# HumanWebAgent: Agent implementation that waits for UI input
# ---------------------------------------------------------------------------


class HumanWebAgent(Agent):
    """Agent that blocks on a queue, waiting for human input from the web UI."""

    def __init__(self, input_queue: Queue, output_queue: Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._call_counter = 0
        # Snapshot of the state/tools currently being awaited, so an LLM
        # sidecar can play a turn on the human's behalf (see /api/llm-turn).
        self._pending_state: Optional[AgentState] = None
        self._pending_tools_info: Optional[List[Dict[str, Any]]] = None

    def get_init_state(self, system_prompt: str, initial_observation: str) -> AgentState:
        return AgentState(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_observation},
            ]
        )

    def generate_next_message(
        self, state: AgentState, tools_info: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], AgentState]:
        # Signal the UI that we're waiting for agent input
        self.output_queue.put({"type": "waiting_for_input", "role": "agent"})
        self._pending_state = state
        self._pending_tools_info = tools_info

        # Block until human (or the LLM sidecar, via /api/llm-turn) provides input
        human_input = self.input_queue.get()
        self._pending_state = None
        self._pending_tools_info = None

        if human_input.get("type") == "stop":
            # Human ended session - send a respond action
            msg = {"role": "assistant", "content": "Session ended by user."}
            return msg, AgentState(messages=state.messages + [msg])

        if human_input.get("type") == "tool_calls":
            # Human executed tool(s)
            tool_calls = []
            for tc in human_input["tool_calls"]:
                self._call_counter += 1
                tool_calls.append(
                    {
                        "id": f"call_human_{self._call_counter}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                )
            msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        else:
            # Human sent a text response
            msg = {"role": "assistant", "content": human_input.get("content", "")}

        return msg, AgentState(
            messages=state.messages + [msg],
            total_cost=state.total_cost,
            total_llm_induced_latency_ms=state.total_llm_induced_latency_ms,
            turn_counter=state.turn_counter,
            least_prompt_tokens=state.least_prompt_tokens,
            latest_prompt_tokens=state.latest_prompt_tokens,
        )


# ---------------------------------------------------------------------------
# WebUIUserSimulationEnv: User that waits for UI input
# ---------------------------------------------------------------------------


class WebUIUserSimulationEnv(BaseUserSimulationEnv):
    """User simulation that blocks on a queue, waiting for human input from the web UI."""

    def __init__(self, input_queue: Queue, output_queue: Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

    def reset(
        self,
        persona: Optional[str] = None,
        instruction: Optional[str] = None,
        task_type: Optional[TaskType] = TaskType.BASE,
        removed_part: Optional[str] = None,
        disambiguation_element_internal: Optional[str] = None,
    ) -> str:
        # Signal UI that we need the first user message
        self.output_queue.put({
            "type": "waiting_for_input",
            "role": "user",
            "context": {"persona": persona, "instruction": instruction},
        })
        human_input = self.input_queue.get()
        content = human_input.get("content", "")
        self.output_queue.put({"type": "user_message", "content": content})
        return content

    def step(self, content: str) -> str:
        # content is the agent's message to the user - show it first
        # (already shown by the orchestrator events)
        # Signal UI that we need the next user message
        self.output_queue.put({"type": "waiting_for_input", "role": "user"})
        human_input = self.input_queue.get()
        if human_input.get("type") == "stop":
            return "###STOP###"
        user_content = human_input.get("content", "")
        self.output_queue.put({"type": "user_message", "content": user_content})
        return user_content

    def get_total_cost(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# InstrumentedOrchestrator: emits events during execution
# ---------------------------------------------------------------------------


class InstrumentedOrchestrator(AgentOrchestrator):
    """Orchestrator that emits events to a queue for the UI to consume."""

    def __init__(self, agent: Agent, output_queue: Queue, remove_planning_tools: bool = True):
        super().__init__(agent, remove_planning_tools)
        self.output_queue = output_queue

    def execute(
        self, env, task_index: Optional[int] = None, max_num_steps: int = 40
    ) -> SolveResult:
        from car_bench.envs.tool_manipulation import (
            check_hallucinated_removed_part,
            remove_result_element,
            remove_tool_elements,
        )

        # Initialize environment
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        # Emit the first user message only for human-as-agent mode (LLM user).
        # In human-as-user mode, WebUIUserSimulationEnv already emits user_message.
        if isinstance(self.agent, HumanWebAgent):
            self.output_queue.put({"type": "user_message", "content": obs})

        # Prepare tools
        tools_info = env.tools_info
        if self.remove_planning_tools:
            tools_info = remove_tool_elements(
                tools_info, env.tools_info, ["planning_tool", "think"]
            )
        if env.task.removed_part:
            tools_info = remove_tool_elements(
                tools_info, env.tools_info, env.task.removed_part
            )

        # Store tools info for UI access
        self._tools_info = tools_info

        # Initialize agent state
        system_prompt = env.wiki if env.wiki is not None else ""
        state = self.agent.get_init_state(system_prompt, obs)

        # Main loop
        for step in range(max_num_steps):
            next_message, state = self.agent.generate_next_message(state, tools_info)

            # Check hallucination
            if env.task.removed_part:
                if next_message.get("tool_calls"):
                    hallucinated = check_hallucinated_removed_part(
                        env.task.removed_part,
                        next_message["tool_calls"],
                        env.task.task_type,
                    )
                    if hallucinated:
                        self.output_queue.put({
                            "type": "warning",
                            "content": f"Hallucination detected: used removed part {env.task.removed_part}",
                        })

            actions = message_to_actions(next_message)

            is_user_action = actions[0].name in USER_AS_A_TOOL_ACTION_NAMES

            # Emit events BEFORE execution
            if is_user_action:
                # User interaction turn - emit agent_message before run_steps
                # because run_steps will block waiting for the human's reply
                if (
                    actions[0].name != RESPOND_ACTION_NAME
                    and actions[0].name in USER_AS_A_TOOL_ACTION_NAMES
                ):
                    agent_content = json.loads(
                        next_message["tool_calls"][0]["function"]["arguments"]
                    )["message_to_user"]
                else:
                    agent_content = next_message.get("content", "")

                self.output_queue.put({
                    "type": "agent_message",
                    "content": agent_content,
                })
            else:
                # Tool calls - emit before execution
                for a in actions:
                    self.output_queue.put({
                        "type": "tool_call",
                        "name": a.name,
                        "args": a.kwargs,
                    })

            # Execute (may block for user interaction turns)
            env_response = env.run_steps(actions, state.messages)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # Handle responses
            if not is_user_action and not env_response.done:
                for idx, tool_call in enumerate(next_message["tool_calls"]):
                    observation = env_response.observation[idx]

                    # Apply result removal for hallucination tasks
                    if env.task.removed_part and any(
                        rp.split(".")[0] == "result" for rp in env.task.removed_part
                    ):
                        for rp in env.task.removed_part:
                            if (
                                rp.split(".")[0] == "result"
                                and tool_call["function"]["name"] == rp.split(".")[1]
                            ):
                                observation = json.dumps(
                                    remove_result_element(
                                        json.loads(observation), env.task.removed_part
                                    )
                                )

                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": observation,
                    })

                    self.output_queue.put({
                        "type": "tool_result",
                        "name": tool_call["function"]["name"],
                        "result": observation,
                    })
            elif is_user_action:
                # User interaction turn - process the response
                state = AgentState(
                    messages=state.messages,
                    total_cost=state.total_cost,
                    total_llm_induced_latency_ms=state.total_llm_induced_latency_ms,
                    turn_counter=state.turn_counter + 1,
                    least_prompt_tokens=state.least_prompt_tokens,
                    latest_prompt_tokens=state.latest_prompt_tokens,
                )

                user_obs = env_response.observation[0]
                state.messages.append({"role": "user", "content": user_obs})

                # For LLM user (human-as-agent mode), emit the user message
                # WebUIUserSimulationEnv emits it itself; LLM user doesn't
                if isinstance(self.agent, HumanWebAgent):
                    self.output_queue.put({
                        "type": "user_message",
                        "content": user_obs,
                    })

            # Emit state update
            try:
                self.output_queue.put({
                    "type": "state_update",
                    "vehicle_state": context_state.get().model_dump(),
                })
            except Exception:
                pass

            if env_response.done:
                break

        # Build final info
        info = {
            **info,
            "total_agent_cost": state.total_cost,
            "total_llm_induced_latency_ms": state.total_llm_induced_latency_ms,
        }
        if env.task.removed_part:
            info["removed_part"] = env.task.removed_part

        return SolveResult(
            reward=reward, info=info, messages=state.messages, total_cost=state.total_cost
        )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_session: Dict[str, Any] = {}


def _init_context_state(env, idx):
    """Initialize context vars for a task. Adapted from run.py."""
    task = env.tasks[idx]
    default_task_config = TaskConfig()
    token_task_config = task_config.set(default_task_config)
    task_cfg = task_config.get()
    task_cfg.update_state(calendar_id=task.calendar_id)

    default_context_state = ContextState()
    default_fixed_context = FixedContext()
    token_context = context_state.set(default_context_state)
    token_fixed = fixed_context.set(default_fixed_context)

    vehicle_ctx = context_state.get()
    fixed_ctx = fixed_context.get()
    fixed_ctx.update_state(**task.context_init_config)
    vehicle_ctx.update_state(**task.context_init_config)

    # Fill wiki placeholders
    if "{{placeholder_location_based_on_task_context_init_config}}" in env.wiki:
        env.wiki = env.wiki.replace(
            "{{placeholder_location_based_on_task_context_init_config}}",
            fixed_ctx.current_location.model_dump_json(),
        )
    if "{{placeholder_datetime_based_on_task_context_init_config}}" in env.wiki:
        env.wiki = env.wiki.replace(
            "{{placeholder_datetime_based_on_task_context_init_config}}",
            fixed_ctx.current_datetime.model_dump_json(),
        )

    token_policy = policy_errors_during_runtime.set([])
    token_tool_exec = tool_execution_errors_during_runtime.set([])
    token_end_conv = end_conversation_failure.set([])

    return {
        "token_context": token_context,
        "token_fixed": token_fixed,
        "token_task_config": token_task_config,
        "token_policy": token_policy,
        "token_tool_exec": token_tool_exec,
        "token_end_conv": token_end_conv,
    }


def _reset_context_state(tokens):
    context_state.reset(tokens["token_context"])
    fixed_context.reset(tokens["token_fixed"])
    task_config.reset(tokens["token_task_config"])
    policy_errors_during_runtime.reset(tokens["token_policy"])
    tool_execution_errors_during_runtime.reset(tokens["token_tool_exec"])
    end_conversation_failure.reset(tokens["token_end_conv"])


def _run_orchestrator_inner(
    env, agent, task_index, output_queue, evaluate_policy, tokens
):
    """Inner function that runs inside the copied context."""
    try:
        orchestrator = InstrumentedOrchestrator(
            agent, output_queue, remove_planning_tools=True
        )
        result = orchestrator.execute(env=env, task_index=task_index)

        # Store tools info from orchestrator for UI access
        _session["tools_info"] = getattr(orchestrator, "_tools_info", [])

        output_queue.put({
            "type": "done",
            "reward": result.reward,
            "info": result.info,
            "messages": result.messages,
        })
    except Exception as e:
        traceback.print_exc()
        output_queue.put({
            "type": "error",
            "content": str(e),
            "traceback": traceback.format_exc(),
        })
    finally:
        _session["done"] = True


def _run_orchestrator_thread(
    ctx, env, agent, task_index, output_queue, evaluate_policy, tokens
):
    """Run the orchestrator in a background thread with copied context vars."""
    ctx.run(
        _run_orchestrator_inner,
        env, agent, task_index, output_queue, evaluate_policy, tokens,
    )


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("interactive.html")


@app.route("/api/task-types")
def api_task_types():
    return jsonify({
        "task_types": ["base", "hallucination", "disambiguation"],
        "task_splits": ["train", "test"],
    })


@app.route("/api/tasks")
def api_tasks():
    task_type = request.args.get("type", "base")
    task_split = request.args.get("split", "train")
    try:
        tasks = load_tasks(task_type, task_split)
        task_list = []
        for i, t in enumerate(tasks):
            task_list.append({
                "index": i,
                "task_id": t.task_id,
                "task_type": t.task_type.value,
                "instruction": t.instruction,
                "persona": t.persona,
                "removed_part": t.removed_part,
                "disambiguation_element_internal": t.disambiguation_element_internal,
                "disambiguation_element_user": t.disambiguation_element_user,
            })
        return jsonify({"tasks": task_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start an interactive session using the real orchestrator."""
    data = request.json
    task_type = data.get("task_type", "base")
    task_split = data.get("task_split", "train")
    task_index = data.get("task_index", 0)
    role = data.get("role", "user")  # "user" or "agent"
    evaluate_policy = data.get("evaluate_policy", True)
    agent_model = data.get("agent_model", "claude-sonnet-4-5-20250929")
    agent_provider = data.get("agent_provider", "anthropic")
    user_model = data.get("user_model", "gemini-2.5-flash")
    user_provider = data.get("user_provider", "gemini")

    try:
        # Pre-load mock data
        try:
            car_va_data_manager.initialize()
        except Exception as e:
            print(f"Mock data init failed ({e}).")

        input_queue = Queue()
        output_queue = Queue()

        # Create environment with appropriate user strategy
        if role == "user":
            # Human is user → use WebUIUserSimulationEnv
            web_user = WebUIUserSimulationEnv(input_queue, output_queue)
            env = get_env(
                "car_voice_assistant",
                user_strategy="human",  # placeholder, we'll replace
                user_model="unused",
                task_type=task_type,
                task_split=task_split,
                task_index=task_index,
                evaluate_policy=evaluate_policy,
                score_tool_execution_errors=True,
                score_policy_errors=evaluate_policy,
                policy_evaluator_strategy="llm",
                policy_evaluator_model="gemini-2.5-flash",
                policy_evaluator_provider="gemini",
            )
            # Replace the user with our web UI user
            env.user = web_user

            # Create LLM agent
            agent = ToolCallingAgent(
                tools_info=env.tools_info,
                wiki=env.wiki,
                model=agent_model,
                provider=agent_provider,
                temperature=0.0,
            )
        else:
            # Human is agent → use LLM user, HumanWebAgent
            env = get_env(
                "car_voice_assistant",
                user_strategy="llm",
                user_model=user_model,
                user_provider=user_provider,
                task_type=task_type,
                task_split=task_split,
                task_index=task_index,
                evaluate_policy=evaluate_policy,
                score_tool_execution_errors=True,
                score_policy_errors=evaluate_policy,
                policy_evaluator_strategy="llm",
                policy_evaluator_model="gemini-2.5-flash",
                policy_evaluator_provider="gemini",
            )
            agent = HumanWebAgent(input_queue, output_queue)

        task = env.tasks[task_index] if task_index < len(env.tasks) else env.tasks[0]

        # Init context state
        tokens = _init_context_state(env, task_index)

        # Get initial state for UI
        vehicle_state = context_state.get().model_dump()
        fixed_state = fixed_context.get().model_dump()

        # Build ground-truth actions
        gt_actions = [{"name": a.name, "kwargs": a.kwargs} for a in task.actions]

        # Filter tools: remove planning tools + task-specific removed parts
        # This matches what the orchestrator does in execute()
        filtered_tools = remove_tool_elements(
            env.tools_info, env.tools_info, ["planning_tool", "think"]
        )
        if task.removed_part:
            filtered_tools = remove_tool_elements(
                filtered_tools, env.tools_info, task.removed_part
            )

        # Store session
        _session.clear()
        _session["task"] = task
        _session["env"] = env
        _session["agent"] = agent
        _session["input_queue"] = input_queue
        _session["output_queue"] = output_queue
        _session["role"] = role
        _session["tokens"] = tokens
        _session["done"] = False
        _session["wiki"] = env.wiki
        _session["tools_info"] = filtered_tools
        # Kept so role=="agent" sessions can spin up an LLM sidecar on demand
        # (see /api/llm-turn), even though the LLM agent isn't used directly.
        _session["agent_model"] = agent_model
        _session["agent_provider"] = agent_provider

        # Copy current context so context vars are available in the thread
        ctx = contextvars.copy_context()

        # Start orchestrator thread
        thread = threading.Thread(
            target=_run_orchestrator_thread,
            args=(ctx, env, agent, task_index, output_queue, evaluate_policy, tokens),
            daemon=True,
        )
        _session["thread"] = thread
        thread.start()

        return jsonify({
            "status": "ok",
            "role": role,
            "task": {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "instruction": task.instruction,
                "persona": task.persona,
                "removed_part": task.removed_part,
                "disambiguation_element_internal": task.disambiguation_element_internal,
                "disambiguation_element_user": task.disambiguation_element_user,
                "context_init_config": task.context_init_config,
                "ground_truth_actions": gt_actions,
            },
            "wiki": env.wiki,
            "tools": filtered_tools,
            "vehicle_state": vehicle_state,
            "fixed_context": fixed_state,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/input", methods=["POST"])
def api_input():
    """Push human input to the orchestrator."""
    if not _session or _session.get("done"):
        return jsonify({"error": "No active session", "done": True})

    data = request.json
    _session["input_queue"].put(data)
    return jsonify({"status": "ok"})


@app.route("/api/llm-turn", methods=["POST"])
def api_llm_turn():
    """Let an LLM play the current turn instead of the human (agent role only)."""
    if not _session or _session.get("done"):
        return jsonify({"error": "No active session", "done": True}), 400

    if _session.get("role") != "agent":
        return jsonify({"error": "LLM turn is only available when playing the agent role"}), 400

    agent = _session.get("agent")
    if not isinstance(agent, HumanWebAgent) or agent._pending_state is None:
        return jsonify({"error": "Not currently waiting for an agent turn"}), 400

    data = request.json or {}
    model = data.get("model") or _session.get("agent_model", "claude-sonnet-4-5-20250929")
    provider = data.get("provider") or _session.get("agent_provider", "anthropic")

    try:
        sidecar = ToolCallingAgent(
            tools_info=agent._pending_tools_info,
            wiki=_session.get("wiki", ""),
            model=model,
            provider=provider,
            temperature=0.0,
        )
        next_message, _ = sidecar.generate_next_message(
            agent._pending_state, agent._pending_tools_info
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    tool_calls = next_message.get("tool_calls")
    if tool_calls:
        human_input = {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"] or "{}"),
                }
                for tc in tool_calls
            ],
        }
    else:
        human_input = {"type": "respond", "content": next_message.get("content") or ""}

    _session["input_queue"].put(human_input)
    return jsonify({"status": "ok", "message": human_input})


@app.route("/api/events")
def api_events():
    """Long-poll for events from the orchestrator."""
    if not _session:
        return jsonify({"events": []})

    output_queue = _session.get("output_queue")
    if not output_queue:
        return jsonify({"events": []})

    events = []
    timeout = float(request.args.get("timeout", 15))
    deadline = time.time() + timeout

    # Wait for at least one event
    try:
        first = output_queue.get(timeout=timeout)
        events.append(first)
    except Empty:
        return jsonify({"events": []})

    # Drain any additional events that are immediately available
    while time.time() < deadline:
        try:
            ev = output_queue.get_nowait()
            events.append(ev)
        except Empty:
            break

    # Serialize events (handle non-serializable types)
    safe_events = []
    for ev in events:
        safe_ev = {}
        for k, v in ev.items():
            try:
                json.dumps(v)
                safe_ev[k] = v
            except (TypeError, ValueError):
                safe_ev[k] = str(v)
        safe_events.append(safe_ev)

    return jsonify({"events": safe_events})


@app.route("/api/state")
def api_state():
    if not _session:
        return jsonify({"error": "No active session"}), 400
    try:
        vehicle_state = context_state.get().model_dump()
        fixed_state = fixed_context.get().model_dump()
    except Exception:
        vehicle_state = {}
        fixed_state = {}
    return jsonify({
        "vehicle_state": vehicle_state,
        "fixed_context": fixed_state,
        "done": _session.get("done", False),
    })


@app.route("/api/wiki")
def api_wiki():
    if not _session:
        return jsonify({"error": "No active session"}), 400
    return jsonify({"wiki": _session.get("wiki", ""), "wiki_raw": WIKI_RAW})


@app.route("/api/tools")
def api_tools():
    if not _session:
        return jsonify({"error": "No active session"}), 400
    return jsonify({"tools": _session.get("tools_info", [])})


@app.route("/api/end-session", methods=["POST"])
def api_end_session():
    if not _session:
        return jsonify({"error": "No active session"}), 400
    _session["done"] = True
    # Signal the orchestrator to stop
    if _session.get("input_queue"):
        _session["input_queue"].put({"type": "stop"})
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import logging

    parser = argparse.ArgumentParser(description="Interactive CAR-bench Task UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show all HTTP request logs")
    args = parser.parse_args()

    # Suppress werkzeug request-level logs unless --verbose
    if not args.verbose and not args.debug:
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    print("Pre-loading mock data...")
    try:
        car_va_data_manager.initialize()
        print("Mock data ready.")
    except Exception as e:
        print(f"Mock data loading failed ({e}).")

    print(f"Starting interactive UI on http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
