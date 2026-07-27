import json
from contextvars import ContextVar
from typing import Any, Dict, List, Literal, Optional, TypedDict

from car_bench.envs.policy_evaluator import policy_errors_during_runtime
from car_bench.envs.tool import Tool


class _PlanningState(TypedDict):
    plans: Dict[str, Dict[str, Any]]
    current_plan_id: Optional[str]


_planning_state: ContextVar[Optional[_PlanningState]] = ContextVar(
    "planning_tool_state", default=None
)


def pretty_print_plan(plan_details: Dict[str, Any]) -> str:
    """Format a plan for display."""
    output = f"Plan: {plan_details['title']} (ID: {plan_details['plan_id']})\n"
    output += "=" * len(output) + "\n\n"
    output += f"Progress: {plan_details['progress']['completed']}/{plan_details['progress']['total']} steps completed "
    if plan_details["progress"]["total"] > 0:
        output += f"({plan_details['progress']['percentage']:.1f}%)\n"
    else:
        output += "(0%)\n"
    output += f"Status: {plan_details['status_counts']['completed']} completed, {plan_details['status_counts']['in_progress']} in progress, {plan_details['status_counts']['blocked']} blocked, {plan_details['status_counts']['executable_but_not_started']} executable, {plan_details['status_counts']['unresolved_dependencies_and_not_started']} waiting for dependencies\n\n"
    output += "Steps:\n"

    # Add each step with its status, notes, and dependencies
    for i, step in enumerate(plan_details["steps"]):
        status_symbol = {
            "not_started": "[ ]",
            "in_progress": "[→]",
            "completed": "[✓]",
            "blocked": "[!]",
            "executable_but_not_started": "[E]",
            "unresolved_dependencies_and_not_started": "[X]",
        }.get(step["status"], "[ ]")

        output += f"{i}. {status_symbol} {step['step_description']}\n"
        if step["notes"]:
            output += f"   Notes: {step['notes']}\n"
        if step["step_dependent_on"]:
            deps = ", ".join(str(dep) for dep in step["step_dependent_on"])
            output += f"   Depends on steps: {deps}\n"

    return output


class PlanningTool(Tool):
    "Planning Tool: allows creating and managing plans for solving complex tasks. Provides functionality for creating plans, updating plan steps, and tracking progress."

    @staticmethod
    def reset() -> None:
        """Install an empty plan store for the current task context."""
        _planning_state.set({"plans": {}, "current_plan_id": None})

    @staticmethod
    def _state() -> _PlanningState:
        """Return the task-local state, creating it for direct tool use if needed."""
        state = _planning_state.get()
        if state is None:
            state = {"plans": {}, "current_plan_id": None}
            _planning_state.set(state)
        return state

    @staticmethod
    def invoke(
        data: Dict[str, Any],
        command: str,
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        step_updates: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Args:
            command (str): The command to execute. Available commands: create, update, list, get, set_active, mark_steps, delete.
            plan_id (str, optional): Unique identifier for the plan.
            title (str, optional): Title for the plan.
            steps (List[Dict[str, Any]], optional): List of plan steps.
            step_updates (List[Dict[str, Any]], optional): List of step updates for mark_steps command. Each update should have:
                - step_index (int): Index of the step to update
                - step_status (str, optional): New status for the step
                - step_notes (str, optional): New notes for the step
        Returns:
            str: JSON string with the result of the operation including status and result data.
        """
        response = {}

        try:
            if command == "create":
                result = PlanningTool._create_plan(plan_id, title, steps)
                response["status"] = "SUCCESS"
                response["result"] = result
            elif command == "update":
                result = PlanningTool._update_plan(plan_id, title, steps)
                response["status"] = "SUCCESS"
                response["result"] = result
            elif command == "list":
                result = PlanningTool._list_plans()
                response["status"] = "SUCCESS"
                response["result"] = result
            elif command == "get":
                result = PlanningTool._get_plan(plan_id)
                response["status"] = "SUCCESS"
                response["result"] = result
            elif command == "set_active":
                result = PlanningTool._set_active_plan(plan_id)
                response["status"] = "SUCCESS"
                response["result"] = result
            elif command == "mark_steps":
                result = PlanningTool._mark_steps(plan_id, step_updates)
                response["status"] = "SUCCESS"
                response["result"] = result
            elif command == "delete":
                result = PlanningTool._delete_plan(plan_id)
                response["status"] = "SUCCESS"
                response["result"] = result
            else:
                response["status"] = "FAILURE"
                response["errors"] = {
                    "PLANNING_TOOL_001": f"Unrecognized command: {command}. Allowed commands are: create, update, list, get, set_active, mark_steps, delete"
                }
        except Exception as e:
            response["status"] = "FAILURE"
            response["errors"] = {"PLANNING_TOOL_ERROR": str(e)}

        return json.dumps(response)

    @staticmethod
    def _check_step_dependencies(plan: Dict, step_index: int) -> bool:
        """Check if all dependencies for a step are completed."""
        step = plan["steps"][step_index]
        for dep_index in step["step_dependent_on"]:
            if plan["step_statuses"][dep_index] != "completed":
                return False
        return True

    @staticmethod
    def _update_step_status(plan: Dict, step_index: int) -> None:
        """Update a step's status based on its dependencies."""
        step = plan["steps"][step_index]
        if not step["step_dependent_on"]:
            # No dependencies, mark as executable if not started
            if plan["step_statuses"][step_index] == "not_started":
                plan["step_statuses"][step_index] = "executable_but_not_started"
        else:
            # Has dependencies, check if all are completed
            if PlanningTool._check_step_dependencies(plan, step_index):
                if (
                    plan["step_statuses"][step_index]
                    == "unresolved_dependencies_and_not_started"
                ):
                    plan["step_statuses"][step_index] = "executable_but_not_started"
            else:
                if plan["step_statuses"][step_index] == "not_started":
                    plan["step_statuses"][
                        step_index
                    ] = "unresolved_dependencies_and_not_started"

    @staticmethod
    def _create_plan(
        plan_id: Optional[str],
        title: Optional[str],
        steps: Optional[List[Dict[str, Any]]],
    ) -> Dict:
        """Create a new plan with the given ID, title, and steps."""
        state = PlanningTool._state()
        plans = state["plans"]

        if not plan_id:
            raise ValueError("Parameter `plan_id` is required for command: create")

        if plan_id in plans:
            raise ValueError(
                f"A plan with ID '{plan_id}' already exists. Use 'update' to modify existing plans."
            )

        if not title:
            raise ValueError("Parameter `title` is required for command: create")

        if not steps or not isinstance(steps, list):
            raise ValueError(
                "Parameter `steps` must be a non-empty list for command: create"
            )

        # Validate step structure
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} must be a dictionary")
            if "step_description" not in step:
                raise ValueError(f"Step {i} must contain 'step_description' field")
            if "step_dependent_on" not in step:
                raise ValueError(f"Step {i} must contain 'step_dependent_on' field")
            if not isinstance(step["step_description"], str):
                raise ValueError(f"Step {i} 'step_description' must be a string")
            if not isinstance(step["step_dependent_on"], list):
                raise ValueError(f"Step {i} 'step_dependent_on' must be a list")
            if not all(isinstance(dep, int) for dep in step["step_dependent_on"]):
                raise ValueError(
                    f"Step {i} 'step_dependent_on' must contain only integers"
                )
            if any(dep >= i for dep in step["step_dependent_on"]):
                raise ValueError(f"Step {i} cannot depend on itself or future steps")

        # Create a new plan with initialized step statuses
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
            "step_notes": [""] * len(steps),
        }

        # Initialize step statuses based on dependencies
        for i in range(len(steps)):
            PlanningTool._update_step_status(plan, i)

        plans[plan_id] = plan
        state["current_plan_id"] = plan_id

        return {
            "plan_created": True,
            "plan_id": plan_id,
            "plan_details": PlanningTool._format_plan(plan),
        }

    @staticmethod
    def _update_plan(
        plan_id: Optional[str],
        title: Optional[str],
        steps: Optional[List[Dict[str, Any]]],
    ) -> Dict:
        """Update an existing plan with new title or steps."""
        plans = PlanningTool._state()["plans"]

        if not plan_id:
            raise ValueError("Parameter `plan_id` is required for command: update")

        if plan_id not in plans:
            raise ValueError(f"No plan found with ID: {plan_id}")

        plan = plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, dict) for step in steps
            ):
                raise ValueError(
                    "Parameter `steps` must be a list of dictionaries for command: update"
                )

            # Preserve existing step statuses for unchanged steps
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # Create new step statuses and notes
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # If the step exists at the same position in old steps, preserve status and notes
                if (
                    i < len(old_steps)
                    and step["step_description"] == old_steps[i]["step_description"]
                ):
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

        return {
            "plan_updated": True,
            "plan_id": plan_id,
            "plan_details": PlanningTool._format_plan(plan),
        }

    @staticmethod
    def _list_plans() -> Dict:
        """List all available plans."""
        state = PlanningTool._state()
        plans = state["plans"]
        current_plan_id = state["current_plan_id"]
        plans_list = []

        for plan_id, plan in plans.items():
            current_marker = plan_id == current_plan_id
            completed = sum(
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])

            plans_list.append(
                {
                    "plan_id": plan_id,
                    "title": plan["title"],
                    "is_active": current_marker,
                    "progress": {
                        "completed": completed,
                        "total": total,
                        "percentage": (completed / total * 100) if total > 0 else 0,
                    },
                }
            )

        return {
            "plans": plans_list,
            "active_plan_id": current_plan_id,
            "total_plans": len(plans_list),
        }

    @staticmethod
    def _get_plan(plan_id: Optional[str]) -> Dict:
        """Get details of a specific plan."""
        state = PlanningTool._state()
        plans = state["plans"]
        current_plan_id = state["current_plan_id"]

        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not current_plan_id:
                raise ValueError(
                    "No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = current_plan_id

        if plan_id not in plans:
            raise ValueError(f"No plan found with ID: {plan_id}")

        plan = plans[plan_id]
        return {
            "plan_id": plan_id,
            "plan_details": PlanningTool._format_plan(plan),
            "is_active": plan_id == current_plan_id,
        }

    @staticmethod
    def _set_active_plan(plan_id: Optional[str]) -> Dict:
        """Set a plan as the active plan."""
        state = PlanningTool._state()
        plans = state["plans"]

        if not plan_id:
            raise ValueError("Parameter `plan_id` is required for command: set_active")

        if plan_id not in plans:
            raise ValueError(f"No plan found with ID: {plan_id}")

        state["current_plan_id"] = plan_id
        return {
            "active_plan_set": True,
            "plan_id": plan_id,
            "plan_details": PlanningTool._format_plan(plans[plan_id]),
        }

    @staticmethod
    def _mark_steps(
        plan_id: Optional[str], step_updates: Optional[List[Dict[str, Any]]]
    ) -> Dict:
        """Mark multiple steps with specific statuses and optional notes."""
        state = PlanningTool._state()
        plans = state["plans"]
        current_plan_id = state["current_plan_id"]

        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not current_plan_id:
                raise ValueError(
                    "No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = current_plan_id

        if plan_id not in plans:
            raise ValueError(f"No plan found with ID: {plan_id}")

        if not step_updates or not isinstance(step_updates, list):
            raise ValueError(
                "Parameter `step_updates` must be a non-empty list for command: mark_steps"
            )

        plan = plans[plan_id]
        updates_applied = []
        completed_steps = set()

        # First pass: validate all updates
        for update in step_updates:
            if not isinstance(update, dict):
                raise ValueError("Each step update must be a dictionary")
            if "step_index" not in update:
                raise ValueError("Each step update must contain 'step_index' field")
            if not isinstance(update["step_index"], int):
                raise ValueError("Step index must be an integer")
            if update["step_index"] < 0 or update["step_index"] >= len(plan["steps"]):
                raise ValueError(
                    f"Invalid step_index: {update['step_index']}. Valid indices range from 0 to {len(plan['steps'])-1}."
                )
            if "step_status" in update and update["step_status"] not in [
                "not_started",
                "in_progress",
                "completed",
                "blocked",
            ]:
                raise ValueError(
                    f"Invalid step_status: {update['step_status']}. Valid statuses are: not_started, in_progress, completed, blocked"
                )

        # Second pass: apply updates
        for update in step_updates:
            step_index = update["step_index"]
            old_status = plan["step_statuses"][step_index]

            if "step_status" in update:
                plan["step_statuses"][step_index] = update["step_status"]
                if update["step_status"] == "completed":
                    completed_steps.add(step_index)

            if "step_notes" in update:
                plan["step_notes"][step_index] = update["step_notes"]

            updates_applied.append(
                {
                    "step_index": step_index,
                    "old_status": old_status,
                    "new_status": plan["step_statuses"][step_index],
                }
            )

        # Third pass: update dependent steps for all completed steps
        for completed_step in completed_steps:
            for i in range(len(plan["steps"])):
                if completed_step in plan["steps"][i]["step_dependent_on"]:
                    PlanningTool._update_step_status(plan, i)

        return {
            "steps_updated": True,
            "plan_id": plan_id,
            "updates_applied": updates_applied,
            "plan_details": PlanningTool._format_plan(plan),
        }

    @staticmethod
    def _delete_plan(plan_id: Optional[str]) -> Dict:
        """Delete a plan."""
        state = PlanningTool._state()
        plans = state["plans"]

        if not plan_id:
            raise ValueError("Parameter `plan_id` is required for command: delete")

        if plan_id not in plans:
            raise ValueError(f"No plan found with ID: {plan_id}")

        del plans[plan_id]

        # If the deleted plan was the active plan, clear the active plan
        was_active = state["current_plan_id"] == plan_id
        if was_active:
            state["current_plan_id"] = None

        return {"plan_deleted": True, "plan_id": plan_id, "was_active_plan": was_active}

    @staticmethod
    def _format_plan(plan: Dict) -> Dict:
        """Format a plan for display."""
        # Calculate progress statistics
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        executable = sum(
            1
            for status in plan["step_statuses"]
            if status == "executable_but_not_started"
        )
        unresolved_dependencies = sum(
            1
            for status in plan["step_statuses"]
            if status == "unresolved_dependencies_and_not_started"
        )
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        # Format steps with status and notes
        formatted_steps = []
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"])
        ):
            formatted_steps.append(
                {
                    "index": i,
                    "step_description": step["step_description"],
                    "step_dependent_on": step["step_dependent_on"],
                    "status": status,
                    "notes": notes,
                }
            )

        return {
            "title": plan["title"],
            "plan_id": plan["plan_id"],
            "progress": {
                "completed": completed,
                "total": total_steps,
                "percentage": (completed / total_steps * 100) if total_steps > 0 else 0,
            },
            "status_counts": {
                "completed": completed,
                "in_progress": in_progress,
                "blocked": blocked,
                "executable_but_not_started": executable,
                "unresolved_dependencies_and_not_started": unresolved_dependencies,
                "not_started": not_started,
            },
            "steps": formatted_steps,
            "is_active": (
                plan["plan_id"] == PlanningTool._state()["current_plan_id"]
            ),
        }

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Tool description visible to LLM.
        """
        return {
            "type": "function",
            "function": {
                "name": "planning_tool",
                "description": "A planning tool that allows creating and managing plans for solving complex tasks. Provides functionality for creating plans, updating plan steps, and tracking progress. Last step of the plan should always be to check if all user intents could be resolved and if not, to note and communicate which intents could not be resolved (because no available tool, data, etc.).",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": [
                                "create",
                                "update",
                                "list",
                                "get",
                                "set_active",
                                "mark_steps",
                                "delete",
                            ],
                            "description": "The command to execute. Available commands: create, update, list, get, set_active, mark_steps, delete.",
                        },
                        "plan_id": {
                            "type": "string",
                            "description": "Unique identifier for the plan. Required for create, update, set_active, and delete commands. Optional for get and mark_steps (uses active plan if not specified).",
                        },
                        "title": {
                            "type": "string",
                            "description": "Title for the plan. Required for create command, optional for update command.",
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["step_description", "step_dependent_on"],
                                "properties": {
                                    "step_description": {
                                        "type": "string",
                                        "description": "Description of what needs to be done in this step",
                                    },
                                    "step_dependent_on": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "description": "List of step indices (0-based) that this step depends on the result of. Cannot depend on itself or future steps. No dependencies means that the step can be executed independently.",
                                    },
                                },
                            },
                            "description": "List of plan steps. Each step must have a description and list of dependencies. Required for create command, optional for update command.",
                        },
                        "step_updates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["step_index"],
                                "properties": {
                                    "step_index": {
                                        "type": "integer",
                                        "description": "Index of the step to update (0-based)",
                                    },
                                    "step_status": {
                                        "type": "string",
                                        "enum": [
                                            "not_started",
                                            "in_progress",
                                            "completed",
                                            "blocked",
                                        ],
                                        "description": "Status to set for the step",
                                    },
                                    "step_notes": {
                                        "type": "string",
                                        "description": "Additional notes for the step. Can be used for intermediate result notes.",
                                    },
                                },
                            },
                            "description": "List of step updates for mark_steps command. Each update can specify a step's status and notes. Only mark as completed if expected result is found, if another tool execution is needed for step mark as in_progress.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }
