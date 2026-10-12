import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"
USER_AS_A_TOOL_ACTION_NAMES = [
    "respond",
    "message_to_user",
    "clarify_with_user_request_of_invalid_tool_arguments",
    "clarify_with_user_request_of_unavailable_tools",
    "disambiguate_with_user_suitable_tool_results",
    "disambiguate_with_user_suitable_tools",
    "disambiguate_with_user_tool_arguments",
    "ask_user_for_confirmation",
]


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]
    index: Optional[int] = None
    dependent_on_action_index: Optional[Union[int, List[int]]] = None


class TaskType(str, Enum):
    BASE = "base"
    HALLUCINATION_MISSING_TOOL = "hallucination_missing_tool"
    HALLUCINATION_MISSING_TOOL_PARAMETER = "hallucination_missing_tool_parameter"
    HALLUCINATION_MISSING_TOOL_RESPONSE = "hallucination_missing_tool_response"
    DISAMBIGUATION_INTERNAL = "disambiguation_internal"
    DISAMBIGUATION_USER = "disambiguation_user"


class Task(BaseModel):
    task_id: str
    calendar_id: str
    actions: List[Action]
    persona: str
    instruction: str
    context_init_config: Dict[str, Any]
    task_type: TaskType
    disambiguation_element_internal: Optional[str] = None
    disambiguation_element_user: Optional[str] = None
    disambiguation_element_note: Optional[str] = None
    removed_part: Optional[List[str]] = None


class RewardOutputInfo(BaseModel):
    r_outputs: float
    outputs: Dict[str, bool]


class RewardActionInfo(BaseModel):
    r_actions: Optional[float] = None
    gt_vehicle_state_hash: Optional[str] = None


class RewardInfo(BaseModel):
    r_actions: Optional[float] = None  # Legacy field for backward compatibility
    r_actions_final: Optional[float] = None  # Final state correctness
    r_actions_intermediate: Optional[float] = None  # Intermediate states correctness
    r_tool_subset: Optional[float] = None
    tool_subset_missing_tools: Optional[List[str]] = (
        []
    )  # Tools from ground truth that were not performed
    r_tool_execution: Optional[float] = None
    tool_execution_errors: Optional[List[str]] = []
    r_policy: Optional[float] = None
    policy_llm_errors: Optional[List[str]] = []
    policy_aut_errors: Optional[List[str]] = []
    r_user_end_conversation: Optional[float] = None
    end_conversation_keyword: Optional[str] = None
    r_outputs: Optional[float] = None
    outputs: Optional[Dict[str, bool]] = {}


class RewardResult(BaseModel):
    reward: float
    info: Union[RewardOutputInfo, RewardActionInfo, RewardInfo]
    actions: List[Action]


class AgentState(BaseModel):
    """State of the agent during execution."""

    messages: List[Dict[str, Any]]
    total_cost: float = 0.0
    total_llm_induced_latency_ms: float = 0.0
    turn_counter: int = 0
    least_prompt_tokens: float = float("inf")
    latest_prompt_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    start_time_s: float = 0.0

    @property
    def average_llm_induced_latency_per_turn_ms(self) -> float:
        return (
            self.total_llm_induced_latency_ms / self.turn_counter
            if self.turn_counter > 0
            else 0.0
        )

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def elapsed_time_s(self) -> float:
        """Wall-clock seconds since the agent state was created."""
        if self.start_time_s <= 0:
            return 0.0
        return time.time() - self.start_time_s

    @property
    def tokens_per_minute(self) -> float:
        """Compute tokens per minute based on wall-clock elapsed time."""
        elapsed = self.elapsed_time_s
        if elapsed <= 0:
            return 0.0
        return self.total_tokens / (elapsed / 60.0)


class SolveResult(BaseModel):
    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    total_cost: Optional[float] = None


class EnvInfo(BaseModel):
    task: Task
    source: Optional[str] = None
    user_cost: Optional[float] = None
    reward_info: Optional[RewardResult] = None


class EnvResponse(BaseModel):
    observation: Union[str, list[str]]
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    observation: str
    info: EnvInfo


class EnvRunResult(BaseModel):
    task_index: int  # Runtime index of the task in the task list
    task_id: Optional[str] = None  # Persistent task identifier (e.g., "base_0")
    reward: float
    info: Dict[str, Any]
    traj: List[Dict[str, Any]]
    trial: int
