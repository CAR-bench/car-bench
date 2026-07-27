import abc
import contextvars
import enum
import json
from typing import Any, Dict, List, Optional, Union

from litellm import completion
from pydantic import BaseModel

from car_bench.model_utils.sampling import evaluation_sampling_parameters

policy_errors_during_runtime: contextvars.ContextVar[List[str]] = (
    contextvars.ContextVar("policy_errors_during_runtime")
)


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _tool_call_name_and_arguments(
    tool_call: Dict[str, Any],
) -> tuple[Optional[str], Dict[str, Any]]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None, {}
    name = function.get("name")
    arguments = _json_object(function.get("arguments"))
    return name if isinstance(name, str) else None, arguments


def _evaluate_fog_light_batches(trajectory: List[Dict[str, Any]]) -> None:
    """Evaluate AUT-POL:013 using the state at each assistant tool batch.

    Supporting low/high-beam adjustments in the same batch are treated as
    atomic companions. Adjustments deferred to a later assistant batch do not
    retroactively make the fog-light activation compliant.
    """

    low_beams: Optional[bool] = None
    high_beams: Optional[bool] = None

    for step in trajectory:
        role = step.get("role")
        if role == "tool" and step.get("name") == "get_exterior_lights_status":
            observation = _json_object(step.get("content"))
            result = observation.get("result")
            if isinstance(result, dict):
                observed_low = result.get("head_lights_low_beams")
                observed_high = result.get("head_lights_high_beams")
                if isinstance(observed_low, bool):
                    low_beams = observed_low
                if isinstance(observed_high, bool):
                    high_beams = observed_high
            continue

        if role != "assistant":
            continue
        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        batch_low_beams = low_beams
        batch_high_beams = high_beams
        activates_fog_lights = False

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name, arguments = _tool_call_name_and_arguments(tool_call)
            if name == "set_head_lights_low_beams" and isinstance(
                arguments.get("on"), bool
            ):
                batch_low_beams = arguments["on"]
            elif name == "set_head_lights_high_beams" and isinstance(
                arguments.get("on"), bool
            ):
                batch_high_beams = arguments["on"]
            elif name == "set_fog_lights" and arguments.get("on") is True:
                activates_fog_lights = True

        if activates_fog_lights and (
            batch_low_beams is not True or batch_high_beams is not False
        ):
            policy_errors_during_runtime.get().append(
                "AUT-POL:013: Policy not followed. Fog lights were activated "
                "without low beams ON and high beams OFF by the end of the "
                "same assistant tool-call batch."
            )

        low_beams = batch_low_beams
        high_beams = batch_high_beams


class PolicyEvaluatorResponseFormat(BaseModel):
    reasoning: str
    policy_followed: bool


class BasePolicyEvaluatorEnv(abc.ABC):
    metadata = {}

    @abc.abstractmethod
    def evaluate_llm(self, policy: str, trajectory: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_aut(self, trajectory: List[Dict]) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanPolicyEvaluatorEnv(BasePolicyEvaluatorEnv):
    def evaluate_llm(self, policy: str, trajectory: str) -> str:
        raise NotImplementedError

    def evaluate_aut(self, trajectory: List[Dict]) -> str:
        raise NotImplementedError

    def get_total_cost(self) -> float:
        raise NotImplementedError


class LLMPolicyEvaluatorEnv(BasePolicyEvaluatorEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__()
        self.model = model
        self.provider = provider
        self.total_cost = 0.0

    def generate_evaluation(self, messages: List[Dict[str, Any]]) -> str:
        # response format is PolicyEvaluatorResponseFormat
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            response_format=PolicyEvaluatorResponseFormat,
            **evaluation_sampling_parameters(self.model, self.provider),
        )
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        return message.content

    def build_prompt(self, policy: str, trajectory: str) -> str:
        LLM_POL_INSTRUCTION = f"""You are an expert data labeler. 
Your task is to grade the adherence of an AI agent to a domain-specific policy within the internal trajectory. 
If the policy is not applicable to the trajectory, or you don't have enough information to evaluate the policy, you should reason "Not Applicable" and return True for policy_followed. 
Keep the reasoning short (1-2 sentences) and clear.

### Policy
Grade if following policy is followed: 
{policy} 

### Trajectory
Within the following trajectory:
<trajectory> 
{trajectory} 
</trajectory>

If the user asks for something that invalidates the policy, you should reason "Not Applicable" and return True for policy_followed.
        """
        return LLM_POL_INSTRUCTION

    def evaluate_llm(self, policy: str, trajectory: str) -> str:
        self.messages = [
            {
                "role": "user",
                "content": self.build_prompt(policy=policy, trajectory=trajectory),
            }
        ]
        return self.generate_evaluation(self.messages)

    def evaluate_aut(self, trajectory: List[Dict]) -> str:
        from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
            context_state,
        )

        vehicle_ctx = context_state.get()
        _evaluate_fog_light_batches(trajectory)
        # actions = [tool_call for step in trajectory for tool_call in step["tool_calls"] if step["role"] == "assistant"]
        for idx, step in enumerate(trajectory):
            if step["role"] == "assistant":
                if "tool_calls" not in step:
                    continue
                if step["tool_calls"] is None:
                    continue
                tool_calls = step["tool_calls"]
                previous_tool_calls = [
                    tool_call["function"]["name"]
                    for step in trajectory[:idx]
                    for tool_call in (step.get("tool_calls") or [])
                    if step["role"] == "assistant" and step.get("tool_calls")
                ]
                if "open_close_sunroof" in [
                    tool_call["function"]["name"] for tool_call in tool_calls
                ]:
                    open_close_sunroof_tool_call = [
                        tool_call
                        for tool_call in tool_calls
                        if tool_call["function"]["name"] == "open_close_sunroof"
                    ][0]
                    open_close_sunroof_tool_call_arguments = json.loads(
                        open_close_sunroof_tool_call["function"]["arguments"]
                    )
                    if open_close_sunroof_tool_call_arguments["percentage"] != 0:
                        # AUT-POL:005
                        if (
                            vehicle_ctx.sunshade_position
                            < open_close_sunroof_tool_call_arguments["percentage"]
                            and not "open_close_sunshade"
                            in [
                                tool_call["function"]["name"]
                                for tool_call in tool_calls
                            ]
                        ):
                            policy_errors_during_runtime.get().append(
                                "AUT-POL:005:The sunroof can only be opened if the sunshade is already fully opened or the sunshade is currently opened in parallel. Otherwise the operation will be blocked."
                            )
                        # AUT-POL:009:
                        if not "get_weather" in previous_tool_calls:
                            policy_errors_during_runtime.get().append(
                                "AUT-POL:009: Weather condition not checked before opening the sunroof."
                            )
                if "set_window_defrost" in [
                    tool_call["function"]["name"] for tool_call in tool_calls
                ]:
                    set_window_defrost_tool_call = [
                        tool_call
                        for tool_call in tool_calls
                        if tool_call["function"]["name"] == "set_window_defrost"
                    ][0]
                    set_window_defrost_tool_call_arguments = json.loads(
                        set_window_defrost_tool_call["function"]["arguments"]
                    )
                    if set_window_defrost_tool_call_arguments["on"] is True:
                        if (
                            set_window_defrost_tool_call_arguments["defrost_window"]
                            == "ALL"
                            or set_window_defrost_tool_call_arguments["defrost_window"]
                            == "FRONT"
                        ):
                            # AUT-POL:010
                            if not (
                                "get_climate_settings" in previous_tool_calls
                                or (
                                    "set_fan_speed" in previous_tool_calls
                                    and "set_fan_airflow_direction"
                                    in previous_tool_calls
                                    and "set_air_conditioning" in previous_tool_calls
                                )
                            ):
                                policy_errors_during_runtime.get().append(
                                    "AUT-POL:010: Climate settings not checked before activating front window defrost."
                                )
                            if (
                                (vehicle_ctx.fan_speed < 2)
                                or (
                                    not "WINDSHIELD"
                                    in vehicle_ctx.fan_airflow_direction
                                )
                                or (vehicle_ctx.air_conditioning == False)
                            ):
                                # AUT-POL:010
                                policy_errors_during_runtime.get().append(
                                    "AUT-POL:010: Policy not followed."
                                )
                if "set_air_conditioning" in [
                    tool_call["function"]["name"] for tool_call in tool_calls
                ]:
                    set_air_conditioning_tool_call = [
                        tool_call
                        for tool_call in tool_calls
                        if tool_call["function"]["name"] == "set_air_conditioning"
                    ][0]
                    set_air_conditioning_tool_call_arguments = json.loads(
                        set_air_conditioning_tool_call["function"]["arguments"]
                    )
                    if set_air_conditioning_tool_call_arguments["on"] is True:
                        # AUT-POL:011
                        if not (
                            "get_climate_settings" in previous_tool_calls
                            or (
                                "open_close_window" in previous_tool_calls
                                and "set_fan_speed" in previous_tool_calls
                            )
                            or not (
                                "get_vehicle_window_positions" in previous_tool_calls
                                or "open_close_window" in previous_tool_calls
                            )
                        ):
                            # AUT-POL:011
                            policy_errors_during_runtime.get().append(
                                "AUT-POL:011: Climate settings or window positions not checked before activating air conditioning."
                            )
                        if (
                            any(
                                [
                                    vehicle_ctx.window_driver_position > 20,
                                    vehicle_ctx.window_passenger_position > 20,
                                    vehicle_ctx.window_driver_rear_position > 20,
                                    vehicle_ctx.window_passenger_rear_position > 20,
                                ]
                            )
                            or vehicle_ctx.fan_speed == 0
                        ):
                            # AUT-POL:011
                            policy_errors_during_runtime.get().append(
                                "AUT-POL:011: Policy not followed. Either one or more windows are open more than 20% or the fan speed is 0."
                            )
                if "set_fog_lights" in [
                    tool_call["function"]["name"] for tool_call in tool_calls
                ]:
                    set_fog_lights_tool_call = [
                        tool_call
                        for tool_call in tool_calls
                        if tool_call["function"]["name"] == "set_fog_lights"
                    ][0]
                    set_fog_lights_tool_call_arguments = json.loads(
                        set_fog_lights_tool_call["function"]["arguments"]
                    )
                    if set_fog_lights_tool_call_arguments["on"] is True:
                        # AUT-POL:013
                        if not (
                            "get_exterior_lights_status" in previous_tool_calls
                            or (
                                "set_head_lights_low_beams" in previous_tool_calls
                                and "set_head_lights_high_beams" in previous_tool_calls
                            )
                        ):
                            policy_errors_during_runtime.get().append(
                                "AUT-POL:013: Low and high beam headlights not checked before activating fog lights."
                            )
                if "set_head_lights_high_beams" in [
                    tool_call["function"]["name"] for tool_call in tool_calls
                ]:
                    set_head_lights_high_beams_tool_call = [
                        tool_call
                        for tool_call in tool_calls
                        if tool_call["function"]["name"] == "set_head_lights_high_beams"
                    ][0]
                    set_head_lights_high_beams_tool_call_arguments = json.loads(
                        set_head_lights_high_beams_tool_call["function"]["arguments"]
                    )
                    if set_head_lights_high_beams_tool_call_arguments["on"] is True:
                        # AUT-POL:014
                        if not (
                            "get_exterior_lights_status" in previous_tool_calls
                            or "set_fog_lights" in previous_tool_calls
                        ):
                            policy_errors_during_runtime.get().append(
                                "AUT-POL:014: Fog lights not checked before activating high beam headlights."
                            )
                        if vehicle_ctx.fog_lights is True:
                            # AUT-POL:014
                            policy_errors_during_runtime.get().append(
                                "AUT-POL:014: Policy not followed."
                            )
                navigation_edit_tools = [
                    "navigation_add_one_waypoint",
                    "navigation_delete_final_destination",
                    "navigation_delete_one_waypoint",
                    "navigation_replace_final_destination",
                    "navigation_replace_one_waypoint",
                ]
                if (
                    sum(
                        tool_call["function"]["name"] in navigation_edit_tools
                        for tool_call in tool_calls
                    )
                    > 1
                ):
                    # TECH-AUT-POL:018
                    policy_errors_during_runtime.get().append(
                        "TECH-AUT-POL:018: Only one navigation editing tool can be used in parallel per step."
                    )

    def get_total_cost(self) -> float:
        return self.total_cost


class ReactUserSimulationEnv(LLMPolicyEvaluatorEnv):
    def evaluate_llm(self, policy: str, trajectory: str) -> str:
        raise NotImplementedError

    def evaluate_aut(self, trajectory: List[Dict]) -> str:
        raise NotImplementedError

    def get_total_cost(self) -> float:
        raise NotImplementedError


class VerifyUserSimulationEnv(LLMPolicyEvaluatorEnv):
    def evaluate_llm(self, policy: str, trajectory: str) -> str:
        raise NotImplementedError

    def evaluate_aut(self, trajectory: List[Dict]) -> str:
        raise NotImplementedError

    def get_total_cost(self) -> float:
        raise NotImplementedError


class ReflectionUserSimulationEnv(LLMPolicyEvaluatorEnv):
    def evaluate_llm(self, policy: str, trajectory: str) -> str:
        raise NotImplementedError

    def evaluate_aut(self, trajectory: List[Dict]) -> str:
        raise NotImplementedError

    def get_total_cost(self) -> float:
        raise NotImplementedError


class PolicyEvaluatorStrategy(enum.Enum):
    HUMAN = "human"
    LLM = "llm"
    REACT = "react"
    VERIFY = "verify"
    REFLECTION = "reflection"


def load_policy_evaluator(
    policy_evaluator_strategy: Union[str, PolicyEvaluatorStrategy],
    model: Optional[str] = "gpt-4.1-mini",
    provider: Optional[str] = None,
) -> BasePolicyEvaluatorEnv:
    if isinstance(policy_evaluator_strategy, str):
        policy_evaluator_strategy = PolicyEvaluatorStrategy(policy_evaluator_strategy)
    if policy_evaluator_strategy == PolicyEvaluatorStrategy.HUMAN:
        return HumanPolicyEvaluatorEnv()
    elif policy_evaluator_strategy == PolicyEvaluatorStrategy.LLM:
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return LLMPolicyEvaluatorEnv(model=model, provider=provider)
    elif policy_evaluator_strategy == PolicyEvaluatorStrategy.REACT:
        if model is None:
            raise ValueError("React user strategy requires a model")
        if provider is None:
            raise ValueError("React user strategy requires a model provider")
        return ReactUserSimulationEnv(model=model, provider=provider)
    elif policy_evaluator_strategy == PolicyEvaluatorStrategy.VERIFY:
        if model is None:
            raise ValueError("Verify user strategy requires a model")
        if provider is None:
            raise ValueError("Verify user strategy requires a model provider")
        return VerifyUserSimulationEnv(model=model, provider=provider)
    elif policy_evaluator_strategy == PolicyEvaluatorStrategy.REFLECTION:
        if model is None:
            raise ValueError("Reflection user strategy requires a model")
        if provider is None:
            raise ValueError("Reflection user strategy requires a model provider")
        return ReflectionUserSimulationEnv(model=model, provider=provider)
    raise ValueError(f"Unknown user strategy {policy_evaluator_strategy}")
