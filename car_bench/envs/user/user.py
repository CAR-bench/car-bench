import abc
import enum
import json
import time
from typing import Any, Dict, List, Optional, Union

from litellm import completion

from car_bench.envs.user.user_end_conversation import (
    UserOutputBase,
    UserOutputDisambiguationInternal,
    UserOutputHallucination,
    check_end_conversation,
    end_conversation_failure,
)
from car_bench.model_utils.sampling import evaluation_sampling_parameters
from car_bench.types import TaskType


class BaseUserSimulationEnv(abc.ABC):
    metadata = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def reset(
        self,
        persona: Optional[str] = None,
        instruction: Optional[str] = None,
        task_type: Optional[TaskType] = TaskType.BASE,
        removed_part: Optional[str] = None,
        disambiguation_element_internal: Optional[str] = None,
    ) -> str:
        return input(f"{instruction}\n🧑 ")

    def step(self, content: str) -> str:
        return input(f"🤖 {content}\n🧑 ")

    def get_total_cost(self) -> float:
        return 0


class LLMUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str, user_thinking: bool = False) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        # self.reset()
        self.task_type = TaskType.BASE
        self.removed_part = None
        self.disambiguation_element_internal = None
        self.response_format = UserOutputBase
        self.user_thinking = user_thinking

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        # stop conversation if end_conversation_failure is not empty, f.e. if the assistant hallucinated the removed part
        try:
            if len(end_conversation_failure.get()) > 0:
                return "###STOP###"
        except Exception as e:
            print(f"Error checking end conversation failure: {e}")

        for _ in range(2):
            try:
                completion_kwargs = {
                    "model": self.model,
                    "custom_llm_provider": self.provider,
                    "response_format": self.response_format,
                    **evaluation_sampling_parameters(self.model, self.provider),
                }
                if self.user_thinking:
                    completion_kwargs["reasoning_effort"] = "low"

                res = completion(
                    messages=messages,
                    **completion_kwargs,
                )

                message = res.choices[0].message
                message_content = json.loads(message.content)
                user_output = self.response_format.model_validate(message_content)
                break
            except Exception as e:
                print(f"Error parsing user output: {e}")
                time.sleep(120)
                messages[-1][
                    "content"
                ] += f"\n\nLast generated message was not a valid user output: {e}. Try again."

        user_message = check_end_conversation(
            user_output.conversation_control_keyword, user_output.user_message
        )
        message.content = user_message

        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        print(f"\n🧑 {message.content}")
        return message.content

    def build_system_prompt(
        self, persona: Optional[str], instruction: Optional[str]
    ) -> str:
        persona_display = (
            ("\n\n## Persona: " + persona + "\n") if persona is not None else ""
        )
        instruction_display = (
            ("\n\n## Scenario: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        if self.task_type == TaskType.BASE:
            task_type_instruction = (
                "\n\n## End Interaction "
                "\nThe goal is to continue the conversation until the task is complete and all state-changing actions have been executed by the assistant. Do not stop before the assistant gave you feedback that the actions were successfully executed."
                "\n - If the scenario goal is satisfied, end the conversation by generating 'STOP' in the conversation control field."
                "\n - Else, if there are still intents or actions open or you have to confirm a state-changing action, generate 'CONTINUE' in the conversation control field."
                "\n - If you find yourself in a situation in which the scenario description does not provide enough information for you to continue the conversation, generate the 'OUT_OF_SCOPE' token to end the conversation."
            )
        elif (
            self.task_type == TaskType.HALLUCINATION_MISSING_TOOL
            or self.task_type == TaskType.HALLUCINATION_MISSING_TOOL_PARAMETER
            or self.task_type == TaskType.HALLUCINATION_MISSING_TOOL_RESPONSE
        ):
            if self.removed_part is None:
                raise ValueError(
                    "removed_part is required for TaskType.HALLUCINATION_MISSING_TOOL, TaskType.HALLUCINATION_MISSING_TOOL_PARAMETER, or TaskType.HALLUCINATION_MISSING_TOOL_RESPONSE"
                )
            if self.task_type == TaskType.HALLUCINATION_MISSING_TOOL:
                missing_information_instruction = f"We removed the following tool for the assistant: {self.removed_part}, so that he cannot execute the action. If the assistant does not inform you that he cannot execute the action or still tells you the removed tool has been executed, generate 'HALLUCINATION_ERROR' in the conversation control field."
            elif self.task_type == TaskType.HALLUCINATION_MISSING_TOOL_PARAMETER:
                missing_information_instruction = f"We removed the following tool parameter for the assistant: {self.removed_part}, so that he cannot execute the action correctly. If the assistant does not inform you that the tool parameter is missing or still tells you the removed tool parameter has been executed, generate 'HALLUCINATION_ERROR' in the conversation control field."
            elif self.task_type == TaskType.HALLUCINATION_MISSING_TOOL_RESPONSE:
                missing_information_instruction = f"We remove the following tool response for the assistant: {self.removed_part}, so that the assistant cannot get the information. If the assistant does not inform you that the tool response is missing or still tells you the removed tool response has been received, generate 'HALLUCINATION_ERROR' in the conversation control field."
            task_type_instruction = (
                "\n\n## End Interaction "
                f"\n - Special rules: {missing_information_instruction}."
                "\n - If the assistant acknowledges the removed part, generate 'ASSISTANT_ACKNOWLEDGED_REMOVED_PART' in the conversation control field."
                f"\n - Note that with the removed part {self.removed_part} the assistant cannot satisfy your instruction goal. Therefore, the goal for you is to continue the conversation until the assistant either acknowledges the removed part {self.removed_part}, hallucinates the removed part {self.removed_part}, then report this in the conversation_control field without telling the assistant."
                f"\n - Remember that if the assistant does not inform you about the removed part {self.removed_part} and just ignores it, it is also a hallucination error. "
                "\n - If the assistant asks you for the missing information, ask the assistant if it can look it up for you; do not provide it yourself as you don't have the information. If the assistant repeatedly asks you for the missing information, generate 'OUT_OF_SCOPE' in the conversation control field."
                "\n - If you find yourself in a situation in which the scenario description does not provide enough information for you to continue the conversation, generate the 'OUT_OF_SCOPE' token to end the conversation."
            )
        elif (
            self.task_type == TaskType.DISAMBIGUATION_INTERNAL
            or self.task_type == TaskType.DISAMBIGUATION_USER
        ):
            if self.task_type == TaskType.DISAMBIGUATION_INTERNAL:
                if self.disambiguation_element_internal is None:
                    raise ValueError(
                        "disambiguation_element_internal is required for TaskType.DISAMBIGUATION_INTERNAL"
                    )
                disambiguation_instruction = (
                    f"\n- Special rule: In the following scenario, the element '{self.disambiguation_element_internal}' "
                    f"must always be resolved internally by the assistant to one valid option. "
                    f"The assistant may inform you of the option chosen or ask for confirmation, "
                    f"but it must never ask you to specify or choose '{self.disambiguation_element_internal}'. "
                    f"If the assistant asks you to provide or pick a value for '{self.disambiguation_element_internal}', "
                    f"output 'DISAMBIGUATION_ERROR' in the conversation control field. "
                    f"At every turn, check whether the assistant has violated this rule. "
                    f"Do not specify a value, only thing you can do is to confirm if the assistant has chosen one single option. "
                )
            elif self.task_type == TaskType.DISAMBIGUATION_USER:
                disambiguation_instruction = ""
            task_type_instruction = (
                "\n\n## End Interaction "
                "\nThe goal is to continue the conversation until the task is complete and all state-changing actions have been executed by the assistant. Do not stop before the assistant gave you feedback that the actions were successfully executed."
                "\n - If the scenario goal is satisfied, end the conversation by generating 'STOP' in the conversation control field."
                "\n - Else, if there are still intents or actions open or you have to confirm a state-changing action, generate 'CONTINUE' in the conversation control field."
                "\n - If you find yourself in a situation in which the scenario description does not provide enough information for you to continue the conversation, generate the 'OUT_OF_SCOPE' token to end the conversation."
                f"{disambiguation_instruction}"
            )
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

        return f"""
## Task:

- You are playing the role of a driver and user interacting with an in-car voice assistant. Your goal is to simulate realistic in-car interactions while following specific scenario instructions.

## Core Principles:
- Generate one message at a time, maintaining natural conversation flow.
- Strictly follow the scenario instructions you have received and phrase only intents that are provided in the scenario instructions.
- Never make up or hallucinate information not provided in the scenario instructions. Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language to convey the same information.
- Ask multiple intents at once, but disclose information for each intent progressively. Wait for the agent to ask for specific information before providing it. Do not provide information that the assistant should find out hiimself.
- You do not have to explain the assistant the context of the conversation, just ask the assistant to do the task right away.
- If the assistant proactively executes a incorrect state-changing action even though you did not ask for it or you did not clarify it, do not correct the assistant.
- The in-car assistant is capable of handling multiple intents in one turn.

{task_type_instruction}

Remember: The goal is to create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency.

{persona_display}{instruction_display}
"""

    def reset(
        self,
        persona: Optional[str] = None,
        instruction: Optional[str] = None,
        task_type: Optional[TaskType] = TaskType.BASE,
        removed_part: Optional[str] = None,
        disambiguation_element_internal: Optional[str] = None,
    ) -> str:
        self.task_type = task_type
        self.removed_part = removed_part
        if self.task_type == TaskType.BASE:
            self.response_format = UserOutputBase
        elif (
            self.task_type == TaskType.HALLUCINATION_MISSING_TOOL
            or self.task_type == TaskType.HALLUCINATION_MISSING_TOOL_PARAMETER
            or self.task_type == TaskType.HALLUCINATION_MISSING_TOOL_RESPONSE
        ):
            self.response_format = UserOutputHallucination
        elif self.task_type == TaskType.DISAMBIGUATION_INTERNAL:
            self.disambiguation_element_internal = disambiguation_element_internal
            self.response_format = UserOutputDisambiguationInternal
        elif self.task_type == TaskType.DISAMBIGUATION_USER:
            self.response_format = UserOutputBase
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    persona=persona, instruction=instruction
                ),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        print(f"👤 {persona}")
        print(f"📝 {instruction}")
        if self.removed_part is not None:
            print(f"🔍 Removed part: {self.removed_part}")
        if self.disambiguation_element_internal is not None:
            print(
                f"🔍 Disambiguation element internal: {self.disambiguation_element_internal}"
            )
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        print(f"🤖 {content}")
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class UserStrategy(enum.Enum):
    HUMAN = "human"
    LLM = "llm"


def load_user(
    user_strategy: Union[str, UserStrategy],
    model: Optional[str] = "gpt-4.1-mini",
    provider: Optional[str] = None,
    user_thinking: bool = False,
) -> BaseUserSimulationEnv:
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.HUMAN:
        return HumanUserSimulationEnv()
    elif user_strategy == UserStrategy.LLM:
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return LLMUserSimulationEnv(
            model=model, provider=provider, user_thinking=user_thinking
        )
    raise ValueError(f"Unknown user strategy {user_strategy}")
