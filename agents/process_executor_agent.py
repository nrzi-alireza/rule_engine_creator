import json
import logging
from datetime import datetime

import yaml

from helper.llm import stream_llm
from helper.openai_function import openai_function
from helper.rule_engine import SimpleRulesEngine
from helper.workflow_engine import (
    ExecutionNode,
    InputNode,
    State,
    StreamResult,
    WorkflowEngine,
)

logger = logging.getLogger(__name__)


@openai_function
def end_process():
    """
    End the current process and return to the user.
    """
    return "Process ended successfully"


class ProcessExecutorState(State):
    process_id: int = None
    topic: str = None
    started: bool = False
    current_step: int = 0
    messages_history: list[dict[str, str]] = []
    rule_engine: SimpleRulesEngine | None = None

    def __init__(self, topic: str, process_id: int):
        super().__init__()
        self.topic = topic
        self.process_id = process_id
        self.started = False
        self.current_step = -1
        self.messages_history = []
        self.rule_engine = SimpleRulesEngine(topic=topic)


class ProcessExecutorAgent(WorkflowEngine):
    def __init__(self, topic: str, process_id: int):
        super().__init__(
            state=ProcessExecutorState(topic=topic, process_id=process_id),
        )
        with open("artifacts/prompts/process_executor.yaml", "r") as f:
            self.executor_prompts = yaml.safe_load(f)

        self.set_start_nodes(
            [
                ExecutionNode(
                    function=self.chat_with_user,
                    args=(self,),
                    kwargs={"system_prompt": self.create_system_prompt()},
                )
            ],
        )

    def is_started(self) -> bool:
        return self.state.started

    def set_started(self):
        self.state.started = True

    def reset(self):
        self.state.started = False
        self.state.current_step = -1
        self.state.messages_history = []

    def before_step(self, external_input: str, state: State) -> None:
        state.messages_history.append({"role": "user", "content": external_input})

    def get_messages(self) -> list[dict[str, str]]:
        return self.state.messages_history[1:]  # ignore the start message

    def create_system_prompt(self) -> str:
        system_prompt_template: str = self.executor_prompts["system_prompt"]
        process = self.state.rule_engine.get_process_by_id(self.state.process_id)
        return system_prompt_template.format(
            process_flow=process.process_flow,
            steps=json.dumps([s.model_dump() for s in process.steps]),
            domain=self.state.topic,
            current_date_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def chat_with_user(self, *args, **kwargs):
        state: ProcessExecutorState = kwargs["state"]
        system_prompt: str = kwargs["system_prompt"]

        llm_response = stream_llm(
            system_prompt=system_prompt,
            user_prompt=state.messages_history[-1]["content"],
            messages_history=state.messages_history[:-1],
            tools=[
                end_process.schema,
            ],
        )

        assistant_output = ""
        tool_calls = []
        for chunk in llm_response:
            if isinstance(chunk, str):
                assistant_output += chunk
                yield StreamResult(chunk=chunk)
            elif isinstance(chunk, list):
                tool_calls.extend(chunk)

        state.messages_history.append(
            {"role": "assistant", "content": assistant_output, "tool_calls": tool_calls if tool_calls else None}
        )

        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                print(f"@@@@ TOOL CALL: {function_name}({arguments})", "\n\n")

                related_messages_history = []
                for msg in self.state.messages_history:
                    if "tool_calls" not in msg and msg["role"] != "tool":
                        related_messages_history.append(msg)

                if function_name == "end_process":
                    function_response = end_process().execute()
                    yield StreamResult(
                        chunk="\n\nProcess ended. You can ask me anything about the process.\n\n", is_complete=False
                    )
                else:
                    raise ValueError(f"Unknown function {function_name}")

                self.state.messages_history.append(
                    {"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(function_response)}
                )

        yield StreamResult(
            chunk="",
            is_complete=True,
            next_nodes=[
                InputNode(),
                ExecutionNode(
                    function=self.chat_with_user,
                    args=(self,),
                    kwargs={"system_prompt": self.create_system_prompt()},
                ),
            ],
        )
