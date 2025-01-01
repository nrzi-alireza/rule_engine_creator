# A simple & pure python workflow engine

from collections import deque
from multiprocessing import Value
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field


class InputNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pass


class ExecutionNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: Callable[..., tuple[list["ExecutionNode"], Any]]  # Returns (next_nodes, result)
    args: Optional[tuple[Any, ...]] = None
    kwargs: Optional[dict[str, Any]] = None

    @property
    def name(self) -> str:
        return self.function.__name__


class StreamResult(BaseModel):
    chunk: str
    is_complete: bool = Field(default=False)
    next_nodes: Optional[list[ExecutionNode | InputNode]] = None


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowEngine:
    def __init__(self, state: State):
        self.state = state
        self.execution_queue = deque()

    def set_start_nodes(self, start_nodes: list[ExecutionNode]) -> None:
        self.execution_queue = deque(start_nodes)

    def before_step(self, external_input: str, state: State) -> None:
        pass

    def after_step(self, state: State) -> None:
        pass

    def step_until_external_input(self, external_input: str):
        self.before_step(external_input, self.state)

        while True:
            if not self.execution_queue:
                raise ValueError("Execution queue is empty - no steps to execute")

            node_to_execute = self.execution_queue.popleft()

            if isinstance(node_to_execute, InputNode):
                break
            elif isinstance(node_to_execute, ExecutionNode):
                func = node_to_execute.function
                args = node_to_execute.args if node_to_execute.args else ()
                kwargs = node_to_execute.kwargs if node_to_execute.kwargs else {}

                if kwargs and ("user_input" in kwargs or "state" in kwargs):
                    raise ValueError("User input node cannot have user_input in kwargs")

                stream_gen: StreamResult = func(
                    *args,
                    **{
                        **kwargs,
                        "input": external_input,
                        "state": self.state,
                    },
                )

                while True:
                    chunk_result = next(stream_gen)
                    if chunk_result.is_complete:
                        break
                    yield chunk_result.chunk

                next_nodes = chunk_result.next_nodes  # last chunk

                if next_nodes:
                    self.execution_queue.extend(next_nodes)

            else:
                raise Value(f"Unknown node in execution queue with type {type(node_to_execute)}")

        self.after_step(self.state)
