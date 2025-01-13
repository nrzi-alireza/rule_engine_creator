import json
import logging
from datetime import datetime

import pandas as pd
import yaml

from helper.llm import stream_llm
from helper.openai_function import openai_function
from helper.rule_engine import Action, Entity, Fact, Process, Rule, SimpleRulesEngine
from helper.workflow_engine import (
    ExecutionNode,
    InputNode,
    State,
    StreamResult,
    WorkflowEngine,
)

logger = logging.getLogger(__name__)


@openai_function
def extract_and_upsert_entity(message: str):
    """
    Extract and upsert entities from the user's and assistant's messages
    Entity is a real-world object or concept that can be observed or measured and have some attributes schema.

    Parameters
    ----------
    message
        Aggregated message from the user and assistant into a single string that contains all the messages related to the entities
    """
    pass


@openai_function
def extract_and_upsert_facts(message: str):
    """
    Extract and upsert facts from the user's and assistant's messages
    Fact is a statement about an entity or instantiation of it (its actual attributes).

    Parameters
    ----------
    message
        Aggregated message from the user and assistant into a single string that contains all the messages related to the facts and its related entity
    """
    pass


@openai_function
def extract_and_upsert_actions(message: str):
    """
    Extract and upsert actions from the user's and assistant's messages
    Action is a task or operation that can be performed in the world specifically in the context of the domain or topic.

    Parameters
    ----------
    message
        Aggregated message from the user and assistant into a single string that contains all the messages related to the action description, its inputs and output schema
    """
    pass


@openai_function
def extract_and_upsert_rules(message: str):
    """
    Extract and upsert rules from the user's and assistant's messages
    Rule is a statement that describes a relationship between an entity and an action.
    That is, when an entity has a certain attribute, an action should be performed.

    Parameters
    ----------
    message
        Aggregated message from the user and assistant into a single string that contains all the messages related to the rule description, its relation to related entity and actions
    """
    pass


@openai_function
def extract_and_upsert_processes(message: str):
    """
    Extract and upsert processes from the user's and assistant's messages
    Process is a sequence of actions (or an algorithm) that are performed in a specific order to achieve a desired outcome.

    Parameters
    ----------
    message
        Aggregated message from the user and assistant into a single,detailed and possibly long description that contains all the messages related to the process description, its relation to related entity, details of each steps and thier needed information, etc.
    """
    pass


class RuleEngineCreatorState(State):
    topic: str = ""
    messages_history: list[dict[str, str]] = []
    rule_engine: SimpleRulesEngine | None = None

    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic
        self.rule_engine = SimpleRulesEngine(topic=topic)
        self.messages_history = self.rule_engine.get_messages()


class RuleEngineCreatorAgent(WorkflowEngine):
    def __init__(self, topic: str):
        super().__init__(
            state=RuleEngineCreatorState(topic=topic),
        )
        with open("artifacts/prompts/assistant.yaml", "r") as f:
            self.assistant_prompts = yaml.safe_load(f)

        self.set_start_nodes(
            [
                ExecutionNode(
                    function=self.chat_with_user,
                    args=(self,),
                    kwargs={"system_prompt": self.create_system_prompt()},
                )
            ],
        )

    def before_step(self, external_input: str, state: State) -> None:
        state.messages_history.append({"role": "user", "content": external_input})
        state.rule_engine.insert_message(state.messages_history[-1])

    def get_messages(self) -> list[dict[str, str]]:
        return self.state.messages_history

    def get_entities(self) -> list[Entity]:
        return self.state.rule_engine.get_entities()

    def get_actions(self) -> list[Action]:
        return self.state.rule_engine.get_actions()

    def get_facts(self) -> list[Fact]:
        return self.state.rule_engine.get_facts()

    def get_rules(self) -> list[Rule]:
        return self.state.rule_engine.get_rules()

    def get_processes(self) -> list[Process]:
        return self.state.rule_engine.get_processes()

    def get_processes_df(self) -> pd.DataFrame:
        return self.state.rule_engine.get_processes_df()

    def create_system_prompt(self) -> str:
        system_prompt_template: str = self.assistant_prompts["system_prompt"]
        return system_prompt_template.format(
            # system info
            domain=self.state.topic,
            current_date_time=datetime.now().strftime("%Y/%m/%d %H:%M"),
            # schemas
            entity_schema=json.dumps(Entity.model_json_schema(), indent=4),
            fact_schema=json.dumps(Fact.model_json_schema(), indent=4),
            action_schema=json.dumps(Action.model_json_schema(), indent=4),
            rule_schema=json.dumps(Rule.model_json_schema(), indent=4),
            process_schema=json.dumps(Process.model_json_schema(), indent=4),
            # current list of each type
            entities=json.dumps([e.model_dump() for e in self.state.rule_engine.get_entities()], indent=4),
            facts=json.dumps([f.model_dump() for f in self.state.rule_engine.get_facts()], indent=4),
            actions=json.dumps([a.model_dump() for a in self.state.rule_engine.get_actions()], indent=4),
            rules=json.dumps([r.model_dump() for r in self.state.rule_engine.get_rules()], indent=4),
            processes=json.dumps([p.model_dump() for p in self.state.rule_engine.get_processes()], indent=4),
        )

    def chat_with_user(self, *args, **kwargs):
        state: RuleEngineCreatorState = kwargs["state"]
        system_prompt: str = kwargs["system_prompt"]

        llm_response = stream_llm(
            system_prompt=system_prompt,
            user_prompt=state.messages_history[-1]["content"],
            messages_history=state.messages_history[:-1],
            tools=[
                extract_and_upsert_entity.schema,
                extract_and_upsert_facts.schema,
                extract_and_upsert_actions.schema,
                extract_and_upsert_rules.schema,
                extract_and_upsert_processes.schema,
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
        if assistant_output:
            self.state.rule_engine.insert_message(state.messages_history[-1])

        if tool_calls:

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                print(f"@@@@ TOOL CALL: {function_name}({arguments})", "\n\n")

                related_messages_history = []
                for msg in self.state.messages_history[-50:]:
                    if "tool_calls" not in msg and msg["role"] != "tool":
                        related_messages_history.append(msg)

                if function_name == "extract_and_upsert_entity":
                    yield StreamResult(
                        chunk="\n\nAdding/Updating entities...this may take a while!\n\n", is_complete=False
                    )
                    function_response = self.state.rule_engine.extract_and_upsert_entity(related_messages_history)
                elif function_name == "extract_and_upsert_facts":
                    yield StreamResult(
                        chunk="\n\nAdding/Updating facts...this may take a while!\n\n", is_complete=False
                    )
                    function_response = self.state.rule_engine.extract_and_upsert_facts(related_messages_history)
                elif function_name == "extract_and_upsert_actions":
                    yield StreamResult(
                        chunk="\n\nAdding/Updating actions...this may take a while!\n\n", is_complete=False
                    )

                    function_response = self.state.rule_engine.extract_and_upsert_actions(related_messages_history)
                elif function_name == "extract_and_upsert_rules":
                    yield StreamResult(
                        chunk="\n\nAdding/Updating rules...this may take a while!\n\n", is_complete=False
                    )

                    function_response = self.state.rule_engine.extract_and_upsert_rules(related_messages_history)
                elif function_name == "extract_and_upsert_processes":
                    yield StreamResult(
                        chunk="\n\nAdding/Updating processes...this may take a while!\n\n", is_complete=False
                    )
                    function_response = self.state.rule_engine.extract_and_upsert_processes(related_messages_history)
                else:
                    yield StreamResult(chunk=f"Unknown function {function_name} was called!\n\n", is_complete=False)
                    function_response = f"Unknown function {function_name} was called! try again!"

                self.state.messages_history.append(
                    {"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(function_response)}
                )

            llm_response = stream_llm(
                system_prompt=system_prompt,
                user_prompt=f"Give a summary of the recent tool calls ralted to {tool_calls} results for the user. If there are any errors, mention them in the summary and offer suggestions to the user to fix them.",
                messages_history=state.messages_history,
            )

            for chunk in llm_response:
                if isinstance(chunk, str):
                    yield StreamResult(chunk=chunk)

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
