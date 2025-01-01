import json
import logging
from datetime import datetime
from html import entities
from typing import Any

import yaml
from pydantic import BaseModel

from .db import Database
from .llm import structured_output

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    id: int
    name: str
    attributes_json_schema: dict[str, Any]


class Fact(BaseModel):
    id: int | None = None
    entity_id: int
    name: str
    attributes_json: dict[str, Any]


class Action(BaseModel):
    id: int | None = None
    name: str
    description: str
    input_json_schema: dict[str, Any]
    output_json_schema: dict[str, Any]


class Rule(BaseModel):
    id: int | None = None
    related_entities: list[str]
    preconditions: str
    action: str
    name: str


class SimpleRulesEngine:
    def __init__(self, topic: str):
        self._setup_persistence_storage(topic)
        self.topic = topic

        with open("./artifacts/prompts/rule_engine.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)

    def _setup_persistence_storage(self, topic: str):
        db_name = "_".join(topic.lower().split())
        self.db = Database(db_path=f"./artifacts/dbs/{db_name}.db")
        self.db.connect()
        self._create_tables_if_not_present()

    def _create_tables_if_not_present(self):
        # Messages
        self.db.create_table(
            table_name="Message",
            columns=[
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "role TEXT NOT NULL",
                "content TEXT NOT NULL",
            ],
        )

        # Entity
        self.db.create_table(
            table_name="Entity",
            columns=[
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "name TEXT NOT NULL",
                "attributes_json_schema TEXT NOT NULL",
            ],
        )

        # Fact
        self.db.create_table(
            table_name="Fact",
            columns=[
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "entity_id INTEGER NOT NULL",
                "name TEXT NOT NULL",
                "attributes_json TEXT NOT NULL",
                "FOREIGN KEY (entity_id) REFERENCES Entity(id)",
            ],
        )

        # Action
        self.db.create_table(
            table_name="Action",
            columns=[
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "name TEXT NOT NULL",
                "description TEXT NOT NULL",
                "input_json_schema TEXT NOT NULL",
                "output_json_schema TEXT NOT NULL",
            ],
        )

        # Rule
        self.db.create_table(
            table_name="Rule",
            columns=[
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "name TEXT NOT NULL",
                "related_entities TEXT NOT NULL",
                "preconditions TEXT NOT NULL",
                "action TEXT NOT NULL",
            ],
        )

    def get_messages(self) -> list[dict[str, str]]:
        result = []
        db_result = self.db.fetch_all("SELECT id, role, content FROM Message ORDER BY id")
        for row in db_result:
            result.append({"role": row[1], "content": row[2]})
        return result

    def insert_message(self, message: dict[str, str]) -> None:
        self.db.execute_query(
            "INSERT INTO Message (role, content) VALUES (?, ?)", (message["role"], message["content"])
        )

    def get_entities(self) -> list[Entity]:
        result = []
        db_result = self.db.fetch_all("SELECT id, name, attributes_json_schema FROM Entity")
        for row in db_result:
            result.append(
                Entity(
                    id=row[0],
                    name=row[1],
                    attributes_json_schema=json.loads(row[2]),
                )
            )
        return result

    def upsert_entity(self, entity: Entity) -> None:
        existing_entity = None
        if entity.id and entity.id > 0:
            existing_entity = self.db.fetch_all("SELECT id FROM Entity WHERE id = ?", (entity.id,))

        if existing_entity:  # update only the attributes
            self.db.execute_query(
                "UPDATE Entity SET attributes_json_schema = ? WHERE id = ?",
                (json.dumps(entity.attributes_json_schema), entity.id),
            )
        else:
            self.db.execute_query(
                "INSERT INTO Entity (name, attributes_json_schema) VALUES (?, ?)",
                (entity.name, json.dumps(entity.attributes_json_schema)),
            )

    def get_facts(self) -> list[Fact]:
        result = []
        db_result = self.db.fetch_all("SELECT id, entity_id, name, attributes_json FROM Fact")
        for row in db_result:
            result.append(
                Fact(
                    id=row[0],
                    entity_id=row[1],
                    name=row[2],
                    attributes_json=json.loads(row[3]),
                )
            )
        return result

    def upsert_fact(self, fact: Fact) -> None:
        existing_fact = None
        if fact.id:
            existing_fact = self.db.fetch_all("SELECT id FROM Fact WHERE id = ?", (fact.id,))

        if existing_fact:  # update
            self.db.execute_query(
                "UPDATE Fact SET entity_id = ?, name = ?, attributes_json = ? WHERE id = ?",
                (fact.entity_id, fact.name, json.dumps(fact.attributes_json), fact.id),
            )
        else:
            self.db.execute_query(
                "INSERT INTO Fact (entity_id, name, attributes_json) VALUES (?, ?, ?)",
                (fact.entity_id, fact.name, json.dumps(fact.attributes_json)),
            )

    def get_actions(self) -> list[Action]:
        result = []
        db_result = self.db.fetch_all("SELECT id, name, description, input_json_schema, output_json_schema FROM Action")
        for row in db_result:
            result.append(
                Action(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    input_json_schema=json.loads(row[3]),
                    output_json_schema=json.loads(row[4]),
                )
            )
        return result

    def upsert_action(self, action: Action) -> None:
        existing_action = None
        if action.id:
            existing_action = self.db.fetch_all("SELECT id FROM Action WHERE id = ?", (action.id,))

        if existing_action:  # update
            self.db.execute_query(
                "UPDATE Action SET name = ?, description = ?, input_json_schema = ?, output_json_schema = ? WHERE id = ?",
                (
                    action.name,
                    action.description,
                    json.dumps(action.input_json_schema),
                    json.dumps(action.output_json_schema),
                    action.id,
                ),
            )
        else:
            self.db.execute_query(
                "INSERT INTO Action (name, description, input_json_schema, output_json_schema) VALUES (?, ?, ?, ?)",
                (
                    action.name,
                    action.description,
                    json.dumps(action.input_json_schema),
                    json.dumps(action.output_json_schema),
                ),
            )

    def get_rules(self) -> list[Rule]:
        result = []
        db_result = self.db.fetch_all("SELECT id, related_entities, preconditions, action, name FROM Rule")
        for row in db_result:
            result.append(
                Rule(
                    id=row[0],
                    related_entities=row[1].split(","),
                    preconditions=row[2],
                    action=row[3],
                    name=row[4],
                )
            )
        return result

    def upsert_rule(self, rule: Rule) -> None:
        existing_rule = None
        if rule.id:
            existing_rule = self.db.fetch_all("SELECT id FROM Rule WHERE id = ?", (rule.id,))

        if existing_rule:  # update
            self.db.execute_query(
                "UPDATE Rule SET related_entities = ?, preconditions = ?, action = ?, name = ? WHERE id = ?",
                (
                    ",".join(rule.related_entities),
                    rule.preconditions,
                    rule.action,
                    rule.name,
                    rule.id,
                ),
            )
        else:
            self.db.execute_query(
                "INSERT INTO Rule (related_entities, preconditions, action, name) VALUES (?, ?, ?, ?)",
                (
                    ",".join(rule.related_entities),
                    rule.preconditions,
                    rule.action,
                    rule.name,
                ),
            )

    def create_system_prompt(self) -> str:
        system_prompt_template: str = self.prompts["system_prompt"]
        return system_prompt_template.format(
            # system info
            domain=self.topic,
            current_date_time=datetime.now().strftime("%Y/%m/%d %H:%M"),
            # schemas
            entity_schema=json.dumps(Entity.model_json_schema(), indent=4),
            fact_schema=json.dumps(Fact.model_json_schema(), indent=4),
            action_schema=json.dumps(Action.model_json_schema(), indent=4),
            rule_schema=json.dumps(Rule.model_json_schema(), indent=4),
            # current list of each type
            entities=json.dumps([e.model_dump() for e in self.get_entities()], indent=4),
            facts=json.dumps([f.model_dump() for f in self.get_facts()], indent=4),
            actions=json.dumps([a.model_dump() for a in self.get_actions()], indent=4),
            rules=json.dumps([r.model_dump() for r in self.get_rules()], indent=4),
        )

    def extract_and_upsert_entity(self, messages_history: list[dict[str, str]]) -> None:
        class Output(BaseModel):
            class EntityType(BaseModel):
                id: int
                name: str
                attributes_json_schema: str

            reason_to_create_new_entities: str
            entities_to_create: list[EntityType]

            reason_to_update_existing_entities: str
            entities_to_update: list[EntityType]

        instruction_prompt_template: str = self.prompts["extract_and_upsert_entities"]
        instruction_prompt = instruction_prompt_template.format(
            entities=json.dumps([e.model_dump() for e in self.get_entities()], indent=4),
        )

        llm_response: Output = structured_output(
            system_prompt=self.create_system_prompt(),
            user_prompt=instruction_prompt,
            messages_history=messages_history,
            response_format=Output,
        )

        for new_entity in llm_response.entities_to_create:
            try:
                self.upsert_entity(
                    Entity(
                        id=0,
                        name=new_entity.name,
                        attributes_json_schema=json.loads(
                            new_entity.attributes_json_schema,
                        ),
                    )
                )
            except Exception as e:
                logger.warning(f"Error inserting entity {new_entity.name} - {new_entity.attributes_json_schema}: {e}")

        for updated_entity in llm_response.entities_to_update:
            try:
                self.upsert_entity(
                    Entity(
                        id=updated_entity.id,
                        name=updated_entity.name,
                        attributes_json_schema=json.loads(
                            updated_entity.attributes_json_schema,
                        ),
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Error updating entity {updated_entity.name} - {updated_entity.attributes_json_schema}: {e}"
                )

    def extract_and_upsert_facts(self, messages_history: list[dict[str, str]]) -> None:
        class Output(BaseModel):
            class FactType(BaseModel):
                id: int
                entity_id: int
                name: str
                attributes: str

            reason_to_create_new_facts: str
            entities_to_create: list[FactType]

            reason_to_update_existing_facts: str
            entities_to_update: list[FactType]

        instruction_prompt_template: str = self.prompts["extract_and_upsert_facts"]
        instruction_prompt = instruction_prompt_template.format(
            entities=json.dumps([e.model_dump() for e in self.get_entities()], indent=4),
            facts=json.dumps([f.model_dump() for f in self.get_facts()], indent=4),
        )

        llm_response: Output = structured_output(
            system_prompt=self.create_system_prompt(),
            user_prompt=instruction_prompt,
            messages_history=messages_history,
            response_format=Output,
        )

        # TODO: check with entity schema
        for new_fact in llm_response.entities_to_create:
            try:
                self.upsert_fact(
                    Fact(
                        id=0,
                        entity_id=new_fact.entity_id,
                        name=new_fact.name,
                        attributes_json=json.loads(
                            new_fact.attributes,
                        ),
                    )
                )
            except Exception as e:
                logger.warning(f"Error inserting fact {new_fact.name} - {new_fact.attributes}: {e}")

        for updated_fact in llm_response.entities_to_update:
            try:
                self.upsert_fact(
                    Fact(
                        id=updated_fact.id,
                        entity_id=updated_fact.entity_id,
                        name=updated_fact.name,
                        attributes_json=json.loads(
                            updated_fact.attributes,
                        ),
                    )
                )
            except Exception as e:
                logger.warning(f"Error updating fact '{updated_fact.name}' - {updated_fact.attributes}: {e}")

    def extract_and_upsert_actions(self, messages_history: list[dict[str, str]]) -> None:
        class Output(BaseModel):
            class ActionType(BaseModel):
                id: int
                name: str
                description: str
                input_json_schema: str
                output_json_schema: str

            reason_to_create_new_actions: str
            actions_to_create: list[ActionType]

            reason_to_update_existing_actions: str
            actions_to_update: list[ActionType]

        instruction_prompt_template: str = self.prompts["extract_and_upsert_actions"]
        instruction_prompt = instruction_prompt_template.format(
            actions=json.dumps([a.model_dump() for a in self.get_actions()], indent=4),
        )

        llm_response: Output = structured_output(
            system_prompt=self.create_system_prompt(),
            user_prompt=instruction_prompt,
            messages_history=messages_history,
            response_format=Output,
        )

        for new_action in llm_response.actions_to_create:
            try:
                self.upsert_action(
                    Action(
                        id=0,
                        name=new_action.name,
                        description=new_action.description,
                        input_json_schema=json.loads(new_action.input_json_schema),
                        output_json_schema=json.loads(new_action.output_json_schema),
                    ),
                )
            except Exception as e:
                logger.warning(f"Error inserting action {new_action.name} - {new_action.description}: {e}")

        for updated_action in llm_response.actions_to_update:
            try:
                self.upsert_action(
                    Action(
                        id=updated_action.id,
                        name=updated_action.name,
                        description=updated_action.description,
                        input_json_schema=json.loads(updated_action.input_json_schema),
                        output_json_schema=json.loads(updated_action.output_json_schema),
                    )
                )
            except Exception as e:
                logger.warning(f"Error updating action '{updated_action.name}' - {updated_action.attributes}: {e}")

    def extract_and_upsert_rules(self, messages_history: list[dict[str, str]]) -> None:
        class Output(BaseModel):
            class RuleType(BaseModel):
                id: int
                related_entities: list[str]
                preconditions: str
                action: str
                name: str

            reason_to_create_new_rules: str
            rules_to_create: list[RuleType]

            reason_to_update_existing_rules: str
            rules_to_update: list[RuleType]

        instruction_prompt_template: str = self.prompts["extract_and_upsert_actions"]
        instruction_prompt = instruction_prompt_template.format(
            entities=json.dumps([e.model_dump() for e in self.get_entities()], indent=4),
            actions=json.dumps([a.model_dump() for a in self.get_actions()], indent=4),
            rules=json.dumps([r.model_dump() for r in self.get_rules()], indent=4),
        )

        llm_response: Output = structured_output(
            system_prompt=self.create_system_prompt(),
            user_prompt=instruction_prompt,
            messages_history=messages_history,
            response_format=Output,
        )

        for new_rule in llm_response.rules_to_create:
            try:
                self.upsert_rule(
                    Rule(
                        id=0,
                        related_entities=new_rule.related_entities,
                        preconditions=new_rule.preconditions,
                        action=new_rule.action,
                        name=new_rule.name,
                    ),
                )
            except Exception as e:
                logger.warning(
                    f"Error inserting rule {new_rule.name} - Action: {new_rule.action} - Entity: {new_rule.related_entities} - {new_rule.preconditions}: {e}"
                )

        for updated_rule in llm_response.rules_to_update:
            try:
                self.upsert_rule(
                    Rule(
                        id=updated_rule.id,
                        related_entities=updated_rule.related_entities,
                        preconditions=updated_rule.preconditions,
                        action=updated_rule.action,
                        name=updated_rule.name,
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Error updating rule '{updated_rule.name}' - Action: {updated_rule.action} - Entity: {updated_rule.related_entities} - {updated_rule.preconditions}: {e}"
                )
