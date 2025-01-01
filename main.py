from typing import Dict, List

import pandas as pd
import streamlit as st

from rule_engine_creator_agent import RuleEngineCreatorAgent


def init_session_state():
    if "state_initialized" not in st.session_state:
        st.session_state.state_initialized = True
        st.session_state.topic_rule_engine_agents = dict()

        st.session_state.currect_active_topic = "Default"
        setup_topic("Default")


def setup_topic(topic: str):
    if topic not in st.session_state.topic_rule_engine_agents:
        st.session_state.topic_rule_engine_agents[topic] = RuleEngineCreatorAgent(topic=topic)
    else:
        st.session_state.currect_active_topic = topic
    st.rerun()


def get_currect_active_topic() -> str:
    return st.session_state.currect_active_topic


def get_topics() -> list[str]:
    return list(st.session_state.topic_rule_engine_agents.keys())


def get_active_topic_agent() -> RuleEngineCreatorAgent:
    return st.session_state.topic_rule_engine_agents[get_currect_active_topic()]


def add_message_to_active_topic(message: Dict):
    st.session_state.topic_messages[get_currect_active_topic()].append(message)


def get_active_topic_messages() -> List[Dict]:
    return get_active_topic_agent().get_messages()


@st.dialog("Create a new topic")
def create_new_topic():
    st.write(f"Describe the topic you want to create")
    topic = st.text_input("Topic: ", help="Small tech startups...")
    if st.button("Create"):
        setup_topic(topic)
        st.rerun()


def main():
    init_session_state()

    st.set_page_config(page_title="Rule Engine Creator", layout="wide")
    st.title("Rule Engine Creator")

    agent = get_active_topic_agent()

    # Display chat messages
    for message in get_active_topic_messages():
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # inputs
    prompt = st.chat_input("What would you like to know about creating rules?")

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent_response_generator = agent.step_until_external_input(prompt)
                response = st.write_stream(agent_response_generator)

    # Compute Fact & rule tables
    entities = agent.get_entities()
    entity_df = pd.DataFrame(
        {
            "ID": [entity.id for entity in entities],
            "Name": [entity.name for entity in entities],
            "Attributes": [entity.attributes_json_schema for entity in entities],
        }
    )

    facts = agent.get_facts()
    fact_df = pd.DataFrame(
        {
            "ID": [fact.id for fact in facts],
            "Name": [fact.name for fact in facts],
            "Entity": [entity_df.loc[entity_df["ID"] == fact.entity_id, "Name"].values[0] for fact in facts],
            "Attributes": [fact.attributes_json for fact in facts],
        }
    )

    actions = agent.get_actions()
    action_df = pd.DataFrame(
        {
            "ID": [action.id for action in actions],
            "Name": [action.name for action in actions],
            "Description": [action.description for action in actions],
            "Input": [action.input_json_schema for action in actions],
            "Output": [action.output_json_schema for action in actions],
        }
    )

    rules = agent.get_rules()
    rule_df = pd.DataFrame(
        {
            "Name": [rule.name for rule in rules],
            "Related Entities": [rule.related_entities for rule in rules],
            "Preconditions": [rule.preconditions for rule in rules],
            "Action": [rule.action for rule in rules],
        }
    )

    # Sidebar
    with st.sidebar:
        st.title("Topics")
        topic = st.selectbox(
            "Select a topic",
            get_topics(),
        )
        if topic != get_currect_active_topic():
            setup_topic(topic)

        st.button("New topic", on_click=create_new_topic)

        st.title("Entities")
        st.dataframe(
            entity_df,
            hide_index=True,
            height=300,
            use_container_width=True,
        )

        st.title("Facts")
        st.dataframe(
            fact_df,
            hide_index=True,
            height=300,
            use_container_width=True,
        )

        st.title("Actions")
        st.dataframe(
            action_df,
            hide_index=True,
            height=300,
            use_container_width=True,
        )

        st.title("Rules")
        st.dataframe(
            rule_df,
            hide_index=True,
            height=300,
            use_container_width=True,
        )

        st.divider()
        st.title("Options")
        if st.button("Clear Chat"):
            pass


if __name__ == "__main__":
    main()
