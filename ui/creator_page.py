import pandas as pd
import streamlit as st

from agents.rule_engine_creator_agent import RuleEngineCreatorAgent
from ui.session_data import (
    get_active_topic_agent,
    get_active_topic_messages,
    get_current_active_topic,
    get_topics,
    setup_topic,
)


@st.dialog("Create a new topic")
def create_new_topic():
    st.write(f"Describe the topic you want to create")
    topic = st.text_input("Topic: ", help="Small tech startups...")
    if st.button("Create"):
        setup_topic(topic)
        st.rerun()


def get_dataframes(agent: RuleEngineCreatorAgent):
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

    process_df = agent.get_processes_df()
    return entity_df, fact_df, action_df, rule_df, process_df


st.title("Rule Engine Creator")
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
            agent_response_generator = get_active_topic_agent().step_until_external_input(prompt)
            response = st.write_stream(agent_response_generator)
# Sidebar
with st.sidebar:
    st.title("Topics")
    topic = st.selectbox(
        "Select a topic",
        get_topics(),
    )
    if topic != get_current_active_topic():
        setup_topic(topic)

    st.button("New topic", on_click=create_new_topic)

    entity_df, fact_df, action_df, rule_df, process_df = get_dataframes(get_active_topic_agent())

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

    st.title("Processes")
    st.dataframe(
        process_df,
        hide_index=True,
        height=300,
        use_container_width=True,
    )
