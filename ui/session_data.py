import glob
import os
from typing import Dict, List

import streamlit as st

from agents.process_executor_agent import ProcessExecutorAgent
from agents.rule_engine_creator_agent import RuleEngineCreatorAgent


def load_all_topics():
    # Get all .db files in artifacts/dbs directory
    db_files = glob.glob("./artifacts/dbs/*.db")

    # Extract filenames without path and extension, replace _ with space
    topics = [os.path.basename(f)[:-3].replace("_", " ") for f in db_files]
    return topics


def init_session_state():
    if "state_initialized" not in st.session_state:
        st.session_state.state_initialized = True
        st.session_state.topic_rule_engine_agents = dict()
        st.session_state.topic_process_executor_agents = dict()

        st.session_state.current_active_process_id = None
        st.session_state.currect_active_process = None

        topics = load_all_topics()
        if len(topics) > 0:
            for topic in topics:
                setup_topic(topic, rerun=False)
        else:
            st.session_state.currect_active_topic = "Default"
            setup_topic("Default")

        st.rerun()


def setup_topic(topic: str, rerun: bool = True):
    if topic not in st.session_state.topic_rule_engine_agents:
        st.session_state.topic_rule_engine_agents[topic] = RuleEngineCreatorAgent(topic=topic)

    st.session_state.currect_active_topic = topic
    if rerun:
        st.rerun()


def setup_executor_agent(topic: str, process: str, rerun: bool = True):
    process_id = process.split("-")[0]

    if process == "?":
        st.session_state.currect_active_process = None
        st.session_state.current_active_process_id = "?"
        if rerun:
            st.rerun()
        return

    if (topic, process_id) not in st.session_state.topic_process_executor_agents:
        st.session_state.topic_process_executor_agents[(topic, process_id)] = ProcessExecutorAgent(
            topic=topic, process_id=process_id
        )

    st.session_state.current_active_process_id = process
    st.session_state.currect_active_process = (topic, process_id)
    if rerun:
        st.rerun()


def get_current_active_topic() -> str:
    return st.session_state.currect_active_topic


def get_topics() -> list[str]:
    return list(st.session_state.topic_rule_engine_agents.keys())


def get_active_topic_agent() -> RuleEngineCreatorAgent:
    return st.session_state.topic_rule_engine_agents[get_current_active_topic()]


def add_message_to_active_topic(message: Dict):
    st.session_state.topic_messages[get_current_active_topic()].append(message)


def get_current_active_process() -> tuple:
    return st.session_state.currect_active_process


def get_current_process() -> str:
    return st.session_state.current_active_process_id


def get_active_process_agent() -> ProcessExecutorAgent:
    if get_current_active_process():
        return st.session_state.topic_process_executor_agents[get_current_active_process()]

    return None


def get_active_topic_messages() -> List[Dict]:
    return get_active_topic_agent().get_messages()


def get_active_process_messages() -> List[Dict]:
    if get_active_process_agent():
        return get_active_process_agent().get_messages()
    return []
