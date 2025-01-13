import streamlit as st

from ui.session_data import (
    get_active_process_agent,
    get_active_process_messages,
    get_active_topic_agent,
    get_current_active_topic,
    get_current_process,
    get_topics,
    setup_executor_agent,
    setup_topic,
)

st.title("Simulator")

if get_active_process_agent():
    if not get_active_process_agent().is_started():
        agent_response_generator = get_active_process_agent().step_until_external_input(
            "Hi! Start and run the process."
        )
        with st.chat_message("assistant"):
            st.write_stream(agent_response_generator)
        get_active_process_agent().set_started()
    else:
        for message in get_active_process_messages():
            with st.chat_message(message["role"]):
                st.write(message["content"])

    prompt = st.chat_input("Your response...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent_response_generator = get_active_process_agent().step_until_external_input(prompt)
                response = st.write_stream(agent_response_generator)

with st.sidebar:
    st.title("Topics")
    topic = st.selectbox(
        "Select a topic",
        get_topics(),
    )
    if topic != get_current_active_topic():
        setup_topic(topic)

    st.title("Processes")
    processes = get_active_topic_agent().get_processes()
    selected_process = st.selectbox(
        "Select a process to simulate: ",
        [
            "?",
        ]
        + [f"{process.id}-{process.name}" for process in processes],
    )

    if selected_process and selected_process != get_current_process():
        setup_executor_agent(get_current_active_topic(), selected_process)
        st.rerun()

    reset_button = st.button("Reset Simulation")
    if reset_button and get_active_process_agent():
        get_active_process_agent().reset()
        st.rerun()
