import streamlit as st

from ui.session_data import init_session_state

init_session_state()
st.set_page_config(page_title="Rule Engine", layout="wide")

creator_page = st.Page("ui/creator_page.py", title="Model Creator", icon="🧩")
simulator_page = st.Page("ui/simulator_page.py", title="Process Simulator", icon="🔍")

pg = st.navigation(
    [
        creator_page,
        simulator_page,
    ],
    position="sidebar",
)

pg.run()
