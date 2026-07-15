"""
PhD Application Assistant — Streamlit UI
"""

import streamlit as st
from src.state_manager import GlobalState, SearchStrategy
from src.agent_engine import AgentEngine

st.set_page_config(page_title="PhD Application Assistant", page_icon="a", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
    .stProgress > div > div { background-color: #2563eb; }
    .log-entry { font-family:monospace; font-size:0.85em; padding:2px 0; border-bottom:1px solid #f1f5f9; }
    .carpet { color:#f59e0b; font-weight:600; }
    .precision { color:#2563eb; font-weight:600; }
    .snowball { color:#10b981; font-weight:600; }
    .stop-reason { background:#fef2f2; border-left:3px solid #ef4444; padding:8px 12px; margin:4px 0; }
</style>""", unsafe_allow_html=True)

if "state" not in st.session_state:
    st.session_state.state = GlobalState.load()
if "engine" not in st.session_state:
    st.session_state.engine = AgentEngine(st.session_state.state)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

state = st.session_state.state
engine = st.session_state.engine

# ── Sidebar ──
with st.sidebar:
    st.title("Application Assistant")

    mode = st.radio("Mode", ["Semi-Automatic", "Full-Automatic"],
                    index=0 if state.run_mode == "semi_auto" else 1)
    state.run_mode = "semi_auto" if "Semi" in mode else "full_auto"
    is_auto = state.run_mode == "full_auto"

    st.divider()
    stats = state.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("Found", stats["total_found"]); c2.metric("Qualified", stats["total_qualified"])
    c3, c4 = st.columns(2)
    c3.metric("Selected", stats["total_selected"]); c4.metric("Emails", stats["emails_generated"])

    if is_auto:
        st.divider()
        pct = min(stats["total_qualified"] / max(state.auto_target, 1) * 100, 100)
        st.progress(pct / 100, f"{pct:.0f}% ({stats['total_qualified']}/{state.auto_target})")
        st.caption(f"Round {stats['current_round']} | {stats['strategy']} | ${stats['api_cost']}")

    st.divider()
    if st.button("Save"): state.save(); st.success("Saved")

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["Search Strategy", "Professor Filter", "Email Review"])

with tab1:
    from ui.tabs.search_strategy import render_search_strategy
    render_search_strategy(state, engine, is_auto)
with tab2:
    from ui.tabs.professor_filter import render_professor_filter
    render_professor_filter(state, engine)
with tab3:
    from ui.tabs.email_review import render_email_review
    render_email_review(state, engine)
