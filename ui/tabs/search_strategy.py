"""
Search Strategy Tab — auto cockpit + semi-auto chat
"""

import streamlit as st
import threading, time


def render_search_strategy(state, engine, is_auto: bool):
    if is_auto:
        _render_auto_cockpit(state, engine)
    else:
        _render_semi_auto_chat(state, engine)


# ═══════════════════════════════════════
# Auto Cockpit
# ═══════════════════════════════════════

def _render_auto_cockpit(state, engine):
    st.header("Full-Automatic Mode")

    # Config row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state.auto_target = st.number_input("Target count", 5, 200, state.auto_target, 5,
                                             help="Total qualified professors to find")
    with c2:
        state.auto_min_score = st.slider("Min score", 0, 100, state.auto_min_score, 5,
                                          help="Minimum quality score for auto-selection")
    with c3:
        state.auto_max_rounds = st.number_input("Max rounds", 3, 30, state.auto_max_rounds, 1)
    with c4:
        state.auto_stagnant_stop = st.number_input("Stagnant stop", 1, 10, state.auto_stagnant_stop, 1,
                                                    help="Stop after N rounds with no new professors")

    st.divider()

    # Strategy display
    stats = state.get_stats()
    strat = state.current_strategy
    strat_class = {"carpet_bomb": "carpet", "precision": "precision", "snowball": "snowball"}.get(strat, "")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Current strategy**: <span class='{strat_class}'>{strat.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"Round {stats['current_round']} | Found {stats['total_found']} | Qualified {stats['total_qualified']}/{state.auto_target} | Stagnant {stats['consecutive_no_new']}")

        # Progress
        if engine.is_running:
            p = engine._progress
            if p.get("total", 0) > 0:
                st.progress(p["current"] / p["total"], p.get("message", ""))
    with col2:
        # Start / Stop
        if not engine.is_running:
            if st.button("Start Auto Search", type="primary", use_container_width=True):
                state.save()
                thread = threading.Thread(target=engine.run_auto_loop, daemon=True)
                thread.start()
                st.rerun()
        else:
            if st.button("Stop", type="secondary", use_container_width=True):
                engine.stop()
                st.rerun()

    # Strategy log
    st.divider()
    st.subheader("Strategy Log")
    log_container = st.container(height=300)
    with log_container:
        for entry in reversed(state.strategy_log[-30:]):
            st.markdown(f"<div class='log-entry'>{entry}</div>", unsafe_allow_html=True)

    # Auto-refresh when running
    if engine.is_running:
        time.sleep(2)
        st.rerun()


# ═══════════════════════════════════════
# Semi-Auto Chat
# ═══════════════════════════════════════

def _render_semi_auto_chat(state, engine):
    st.header("Semi-Automatic Mode")
    st.caption("Configure search parameters through natural language discussion, then confirm to execute.")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Chat display
        for msg in st.session_state.chat_history:
            role_label = "Assistant" if msg["role"] == "ai" else "You"
            with st.chat_message("assistant" if msg["role"] == "ai" else "user"):
                st.markdown(f"**{role_label}**: {msg['content']}")

        # Initial message
        if not st.session_state.chat_history:
            from scripts.profile_parser import ProfileParser
            p = ProfileParser("profiles/my_profile.json")
            prefs = p.profile.target_preferences
            kws = p.get_research_keywords()[:5]
            initial = (
                f"Based on your profile, I suggest this search plan:\n\n"
                f"- Regions: {', '.join(prefs.locations)}\n"
                f"- Keywords: {', '.join(kws)}\n"
                f"- Ranks: {', '.join(prefs.professor_ranks)}\n"
                f"- Time range: 3 years\n\n"
                f"You can modify any parameter, e.g., 'add Switzerland to regions' or 'replace keyword X with Y'.\n"
                f"Type 'confirm' to execute the search."
            )
            st.session_state.chat_history.append({"role": "ai", "content": initial})
            st.rerun()

        # Input
        user_in = st.chat_input("Type modification or 'confirm' to execute...")
        if user_in:
            st.session_state.chat_history.append({"role": "user", "content": user_in})
            if user_in.strip().lower() in ("confirm", "go", "yes", "execute"):
                with st.spinner("Generating plan..."):
                    from scripts.profile_parser import ProfileParser
                    from scripts.llm_client import LLMClient
                    from scripts.search_strategist import generate_plan_auto
                    mods = [m["content"] for m in st.session_state.chat_history
                            if m["role"] == "user" and m["content"].strip().lower() != "confirm"]
                    plan = generate_plan_auto(ProfileParser("profiles/my_profile.json"), LLMClient(), custom_modifications=mods or None)
                    state.current_plan = plan
                    state.save()
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": f"Plan generated. Keywords: {plan.get('keywords',[])[:5]}. Regions: {plan.get('regions',[])}. Ready to execute.",
                })
                st.rerun()
            else:
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": f"Noted: '{user_in}'. Continue editing or type 'confirm' to execute.",
                })
                st.rerun()

    with col2:
        st.subheader("Current Plan")
        plan = state.current_plan
        if plan:
            st.write(f"**Regions**: {', '.join(plan.get('regions',[]))}")
            st.write(f"**Keywords** ({len(plan.get('keywords',[]))}):")
            for kw in plan.get('keywords', [])[:10]:
                st.write(f"- {kw}")
            st.write(f"**Time**: {plan.get('time_range_years',3)} years")
            st.write(f"**Per query**: {plan.get('max_results_per_query',20)}")
        else:
            st.info("No plan yet. Discuss with the assistant first.")

        st.divider()

        if plan and not engine.is_running:
            if st.button("Execute Search", type="primary", use_container_width=True):
                with st.spinner("Searching..."):
                    result = engine.start_search(plan)
                st.success(f"Round {state.current_round} done. Found {result['professors_found']} professors.")
                st.rerun()

        if st.button("Reset Discussion"):
            st.session_state.chat_history = []; state.current_plan = None; st.rerun()
