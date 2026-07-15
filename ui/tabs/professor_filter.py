"""
Professor Filter Tab
"""

import streamlit as st


def render_professor_filter(state, engine):
    st.header("Professor Filter")

    pending = state.get_pending_professors()
    selected = state.get_selected_professors()
    rejected = [state.professors[pid] for pid in state.rejected_ids if pid in state.professors]

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pending", len(pending)); c2.metric("Selected", len(selected))
    c3.metric("Rejected", len(rejected)); c4.metric("Target", state.auto_target if state.run_mode == "full_auto" else "--")

    # Filters
    with st.expander("Filters", expanded=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            min_score = st.slider("Min score", 0, 100, 0, 5)
        with fc2:
            sort_by = st.selectbox("Sort by", ["Score", "h-index", "Papers"])

    # Batch actions
    if pending:
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if st.button("Accept all >70", use_container_width=True):
                for p in pending:
                    if p.get("_score", 0) >= 70:
                        state.select_professor(p.get("scholar_id", ""))
                state.save(); st.rerun()
        with bc2:
            if st.button("Reject all <20", use_container_width=True):
                for p in pending:
                    if p.get("_score", 0) < 20:
                        state.reject_professor(p.get("scholar_id", ""))
                state.save(); st.rerun()
        with bc3:
            if st.button("Process Selected", type="primary", use_container_width=True):
                sids = state.selected_ids[:]
                if sids:
                    with st.spinner("Processing..."):
                        result = engine.continue_processing(sids)
                    st.success(f"Done: {result['processed']} processed, {result['email_generated']} emails")
                    st.rerun()
                else:
                    st.warning("No professors selected")

    # Cards
    filtered = [p for p in pending if p.get("_score", 0) >= min_score]
    if sort_by == "Score":
        filtered.sort(key=lambda x: x.get("_score", 0), reverse=True)
    elif sort_by == "h-index":
        filtered.sort(key=lambda x: x.get("h_index", 0), reverse=True)
    else:
        filtered.sort(key=lambda x: x.get("publication_count", 0), reverse=True)

    if filtered:
        st.subheader(f"Pending ({len(filtered)})")
        COLS = 3
        for i in range(0, len(filtered), COLS):
            row = filtered[i:i + COLS]
            cols = st.columns(COLS)
            for j, prof in enumerate(row):
                with cols[j]:
                    _card(prof, state)
    else:
        st.info("No pending professors. Run a search first.")


def _card(prof, state):
    pid = prof.get("scholar_id", "")
    name = prof.get("name", "Unknown")[:28]
    inst = prof.get("institution", "Unknown")[:28]
    score = prof.get("_score", 0)
    h_idx = prof.get("h_index", 0)
    pubs = prof.get("publication_count", 0)
    topics = prof.get("research_topics", [])[:4]

    color = "#16a34a" if score >= 70 else ("#ea580c" if score >= 40 else "#dc2626")

    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(inst)
        st.markdown(f"<span style='color:{color};font-weight:bold;font-size:1.1em'>Score: {score}</span> | h={h_idx} | papers={pubs}", unsafe_allow_html=True)
        if topics:
            st.caption(" | ".join(topics[:3]))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Accept", key=f"a_{pid}", use_container_width=True):
                state.select_professor(pid); state.save(); st.rerun()
        with c2:
            if st.button("Reject", key=f"r_{pid}", use_container_width=True):
                state.reject_professor(pid); state.save(); st.rerun()

        with st.expander("Details"):
            bd = prof.get("_breakdown", {})
            st.write(f"Research: {bd.get('research_match',0)} | Publications: {bd.get('publications',0)} | Rank: {bd.get('rank',0)} | Location: {bd.get('location',0)}")
            rp = prof.get("recent_papers", [])[:3]
            if rp:
                st.write("**Recent papers:**")
                for p in rp:
                    st.write(f"- {p.get('title','?')[:60]} ({p.get('year','?')})")
