"""
Email Review Tab
"""

import streamlit as st
from pathlib import Path
import re


def render_email_review(state, engine):
    st.header("Email Review")

    ready = [(pid, prof) for pid, prof in state.professors.items()
             if prof.get("_status") in ("email_generated", "email_ready")]

    if not ready:
        st.info("No emails ready for review. Process professors first in the Professor Filter tab.")
        return

    st.write(f"{len(ready)} professor(s) with draft emails")

    for pid, prof in ready:
        name = prof.get("name", "Unknown")
        inst = prof.get("institution", "?")
        score = prof.get("_score", 0)

        with st.expander(f"{name} ({inst}) — Score: {score}", expanded=False):
            # Find draft
            from scripts.utils import safe_filename
            fname = safe_filename(f"{inst}_{name}")[:80]
            draft_dir = Path("professors") / fname / "drafts"

            if draft_dir.exists():
                drafts = sorted(draft_dir.glob("v*_*.md"))
                if drafts:
                    chosen = st.selectbox("Version", [d.stem for d in drafts], key=f"v_{pid}")
                    cp = draft_dir / f"{chosen}.md"
                    if cp.exists():
                        content = cp.read_text(encoding="utf-8")
                        subj, body = _parse(content)

                        st.text_input("Subject", value=subj, key=f"s_{pid}")
                        new_body = st.text_area("Body", value=body, height=220, key=f"b_{pid}")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("Save Edits", key=f"sv_{pid}"):
                                cp.write_text(f"**Subject**: {subj}\n\n{new_body}", encoding="utf-8")
                                st.success("Saved")
                        with c2:
                            if st.button("AI Polish", key=f"ai_{pid}"):
                                with st.spinner("Polishing..."):
                                    try:
                                        prompt = f"Polish this cold email for clarity and professionalism. Return only the body:\n\n{new_body}"
                                        opt = engine.llm.call(
                                            messages=[{"role": "user", "content": prompt}],
                                            task_type="email_generation",
                                        )
                                        st.text_area("Polished", value=opt, height=220, key=f"po_{pid}")
                                    except Exception as e:
                                        st.error(str(e))
                        with c3:
                            if st.button("Approve", key=f"ap_{pid}", type="primary"):
                                from src.state_manager import ProfessorStatus
                                state.update_status(pid, ProfessorStatus.EMAIL_READY)
                                state.save()
                                st.success("Approved")
                                st.rerun()

    # Batch
    st.divider()
    if st.button("Approve All (>80 score)"):
        from src.state_manager import ProfessorStatus
        for pid, prof in ready:
            if prof.get("_score", 0) >= 80:
                state.update_status(pid, ProfessorStatus.EMAIL_READY)
        state.save()
        st.success("Batch approved")
        st.rerun()


def _parse(content: str):
    m = re.search(r'(?:\*\*)?Subject(?:\*\*)?:\s*(.+?)(?:\n|$)', content)
    subj = m.group(1).strip() if m else ""
    parts = content.split("---")
    body = parts[2].strip() if len(parts) >= 3 else (parts[1].strip() if len(parts) == 2 else content)
    body = re.sub(r'\n---\n\*.*?\*$', '', body, flags=re.DOTALL).strip()
    return subj, body
