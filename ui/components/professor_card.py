"""
可复用教授卡片组件
"""

import streamlit as st
from typing import Dict, Optional


def professor_card(professor: Dict, key: str = "") -> Optional[Dict]:
    """
    渲染教授卡片，返回用户操作。

    Args:
        professor: 教授数据字典
        key: 唯一键（用于 Streamlit widget key）

    Returns:
        {"action": "accept"/"reject"/"discuss", "professor_id": "..."} 或 None
    """
    pid = professor.get("scholar_id", key)
    name = professor.get("name", "Unknown")[:28]
    inst = professor.get("institution", "Unknown")[:28]
    score = professor.get("_score", 0)
    h_idx = professor.get("h_index", 0)
    pubs = professor.get("publication_count", 0)
    topics = professor.get("research_topics", [])[:4]

    # 分数颜色
    if score >= 70:
        color = "#4CAF50"
    elif score >= 40:
        color = "#FF9800"
    else:
        color = "#f44336"

    # 使用 container 构建卡片
    with st.container(border=True):
        st.markdown(f"**👨‍🏫 {name}**")
        st.caption(f"🏛️ {inst}")
        st.markdown(f"<span style='color:{color};font-size:1.1em;font-weight:bold'>⭐ {score}分</span> | h={h_idx} | 📄{pubs}篇", unsafe_allow_html=True)

        if topics:
            st.caption(" | ".join(topics[:3]))

        c1, c2, c3 = st.columns(3)
        result = None

        with c1:
            if st.button("✅ 接受", key=f"acc_{pid}", use_container_width=True):
                result = {"action": "accept", "professor_id": pid}
        with c2:
            if st.button("❌ 拒绝", key=f"rej_{pid}", use_container_width=True):
                result = {"action": "reject", "professor_id": pid}
        with c3:
            if st.button("💬 讨论", key=f"disc_{pid}", use_container_width=True):
                result = {"action": "discuss", "professor_id": pid}

        return result
