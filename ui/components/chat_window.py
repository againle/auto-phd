"""
可复用AI聊天窗口组件
"""

import streamlit as st
from typing import List, Dict, Callable


def chat_window(
    messages: List[Dict],
    on_send: Callable[[str], str],
    placeholder: str = "输入你的问题...",
    height: int = 300,
):
    """
    显示聊天窗口。

    Args:
        messages: 消息列表 [{"role": "user"/"ai", "content": "..."}]
        on_send: 回调函数，接收用户输入，返回AI回复
        placeholder: 输入框占位符
        height: 聊天区域高度
    """
    # 聊天历史
    chat_container = st.container(height=height)
    with chat_container:
        for msg in messages:
            role = "🤖 AI" if msg["role"] == "ai" else "👤 你"
            with st.chat_message("assistant" if msg["role"] == "ai" else "user"):
                st.markdown(f"**{role}**: {msg['content']}")

    # 输入框
    user_input = st.chat_input(placeholder)
    if user_input:
        messages.append({"role": "user", "content": user_input})
        with st.spinner("思考中..."):
            response = on_send(user_input)
        messages.append({"role": "ai", "content": response})
        st.rerun()
