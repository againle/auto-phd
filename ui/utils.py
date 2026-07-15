"""
UI 工具函数
"""

import streamlit as st


def score_color(score: int) -> str:
    if score >= 70:
        return "#4CAF50"
    elif score >= 40:
        return "#FF9800"
    return "#f44336"


def status_icon(status: str) -> str:
    icons = {
        "pending": "⏳", "searched": "🔍", "scored": "📊",
        "selected": "✅", "rejected": "❌", "processing": "⚙️",
        "paper_read": "📖", "email_generated": "✉️",
        "email_ready": "📬", "email_sent": "📤",
    }
    return icons.get(status, "❓")
