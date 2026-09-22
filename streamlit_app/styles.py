"""Custom modern styling, animations, and glassmorphism theme for Streamlit UI."""

import logging
import streamlit as st

log = logging.getLogger(__name__)

CUSTOM_CSS = """
<style>
/* ==========================================================================
   Modern Dark Theme & Glassmorphism Core Design
   ========================================================================== */

:root {
    --bg-primary: #090d16;
    --bg-card: rgba(18, 24, 38, 0.75);
    --bg-card-hover: rgba(28, 36, 56, 0.85);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(99, 102, 241, 0.35);
    --accent-indigo: #6366f1;
    --accent-cyan: #38bdf8;
    --accent-emerald: #10b981;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

/* Base App Overrides */
.stApp {
    background: radial-gradient(circle at 50% 0%, #151b2e 0%, #080c14 70%);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Custom Header / Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 69, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #38bdf8, #818cf8);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 6px;
    margin-bottom: 16px;
    line-height: 1.5;
}

/* Badges and Model Pills */
.badge-group {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.04);
    color: #e2e8f0;
}

.badge-gemini {
    border-color: rgba(56, 189, 248, 0.35);
    background: rgba(56, 189, 248, 0.08);
    color: #7dd3fc;
}

.badge-groq {
    border-color: rgba(249, 115, 22, 0.35);
    background: rgba(249, 115, 22, 0.08);
    color: #fdba74;
}

.badge-faiss {
    border-color: rgba(16, 185, 129, 0.35);
    background: rgba(16, 185, 129, 0.08);
    color: #6ee7b7;
}

/* Stat Cards */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-glow);
}

.stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Quick Prompt Chip Buttons */
.prompt-chips-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Custom Message Card Enhancements */
.stChatMessage {
    background: rgba(18, 24, 38, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(10px);
}

/* Reasoning Container */
.reasoning-box {
    background: rgba(99, 102, 241, 0.06);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 0.9rem;
    color: #cbd5e1;
}

/* Source File Pill */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    margin: 3px 4px 3px 0;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.8rem;
    color: #93c5fd;
}

/* Sidebar Customizations */
section[data-testid="stSidebar"] {
    background: #0d121f !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Streamlit Chat Input Enhancement */
.stChatInput {
    padding-bottom: 8px !important;
}

.stChatInput > div {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: rgba(18, 24, 38, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.2s ease !important;
}

.stChatInput > div:focus-within {
    border-color: rgba(99, 102, 241, 0.6) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25), 0 8px 32px rgba(0, 0, 0, 0.4) !important;
}

.stChatInput textarea {
    min-height: 60px !important;
    padding-top: 14px !important;
    padding-bottom: 14px !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    color: var(--text-primary) !important;
}

/* Button Stylings */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}
</style>
"""


def apply_custom_styles():
    """Inject custom CSS into Streamlit page."""
    try:
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    except Exception as e:
        log.warning(f"Error applying custom CSS: {e}")
