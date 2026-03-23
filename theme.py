"""Theme module for GraphRAG Demo."""

import streamlit as st

_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Plus+Jakarta+Sans:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_FONTS_URL}" rel="stylesheet">
<style>
    /* Fonts */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    code, pre, .stCode {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}

    /* Base layout */
    .stApp {{ background-color: #0f172a; color: #f8fafc; }}
    .main .block-container {{ background-color: #0f172a; padding-top: 2rem; }}

    /* Buttons */
    .stButton > button {{
        background-color: #6366f1; color: white; border: none;
        border-radius: 6px; padding: 0.5rem 1.5rem; font-weight: 600;
        transition: background-color 0.15s ease;
    }}
    .stButton > button:hover {{ background-color: #4f46e5; }}

    /* Glass metric card */
    .metric-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }}

    /* Status boxes */
    .success-box {{
        background-color: #064e3b; border: 1px solid #10b981;
        border-radius: 8px; padding: 1rem; margin: 1rem 0;
    }}
    .info-box {{
        background-color: #1e3a5f; border: 1px solid #6366f1;
        border-radius: 8px; padding: 1rem; margin: 1rem 0;
    }}

    h1, h2, h3 {{ color: #f8fafc; }}

    /* Graph container */
    .graph-container {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        overflow: hidden;
    }}

    /* Skeleton shimmer */
    @keyframes shimmer {{
        0% {{ background-position: -468px 0; }}
        100% {{ background-position: 468px 0; }}
    }}
    .skeleton {{
        background: linear-gradient(
            to right,
            rgba(255,255,255,0.04) 8%,
            rgba(255,255,255,0.10) 18%,
            rgba(255,255,255,0.04) 33%
        );
        background-size: 800px 104px;
        animation: shimmer 1.4s ease-in-out infinite;
        border-radius: 6px;
        height: 1rem;
        margin: 0.4rem 0;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background-color: #0a0f1e; }}

    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}
</style>
"""


def apply_theme() -> None:
    """Inject fonts, hide chrome, and apply glassmorphism CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)
