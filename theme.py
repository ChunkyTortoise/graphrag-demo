"""Theme module for GraphRAG Demo."""

import streamlit as st

_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Lexend:wght@400;500;600;700&"
    "family=Work+Sans:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_FONTS_URL}" rel="stylesheet">
<style>
    /* Fonts */
    html, body, [class*="css"] {{
        font-family: 'Work Sans', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Lexend', sans-serif !important;
    }}
    code, pre, .stCode {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}

    /* Base layout */
    .stApp {{
        background-color: #040D08;
        color: #E2E8F0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Ccircle cx='30' cy='30' r='3' fill='rgba(5,150,105,0.15)'/%3E%3Ccircle cx='100' cy='60' r='4' fill='rgba(5,150,105,0.12)'/%3E%3Ccircle cx='170' cy='30' r='3' fill='rgba(5,150,105,0.15)'/%3E%3Ccircle cx='60' cy='130' r='3' fill='rgba(5,150,105,0.10)'/%3E%3Ccircle cx='150' cy='150' r='4' fill='rgba(5,150,105,0.12)'/%3E%3Ccircle cx='30' cy='170' r='3' fill='rgba(5,150,105,0.10)'/%3E%3Cline x1='30' y1='30' x2='100' y2='60' stroke='rgba(5,150,105,0.08)' stroke-width='1'/%3E%3Cline x1='100' y1='60' x2='170' y2='30' stroke='rgba(5,150,105,0.08)' stroke-width='1'/%3E%3Cline x1='100' y1='60' x2='60' y2='130' stroke='rgba(5,150,105,0.06)' stroke-width='1'/%3E%3Cline x1='60' y1='130' x2='150' y2='150' stroke='rgba(5,150,105,0.06)' stroke-width='1'/%3E%3Cline x1='30' y1='170' x2='60' y2='130' stroke='rgba(5,150,105,0.06)' stroke-width='1'/%3E%3C/svg%3E");
        background-size: 200px 200px;
    }}
    .main .block-container {{ background-color: transparent; padding-top: 2rem; }}

    /* Buttons */
    .stButton > button {{
        background-color: #059669; color: white; border: none;
        border-radius: 6px; padding: 0.5rem 1.5rem; font-weight: 600;
        transition: background-color 0.15s ease;
    }}
    .stButton > button:hover {{ background-color: #047857; }}

    /* Glass metric card */
    .metric-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 0 16px rgba(5, 150, 105, 0.25);
    }}

    /* Status boxes */
    .success-box {{
        background-color: #064e3b; border: 1px solid #10b981;
        border-radius: 8px; padding: 1rem; margin: 1rem 0;
    }}
    .info-box {{
        background-color: #071A0F; border: 1px solid #059669;
        border-radius: 8px; padding: 1rem; margin: 1rem 0;
    }}

    h1, h2, h3 {{ color: #E2E8F0; }}

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
    [data-testid="stSidebar"] {{ background-color: #061208; }}

    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}
</style>
"""


def apply_theme() -> None:
    """Inject fonts, hide chrome, and apply glassmorphism CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)
