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
    /* Warm-editorial tokens (single source of truth).
       One accent (green), spent twice: active state + graph node highlight.
       Radius: one 8px token. No glass, no glow. */
    :root {{
        --bg: #0E1310;          /* warm near-black canvas */
        --sidebar: #0B0F0C;
        --card: #141A16;        /* flat card, no blur */
        --border: #232C26;
        --accent: #2F9E6E;      /* deepened green, off the tailwind default */
        --accent-hover: #268A5F;
        --text: #ECEAE3;        /* warm off-white */
        --text-muted: #9AA39C;
        --chip: #ECEAE3;        /* inverted primary button face */
        --radius: 8px;
    }}

    /* Fonts */
    html, body, [class*="css"] {{
        font-family: 'Work Sans', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Lexend', sans-serif !important;
        color: var(--text);
        letter-spacing: -0.02em;
    }}
    /* Confident display scale (D5) */
    h1 {{ font-size: 3.5rem !important; line-height: 1.0 !important; font-weight: 700 !important; }}
    code, pre, .stCode {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}

    /* Base layout: warm near-black, subtle graph-node signature pattern in accent */
    .stApp {{
        background-color: var(--bg);
        color: var(--text);
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Ccircle cx='30' cy='30' r='3' fill='rgba(47,158,110,0.12)'/%3E%3Ccircle cx='100' cy='60' r='4' fill='rgba(47,158,110,0.10)'/%3E%3Ccircle cx='170' cy='30' r='3' fill='rgba(47,158,110,0.12)'/%3E%3Ccircle cx='60' cy='130' r='3' fill='rgba(47,158,110,0.08)'/%3E%3Ccircle cx='150' cy='150' r='4' fill='rgba(47,158,110,0.10)'/%3E%3Ccircle cx='30' cy='170' r='3' fill='rgba(47,158,110,0.08)'/%3E%3Cline x1='30' y1='30' x2='100' y2='60' stroke='rgba(47,158,110,0.06)' stroke-width='1'/%3E%3Cline x1='100' y1='60' x2='170' y2='30' stroke='rgba(47,158,110,0.06)' stroke-width='1'/%3E%3Cline x1='100' y1='60' x2='60' y2='130' stroke='rgba(47,158,110,0.05)' stroke-width='1'/%3E%3Cline x1='60' y1='130' x2='150' y2='150' stroke='rgba(47,158,110,0.05)' stroke-width='1'/%3E%3Cline x1='30' y1='170' x2='60' y2='130' stroke='rgba(47,158,110,0.05)' stroke-width='1'/%3E%3C/svg%3E");
        background-size: 200px 200px;
    }}
    .main .block-container {{ background-color: transparent; padding-top: 2rem; }}

    /* Primary action: inverted light chip (accent is spent on the graph) */
    .stButton > button {{
        background-color: var(--chip); color: var(--bg); border: none;
        border-radius: var(--radius); padding: 0.5rem 1.5rem; font-weight: 600;
        transition: background-color 0.15s ease;
    }}
    .stButton > button:hover {{ background-color: #FFFFFF; }}

    /* Flat metric card: no glass, no glow */
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem;
        text-align: center;
    }}

    /* Status boxes */
    .success-box {{
        background-color: var(--card); border: 1px solid var(--accent);
        border-radius: var(--radius); padding: 1rem; margin: 1rem 0;
    }}
    .info-box {{
        background-color: var(--card); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 1rem; margin: 1rem 0;
    }}

    /* Graph container */
    .graph-container {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }}

    /* Skeleton shimmer (functional loading state) */
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
        border-radius: var(--radius);
        height: 1rem;
        margin: 0.4rem 0;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background-color: var(--sidebar); }}

    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}
</style>
"""


def apply_theme() -> None:
    """Inject fonts, hide chrome, and apply the warm-editorial theme."""
    st.markdown(_CSS, unsafe_allow_html=True)
