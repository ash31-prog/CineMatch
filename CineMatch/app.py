import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from html import escape

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "movies.csv"

st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent light lavender theme, including Streamlit's dropdown portal.
st.markdown("""
<style>
:root { color-scheme: light; }
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: #F8F6FB !important;
    color: #292536 !important;
}
[data-testid="stHeader"] { background: #F8F6FB !important; }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stMain"] * { color: #292536; }

h1 {
    color: #292536 !important;
    font-size: 2.65rem !important;
    font-weight: 760 !important;
    letter-spacing: -1.3px !important;
    margin-bottom: .25rem !important;
}
h2, h3, h4 { color: #292536 !important; font-weight: 700 !important; }
p, label, li { color: #393344; }
[data-testid="stSidebar"] {
    background: #F0EAF6 !important;
    border-right: 1px solid #E2D8EC !important;
}
[data-testid="stSidebar"] * { color: #292536 !important; }

[data-testid="stSelectbox"] label {
    color: #393344 !important;
    font-weight: 600 !important;
    margin-bottom: 7px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #292536 !important;
    border: 1px solid #D9CDE6 !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #292536 !important;
    fill: #292536 !important;
}

/* Streamlit renders the options in a portal outside the main app container.
   These selectors force that portal and every option to remain readable. */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] [data-baseweb="menu"],
div[data-baseweb="menu"],
ul[role="listbox"],
div[role="listbox"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #292536 !important;
    border: 1px solid #E2D8EC !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 28px rgba(43, 31, 59, .14) !important;
    z-index: 999999 !important;
    max-height: 300px !important;
    overflow-y: auto !important;
}
div[data-baseweb="popover"] *,
div[data-baseweb="menu"] *,
ul[role="listbox"] *,
div[role="listbox"] * {
    color: #292536 !important;
    background-color: transparent !important;
}
li[role="option"],
div[role="option"],
[data-baseweb="menu"] li {
