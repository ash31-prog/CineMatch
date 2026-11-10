import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "movies.csv"

st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force a consistent light palette even when the viewer's Streamlit account
# or browser is using dark mode. All text has explicit high-contrast colours.
st.markdown("""
<style>
:root {
    color-scheme: light;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #F7F4EE !important;
    color: #292536 !important;
}
[data-testid="stHeader"] {
    background: rgba(247,244,238,0.96) !important;
}
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stMain"], [data-testid="stMain"] * {
    color: #292536;
}
h1, h2, h3, h4, p, label, span, div, li, small {
    color: #292536;
}
h1 { font-size: 2.55rem !important; font-weight: 750 !important; letter-spacing: -1.2px; }
h2 { font-weight: 700 !important; }
[data-testid="stSidebar"] {
    background: #ECE6F3 !important;
    border-right: 1px solid #DCD2E8;
}
[data-testid="stSidebar"] * { color: #292536 !important; }
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #292536 !important;
    border: 1px solid #D9D1E2 !important;
    border-radius: 10px !important;
}
[data-baseweb="select"] * { color: #292536 !important; }
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: #FFFFFF !important;
}
[data-baseweb="menu"] * { color: #292536 !important; }
.stButton > button {
    background: #75618F !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 10px !important;
    min-height: 46px;
    font-weight: 650 !important;
}
.stButton > button:hover {
    background: #5F4C78 !important;
    color: #FFFFFF !important;
    border: 0 !important;
}
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E4DCEB !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
div[data-testid="stMetric"] * { color: #292536 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #E4DCEB !important;
    border-radius: 14px !important;
}
hr { border-color: #DED6E6 !important; }
[data-testid="stCaptionContainer"] * { color: #655D70 !important; }
.stRadio label, .stSlider label { color: #292536 !important; }
[data-testid="stAlert"] * { color: #292536 !important; }
a { color: #675080 !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    df = pd.read_csv(DATA).fillna("")
    df["content"] = (
        df["genres"].str.replace(",", " ", regex=False)
        + " " + df["overview"] + " " + df["keywords"]
    )
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["content"])
    sim = cosine_similarity(matrix)
    knn = NearestNeighbors(
        metric="cosine", algorithm="brute",
        n_neighbors=min(6, len(df))
    ).fit(matrix)
    return df, matrix, sim, knn

df, matrix, similarity, knn = load_engine()

def recommend_cosine(title, count):
    idx = df.index[df["title"] == title][0]
    ranked = sorted(
        ((i, float(score)) for i, score in enumerate(similarity[idx]) if i != idx),
        key=lambda item: item[1], reverse=True
    )[:count]
    return [(df.iloc[i], score) for i, score in ranked]

def recommend_knn(title, count):
    idx = df.index[df["title"] == title][0]
    distances, indices = knn.kneighbors(
        matrix[idx], n_neighbors=min(count + 1, len(df))
    )
    result = []
    for i, distance in zip(indices[0], distances[0]):
        if i != idx:
            result.append((df.iloc[i], float(1 - distance)))
        if len(result) >= count:
            break
    return result

# Header
st.markdown("<div style='font-size:0.78rem;letter-spacing:3px;font-weight:700;color:#806B98;margin-bottom:8px'>CONTENT-BASED RECOMMENDATION</div>", unsafe_allow_html=True)
st.title("CineMatch")
st.markdown(
    "<p style='font-size:1.08rem;color:#625B6B !important;max-width:760px'>"
    "Find your next film through the stories, genres and themes you already enjoy."
    "</p>", unsafe_allow_html=True
)
st.write("")

with st.sidebar:
    st.markdown("## Recommendation settings")
    method = st.radio(
        "Recommendation algorithm",
        ["Cosine Similarity", "K-Nearest Neighbors (KNN)"]
    )
    count = st.slider("Number of recommendations", 3, 5, 5)
    st.divider()
    st.markdown(f"**Movie catalogue**  \n{len(df)} titles")
    st.markdown("**Feature model**  \nTF-IDF")
    st.markdown("**Similarity metric**  \nCosine")

left, right = st.columns([3, 1])
with left:
    selected = st.selectbox("Choose a movie", sorted(df["title"].tolist()))
with right:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = st.button("Find similar movies", use_container_width=True)

if run:
    results = (
        recommend_cosine(selected, count)
        if method == "Cosine Similarity"
        else recommend_knn(selected, count)
    )
    st.markdown("---")
    st.subheader(f"Recommendations similar to {selected}")
    if results:
        for rank, (movie, score) in enumerate(results, 1):
            with st.container(border=True):
                c1, c2 = st.columns([0.7, 5])
                with c1:
                    st.markdown(
                        f"<div style='font-size:1.35rem;font-weight:750;color:#75618F'>"
                        f"{rank:02d}</div>", unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='font-size:0.85rem;color:#625B6B'>{score*100:.1f}% match</div>",
                        unsafe_allow_html=True
                    )
                with c2:
                    st.markdown(f"**{movie['title']}**")
                    st.caption(movie["genres"])
                    st.write(movie["overview"])
    else:
        st.info("No recommendations were found for this title.")

st.markdown("---")
st.subheader("Catalogue overview")
genre_counts = {}
for genres in df["genres"]:
    for genre in genres.split(","):
        genre = genre.strip()
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

m1, m2, m3 = st.columns(3)
m1.metric("Movies", len(df))
m2.metric("Genres", len(genre_counts))
m3.metric("Algorithms", 2)

# Draw a light-theme chart using HTML bars, avoiding theme-dependent dark charts.
max_count = max(genre_counts.values()) if genre_counts else 1
bars = []
for genre, value in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
    width = max(3, int(value / max_count * 100))
    bars.append(
        f"<div style='display:grid;grid-template-columns:120px 1fr 32px;gap:12px;"
        f"align-items:center;margin:10px 0'>"
        f"<span style='font-size:0.9rem;color:#393244'>{genre}</span>"
        f"<div style='height:11px;background:#EEE8F3;border-radius:20px;overflow:hidden'>"
        f"<div style='height:11px;width:{width}%;background:#9A83B4;border-radius:20px'></div></div>"
        f"<span style='font-size:0.85rem;color:#625B6B;text-align:right'>{value}</span></div>"
    )
st.markdown(
    "<div style='background:#FFFFFF;border:1px solid #E4DCEB;border-radius:14px;"
    "padding:18px 20px'>" + "".join(bars) + "</div>",
    unsafe_allow_html=True
)
