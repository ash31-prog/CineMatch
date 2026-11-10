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
    color: #292536 !important;
    background: #FFFFFF !important;
    min-height: 38px !important;
}
li[role="option"]:hover,
div[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background: #EEE7F5 !important;
    color: #4F3D68 !important;
}

.stButton > button {
    width: 100%;
    min-height: 44px !important;
    background: #75618F !important;
    color: #FFFFFF !important;
    border: 1px solid #75618F !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
    box-shadow: none !important;
}
.stButton > button p { color: #FFFFFF !important; }
.stButton > button:hover {
    background: #5F4C78 !important;
    border-color: #5F4C78 !important;
}
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E4DCEB !important;
    border-radius: 13px !important;
    padding: 15px 17px !important;
}
div[data-testid="stMetric"] * { color: #292536 !important; }
div[data-testid="stMetricLabel"] { color: #655B70 !important; }
hr { border-color: #E3DAEC !important; }
[data-testid="stCaptionContainer"] * { color: #71687B !important; }
.stRadio label, .stSlider label { color: #292536 !important; }
[data-testid="stAlert"] {
    background: #F0EAF6 !important;
    border: 1px solid #DED2E9 !important;
}
a { color: #675080 !important; }
#MainMenu, footer { visibility: hidden; }

/* Result cards use one predictable grid; no nested Streamlit columns/borders. */
.cm-results { display: grid; gap: 12px; margin: 10px 0 22px; }
.cm-card {
    display: grid;
    grid-template-columns: 82px minmax(0, 1fr);
    align-items: stretch;
    gap: 18px;
    width: 100%;
    box-sizing: border-box;
    padding: 17px 19px;
    background: #FFFFFF;
    border: 1px solid #E3DAEC;
    border-radius: 13px;
}
.cm-rank {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    border-right: 1px solid #EEE8F3;
    padding-right: 14px;
}
.cm-number {
    color: #75618F;
    font-size: 1.45rem;
    line-height: 1.2;
    font-weight: 750;
}
.cm-match {
    color: #71687B;
    font-size: .82rem;
    margin-top: 5px;
    white-space: nowrap;
}
.cm-info { min-width: 0; display: flex; flex-direction: column; justify-content: center; }
.cm-title {
    color: #292536;
    font-size: 1.04rem;
    font-weight: 700;
    line-height: 1.35;
    margin: 0 0 5px;
    overflow-wrap: anywhere;
}
.cm-genres {
    color: #81768C;
    font-size: .83rem;
    margin: 0 0 9px;
}
.cm-overview {
    color: #393344;
    font-size: .93rem;
    line-height: 1.5;
    margin: 0;
}
.cm-chart {
    background: #FFFFFF;
    border: 1px solid #E3DAEC;
    border-radius: 13px;
    padding: 18px 20px;
}
.cm-bar-row {
    display: grid;
    grid-template-columns: minmax(110px, 155px) minmax(0, 1fr) 28px;
    gap: 12px;
    align-items: center;
    margin: 11px 0;
}
.cm-bar-label {
    color: #393344;
    font-size: .86rem;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
.cm-track {
    height: 10px;
    background: #EEE8F3;
    border-radius: 20px;
    overflow: hidden;
}
.cm-fill { height: 100%; background: #9A83B4; border-radius: 20px; }
.cm-value { color: #71687B; font-size: .82rem; text-align: right; }
@media (max-width: 640px) {
    .cm-card { grid-template-columns: 65px minmax(0,1fr); gap: 12px; padding: 14px; }
    .cm-bar-row { grid-template-columns: 95px minmax(0,1fr) 22px; gap: 8px; }
}
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
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(6, len(df)),
    ).fit(matrix)
    return df, matrix, sim, knn


df, matrix, similarity, knn = load_engine()


def recommend_cosine(title, count):
    idx = df.index[df["title"] == title][0]
    ranked = sorted(
        ((i, float(score)) for i, score in enumerate(similarity[idx]) if i != idx),
        key=lambda item: item[1],
        reverse=True,
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
st.markdown(
    "<div style='font-size:.76rem;letter-spacing:3px;font-weight:750;"
    "color:#806B98;margin-bottom:8px'>CONTENT-BASED RECOMMENDATION</div>",
    unsafe_allow_html=True,
)
st.title("CineMatch")
st.markdown(
    "<p style='font-size:1.05rem;color:#625B6B !important;max-width:760px'>"
    "Find your next film through the stories, genres and themes you already enjoy."
    "</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Recommendation settings")
    method = st.radio(
        "Recommendation algorithm",
        ["Cosine Similarity", "K-Nearest Neighbors (KNN)"],
    )
    count = st.slider("Number of recommendations", 3, 5, 5)
    st.divider()
    st.markdown(f"**Movie catalogue**  \n{len(df)} titles")
    st.markdown("**Feature model**  \nTF-IDF")
    st.markdown("**Similarity metric**  \nCosine")

left, right = st.columns([3.3, 1], vertical_alignment="bottom")
with left:
    selected = st.selectbox(
        "Choose a movie",
        sorted(df["title"].tolist()),
    )
with right:
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
        cards = []
        for rank, (movie, score) in enumerate(results, 1):
            title = escape(str(movie["title"]))
            genres = escape(str(movie["genres"]))
            overview = escape(str(movie["overview"]))
            cards.append(
                "<div class='cm-card'>"
                "<div class='cm-rank'>"
                f"<div class='cm-number'>{rank:02d}</div>"
                f"<div class='cm-match'>{score * 100:.1f}% match</div>"
                "</div>"
                "<div class='cm-info'>"
                f"<div class='cm-title'>{title}</div>"
                f"<div class='cm-genres'>{genres}</div>"
                f"<p class='cm-overview'>{overview}</p>"
                "</div></div>"
            )
        st.markdown(
            "<div class='cm-results'>" + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No recommendations were found for this title.")

st.markdown("---")
st.subheader("Catalogue overview")

genre_counts = {}
for genres in df["genres"]:
    for genre in str(genres).split(","):
        genre = genre.strip()
        if genre:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

m1, m2, m3 = st.columns(3)
m1.metric("Movies", len(df))
m2.metric("Genres", len(genre_counts))
m3.metric("Algorithms", 2)

max_count = max(genre_counts.values()) if genre_counts else 1
bars = []
for genre, value in sorted(
    genre_counts.items(), key=lambda item: item[1], reverse=True
):
    width = max(3, int(value / max_count * 100))
    bars.append(
        "<div class='cm-bar-row'>"
        f"<div class='cm-bar-label'>{escape(genre)}</div>"
        "<div class='cm-track'>"
        f"<div class='cm-fill' style='width:{width}%'></div>"
        "</div>"
        f"<div class='cm-value'>{value}</div>"
        "</div>"
    )
st.markdown(
    "<div class='cm-chart'>" + "".join(bars) + "</div>",
    unsafe_allow_html=True,
)

with st.expander("How CineMatch works"):
    st.markdown("""
    1. **Metadata preparation:** combines each movie's genres, overview and keywords.
    2. **TF-IDF:** converts the text into numerical feature vectors.
    3. **Cosine Similarity:** ranks movies by similarity between their vectors.
    4. **KNN:** retrieves the nearest movie vectors using cosine distance.
    5. **Ranking:** excludes the selected movie and displays the closest matches.
    """)

