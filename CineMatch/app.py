import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "movies.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#fbf7ef 0%,#f2edf8 100%);}
.block-container {padding-top:2rem; max-width:1150px;}
h1,h2,h3 {color:#342b43;}
div[data-testid="stMetric"] {background:#ffffffb8;padding:16px;border-radius:14px;border:1px solid #e7deef;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    df = pd.read_csv(DATA).fillna("")
    df["content"] = (df["genres"].str.replace(",", " ", regex=False)
                     + " " + df["overview"] + " " + df["keywords"])
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    matrix = tfidf.fit_transform(df["content"])
    sim = cosine_similarity(matrix)
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(6,len(df)))
    knn.fit(matrix)
    return df, tfidf, matrix, sim, knn

df, vectorizer, matrix, similarity, knn = load_engine()

def recommend_cosine(title, n=5):
    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted((x for x in scores if x[0] != idx), key=lambda x:x[1], reverse=True)[:n]
    return [(df.iloc[i], float(score)) for i,score in scores]

def recommend_knn(title, n=5):
    idx = df.index[df["title"] == title][0]
    distances, indices = knn.kneighbors(matrix[idx], n_neighbors=min(n+1,len(df)))
    results=[]
    for i,d in zip(indices[0], distances[0]):
        if i != idx:
            results.append((df.iloc[i], float(1-d)))
        if len(results)==n: break
    return results

st.title("🎬 CineMatch")
st.caption("Discover your next favourite movie through content-based machine learning.")
st.write("Choose a movie you already enjoy. CineMatch analyses its genres, plot description and keywords to recommend similar titles.")

with st.sidebar:
    st.header("⚙️ Recommendation settings")
    method = st.radio("Recommendation algorithm", ["Cosine Similarity", "K-Nearest Neighbors (KNN)"])
    count = st.slider("Number of recommendations", 3, 5, 5)
    st.divider()
    st.caption(f"Movie catalogue: {len(df)} titles")
    st.caption("Models: TF-IDF + Cosine Similarity / KNN")

col1, col2 = st.columns([2,1])
with col1:
    selected = st.selectbox("Select a movie", sorted(df["title"].tolist()))
with col2:
    st.write("")
    st.write("")
    run = st.button("✨ Find similar movies", type="primary", use_container_width=True)

if run:
    results = recommend_cosine(selected,count) if method=="Cosine Similarity" else recommend_knn(selected,count)
    st.subheader(f"Because you liked *{selected}*...")
    if not results:
        st.info("No recommendations available.")
    else:
        for rank,(movie,score) in enumerate(results,1):
            with st.container(border=True):
                a,b = st.columns([1,5])
                with a:
                    st.markdown(f"### #{rank}")
                    st.metric("Similarity", f"{score*100:.1f}%")
                with b:
                    st.markdown(f"**{movie['title']}**")
                    st.caption(movie["genres"])
                    st.write(movie["overview"])

st.divider()
st.subheader("📊 Catalogue insights")
c1,c2,c3 = st.columns(3)
c1.metric("Movies",len(df))
c2.metric("Genres",len(set(g.strip() for row in df.genres for g in row.split(","))))
c3.metric("Algorithms",2)
genre_counts = {}
for row in df.genres:
    for genre in row.replace(",", " ").split():
        genre_counts[genre.strip()] = genre_counts.get(genre.strip(),0)+1
chart = pd.DataFrame({"Genre":list(genre_counts.keys()),"Movie count":list(genre_counts.values())}).set_index("Genre")
st.bar_chart(chart)
st.caption("Similarity scores are text-vector similarity scores, not audience ratings or probabilities of liking a movie.")

with st.expander("How does CineMatch work?"):
    st.markdown("""
    1. **Content preparation:** combines each movie's genres, overview and keywords.
    2. **TF-IDF:** converts text into numerical feature vectors and gives more weight to informative terms.
    3. **Cosine Similarity:** measures the angle-based similarity between movie vectors.
    4. **KNN:** retrieves the nearest movie vectors using cosine distance.
    5. **Ranking:** excludes the selected movie and returns the top five neighbours.
    """)
