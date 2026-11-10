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
