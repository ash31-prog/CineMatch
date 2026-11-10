"""Train and save the CineMatch TF-IDF + KNN recommendation artefacts."""
from pathlib import Path
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).parent
df=pd.read_csv(ROOT/"data"/"movies.csv").fillna("")
df["content"]=(df["genres"].str.replace(","," ",regex=False)+" "+df["overview"]+" "+df["keywords"])
vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
X=vectorizer.fit_transform(df["content"])
knn=NearestNeighbors(metric="cosine",algorithm="brute",n_neighbors=6).fit(X)
(ROOT/"models").mkdir(exist_ok=True)
joblib.dump({"vectorizer":vectorizer,"matrix":X,"knn":knn,"titles":df["title"].tolist()},ROOT/"models"/"cinematch_model.joblib")
print(f"Saved model artefacts for {len(df)} movies.")
