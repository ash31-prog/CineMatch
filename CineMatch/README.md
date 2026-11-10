# CineMatch — Movie Recommendation System

## Overview
CineMatch is a content-based movie recommender. It recommends movies whose genres, plot summaries and keywords resemble a selected movie. It demonstrates TF-IDF feature extraction, cosine similarity and K-Nearest Neighbors (KNN).

## Run locally
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Create and activate a virtual environment (optional but recommended):
   - Windows: `py -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Start the app: `streamlit run app.py`
6. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Project files
- `app.py` — Streamlit interface and recommendation logic.
- `data/movies.csv` — small curated demonstration catalogue (40 movies).
- `train_model.py` — standalone TF-IDF + KNN artefact training script.
- `report_outline.md` — report content plan and methodology.
- `viva_questions.md` — sample viva questions and answers.

## Algorithms
- TF-IDF: transforms textual movie metadata into sparse numerical vectors.
- Cosine similarity: ranks movies by vector similarity.
- KNN: retrieves nearest movie vectors using cosine distance.

Note: Cosine similarity and KNN are related but distinct. KNN is the neighbour-retrieval algorithm; cosine similarity defines the distance/similarity measure.

## Limitations
The included 40-title catalogue is a compact demonstration dataset, not a production-scale or independently validated benchmark. Similarity scores indicate metadata similarity, not the probability that a user will enjoy a title. The app does not fetch posters or external data and does not learn individual user ratings.

## Future work
Expand the catalogue, use a licensed public dataset, add user ratings and collaborative filtering, incorporate poster/metadata APIs, and evaluate recommendations with human relevance judgements or held-out interactions.
