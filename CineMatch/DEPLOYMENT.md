# Deploy CineMatch on Streamlit Community Cloud

## 1. Create a GitHub repository
1. Sign in to GitHub and create a new repository named `cinematch`.
2. Extract `CineMatch_Complete_Project.zip`.
3. Upload the contents of the inner `CineMatch` folder to the repository root.
4. Make sure `app.py`, `requirements.txt`, and the `data/` folder are at the repository root level.

Expected repository structure:
```
cinematch/
├── app.py
├── requirements.txt
├── data/
│   └── movies.csv
├── train_model.py
├── README.md
└── viva_questions.md
```

## 2. Deploy
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose the `cinematch` repository and its default branch.
4. Set the main file path to `app.py`.
5. Select **Deploy**.

Streamlit Cloud will install dependencies from `requirements.txt` and build the app.
The app does not require API keys or secrets.

## 3. Test the deployed app
- Choose a movie.
- Select Cosine Similarity or KNN.
- Click **Find similar movies**.
- Check that up to five recommendations appear.

## Important notes
- Do not upload `.venv/`, `__pycache__/`, or local Python installation files.
- Keep `data/movies.csv` in the `data` folder.
- The app uses a compact, curated 40-title demonstration dataset.
- Similarity scores describe metadata similarity, not the probability that a viewer will like a movie.
- The pinned dependency versions provide a reproducible deployment environment; Streamlit Cloud installs these in its own managed environment and does not rely on the blocked local Windows DLLs.
