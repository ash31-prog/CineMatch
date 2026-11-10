# CineMatch — Machine Learning Based Movie Recommendation System
## Abstract
CineMatch is a content-based movie recommendation application that suggests titles similar to a movie selected by the user. It uses movie genres, plot summaries and descriptive keywords as content features. Text is converted into numerical vectors using Term Frequency–Inverse Document Frequency (TF-IDF). Recommendations are produced using cosine similarity and K-Nearest Neighbors (KNN) with cosine distance. A Streamlit interface enables users to select a title, choose an algorithm and view the top five recommendations. The project demonstrates a complete lightweight machine-learning pipeline, from data preparation to interactive inference.

## 1. Problem Statement
Movie catalogues contain many titles, making it difficult for viewers to discover films matching their interests. Manual browsing is time-consuming and simple genre filters may overlook similarities in story themes and descriptive content. CineMatch addresses this discovery problem by comparing movie metadata and ranking titles according to content similarity.

## 2. Objective
- Build a content-based movie recommendation system.
- Represent movie metadata as numerical features using TF-IDF.
- Apply cosine similarity to rank related movies.
- Implement KNN to retrieve nearest movie neighbours.
- Develop an interactive interface for live recommendations.
- Explain the strengths and limitations of the approach.

## 3. Project Description
The system uses a curated demonstration catalogue containing 40 movie records. Each record contains a title, genres, overview and keywords. The fields are combined into a single text document for each movie. The resulting feature matrix is used by both recommendation methods. The user selects a movie and the system returns up to five other titles with the highest similarity.

## 4. Proposed System
The proposed system follows a content-based filtering approach. Unlike collaborative filtering, it does not require a history of ratings from many users. It analyses item attributes directly. This makes the system suitable for a small standalone demonstration and allows recommendations for titles even when no user-rating history exists.

## 5. Module Diagram
```text
Movie Catalogue (CSV)
        |
        v
Data Loading and Cleaning
        |
        v
Metadata Combination
(genres + overview + keywords)
        |
        v
TF-IDF Feature Extraction
        |
        v
Numerical Movie Feature Matrix
        |
        +----------------------+
        |                      |
        v                      v
Cosine Similarity          KNN Retrieval
        |                      |
        +----------+-----------+
                   |
                   v
        Exclude Selected Movie
                   |
                   v
          Rank Top 5 Results
                   |
                   v
        Streamlit Recommendation UI
```

## 6. Module Description
### 6.1 Data Loading Module
Reads the movie catalogue from a CSV file using Pandas. Missing text fields are replaced with empty strings so that feature construction can proceed safely.
### 6.2 Metadata Preparation Module
Combines genres, plot overview and keywords. Combining multiple fields gives the system more context than using genre alone.
### 6.3 Feature Extraction Module
TF-IDF converts the text corpus into a sparse matrix. Frequent terms across many movies receive lower inverse-document-frequency weight, while more informative terms receive greater weight.
### 6.4 Similarity Module
Cosine similarity measures the orientation of two feature vectors. Higher values indicate more similar metadata. The selected movie is excluded from the returned list.
### 6.5 KNN Module
KNN retrieves the nearest feature vectors. The implementation uses cosine distance, where distance equals one minus cosine similarity.
### 6.6 User Interface Module
Streamlit provides a movie selector, algorithm selector, recommendation count, result cards and a genre catalogue chart.

## 7. Algorithms
### 7.1 TF-IDF
TF-IDF represents text numerically. Term Frequency measures how often a term appears in a document. Inverse Document Frequency reduces the influence of terms appearing in many documents. The vectorizer uses English stop-word removal and unigrams plus bigrams.
### 7.2 Cosine Similarity
Cosine similarity compares two vectors using the cosine of the angle between them. For non-negative TF-IDF vectors, a larger score indicates greater content similarity. The system sorts candidate movies in descending order and returns the top results.
### 7.3 K-Nearest Neighbors
KNN finds the closest movie vectors to the selected movie. The project uses cosine distance and brute-force neighbour search, which is appropriate for a small catalogue. The selected movie itself is removed from the result list.

## 8. Implementation
### 8.1 Tools and Technologies
- Python: core programming language.
- Pandas and NumPy: data handling.
- Scikit-learn: TF-IDF, cosine similarity and KNN.
- Streamlit: interactive web interface.
- Joblib: saving model artefacts.
- VS Code: development environment.
### 8.2 Dataset
The bundled demonstration catalogue contains 40 manually curated movie entries across genres such as action, adventure, animation, biography, comedy, crime, drama, fantasy, horror, mystery, romance, science fiction and thriller. Each entry includes a short synopsis and keywords. The dataset is intended for demonstrating the method, not for making claims about all films.
### 8.3 Preprocessing
Text fields are checked for missing values, combined and passed to TfidfVectorizer. English stop words are removed. Unigrams and bigrams are used to capture individual terms and short phrases.
### 8.4 Model Construction
The TF-IDF matrix is created once and cached by Streamlit. Cosine similarity can rank the full matrix, while KNN retrieves nearest neighbours from the same matrix.
### 8.5 User Interaction
The user selects a movie and an algorithm. On clicking the recommendation button, the application returns up to five similar movies and displays their genres, summaries and similarity scores.

## 9. Demonstration
During the demonstration:
1. Launch the application using `streamlit run app.py`.
2. Select a movie from the dropdown.
3. Choose Cosine Similarity or KNN in the sidebar.
4. Click “Find similar movies”.
5. Observe the ranked recommendations and similarity scores.
6. Change the selected movie and repeat.
Screenshots should be captured from the actual running application and inserted in this section.

## 10. Performance Analysis
The project is a recommendation system rather than a supervised classification model. Accuracy is therefore not an appropriate primary metric because there are no ground-truth labels indicating the correct five recommendations for every title. The report should compare:
- **Catalogue coverage:** proportion of catalogue titles for which recommendations can be generated.
- **Response time:** time required to generate recommendations.
- **Top-k similarity:** similarity scores of returned neighbours.
- **Qualitative relevance:** manually inspect whether returned titles share meaningful themes.
For a formal evaluation, create a labelled relevance set or obtain user relevance judgements, then calculate Precision@5, Recall@5 or NDCG@5. Do not report invented scores. In the demonstration catalogue, both methods operate over the same TF-IDF features; KNN uses cosine distance to retrieve neighbours, so their rankings should be closely related.

## 11. Conclusion
CineMatch demonstrates a practical content-based recommendation pipeline. It converts movie metadata into TF-IDF vectors and retrieves related titles using cosine similarity and KNN. The Streamlit interface makes the process accessible through a simple movie-selection workflow. The project shows how text processing and nearest-neighbour methods can support recommendation tasks without requiring user-rating history.

## 12. Future Enhancement
- Expand the catalogue using a properly licensed public movie dataset.
- Add user ratings and collaborative filtering.
- Combine content-based and collaborative methods in a hybrid recommender.
- Add user profiles, watchlists and feedback.
- Use validated external metadata and poster services.
- Conduct user studies and calculate Precision@K and NDCG.
- Deploy the application as a hosted web service.

## References
1. Scikit-learn documentation, `TfidfVectorizer`.
2. Scikit-learn documentation, `cosine_similarity`.
3. Scikit-learn documentation, `NearestNeighbors`.
4. Streamlit documentation.
5. Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008.
