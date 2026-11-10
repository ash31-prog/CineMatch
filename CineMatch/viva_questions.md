# CineMatch — Viva Questions and Answers

1. **What is a recommendation system?**  
   A system that suggests relevant items to users based on item information, user preferences or interactions.

2. **Which recommendation approach does CineMatch use?**  
   Content-based filtering. It compares movie attributes such as genres, overview and keywords.

3. **What is TF-IDF?**  
   A text-weighting method that represents documents numerically. It increases the weight of informative terms and reduces the weight of common terms.

4. **Why is text converted into vectors?**  
   Machine-learning similarity algorithms operate on numerical representations rather than raw text.

5. **What is cosine similarity?**  
   A measure of similarity based on the cosine of the angle between two vectors. Higher values mean more similar vector directions.

6. **What is KNN?**  
   K-Nearest Neighbors retrieves the K closest data points to a query point according to a chosen distance metric.

7. **How are KNN and cosine similarity related here?**  
   KNN is the neighbour-search method; cosine distance is the metric used to determine which movie vectors are nearest.

8. **Why use content-based filtering?**  
   It does not need a large user-rating history and can recommend items based on their metadata.

9. **What is a limitation of this project?**  
   Recommendations depend on the quality and coverage of the movie descriptions and keywords. Similarity does not guarantee user preference.

10. **Why is accuracy not used?**  
    The project does not have labelled correct recommendations for each movie. Ranking metrics require relevance labels or user feedback.

11. **What is a sparse matrix?**  
    A matrix in which most entries are zero. TF-IDF commonly produces sparse matrices because each document contains only a small portion of all vocabulary terms.

12. **What is content-based filtering's cold-start limitation?**  
    A new item needs descriptive metadata before it can be recommended. It does not have the same user-interaction cold-start problem as collaborative filtering.

13. **What does `stop_words='english'` do?**  
    It removes common English words that usually carry limited information for retrieval.

14. **Why use bigrams?**  
    Bigrams capture two-word phrases that may carry more meaning than isolated words.

15. **How can the project be improved?**  
    Expand the catalogue, add user ratings, use hybrid recommendation, collect relevance feedback and evaluate with Precision@K or NDCG.
