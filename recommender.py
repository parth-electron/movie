"""
Content-based movie recommendation engine.

Approach:
    1. Combine each movie's genres + overview into one "content" string.
    2. Vectorize all movies' content using TF-IDF.
    3. Compute pairwise cosine similarity between all movies.
    4. To recommend for a given movie, look up its row in the similarity
       matrix and return the highest-scoring movies (excluding itself).

This is a classic, dependency-light approach that works well without
needing any user rating history (unlike collaborative filtering) and
without needing an external API/dataset download.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from movies_data import MOVIES


class MovieRecommender:
    def __init__(self, movies=None):
        self.movies = movies if movies is not None else MOVIES
        self.df = pd.DataFrame(self.movies)

        # Build a single text blob per movie: genres (repeated for weight) + overview
        self.df["content"] = self.df.apply(self._build_content, axis=1)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # Map title -> row index, and a lowercase lookup for case-insensitive search
        self.title_to_index = {title: idx for idx, title in enumerate(self.df["title"])}
        self.lower_title_to_title = {t.lower(): t for t in self.df["title"]}

    @staticmethod
    def _build_content(row):
        genres_text = " ".join(row["genres"]) * 2  # weight genres more heavily
        return f"{genres_text} {row['overview']}"

    def all_titles(self):
        """Return a sorted list of all movie titles (for search/autocomplete)."""
        return sorted(self.df["title"].tolist())

    def search(self, query, limit=10):
        """Simple case-insensitive substring search over titles."""
        query = query.lower().strip()
        if not query:
            return []
        matches = [t for t in self.df["title"] if query in t.lower()]
        return matches[:limit]

    def get_movie(self, title):
        """Return the movie dict for an exact title, or None."""
        idx = self.title_to_index.get(title)
        if idx is None:
            return None
        return self.df.iloc[idx].to_dict()

    def recommend(self, title, top_n=10):
        """
        Return up to top_n recommended movies similar to `title`.
        Matching is case-insensitive. Returns None if the title isn't found.
        Each result is a dict: title, genres, overview, similarity (0-1).
        """
        resolved_title = self.lower_title_to_title.get(title.lower())
        if resolved_title is None:
            return None

        idx = self.title_to_index[resolved_title]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores.sort(key=lambda x: x[1], reverse=True)

        # Skip the first result: it's always the movie itself (similarity = 1.0)
        top_matches = [s for s in scores if s[0] != idx][:top_n]

        results = []
        for movie_idx, score in top_matches:
            row = self.df.iloc[movie_idx]
            results.append({
                "title": row["title"],
                "genres": row["genres"],
                "overview": row["overview"],
                "similarity": round(float(score), 3),
            })
        return results


# A module-level singleton so the (cheap) TF-IDF computation happens once
# per process, not once per request.
recommender = MovieRecommender()


if __name__ == "__main__":
    # Quick manual test: python recommender.py
    demo_title = "Inception"
    print(f"Recommendations for '{demo_title}':\n")
    for movie in recommender.recommend(demo_title, top_n=5):
        print(f"  {movie['similarity']:.3f}  {movie['title']}  ({', '.join(movie['genres'])})")
