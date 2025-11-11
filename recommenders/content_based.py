# recommenders/content_based.py

from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


class ImdbContentBasedRecommender(BaseRecommender):
    """
    Content-based movie recommender using IMDb 'movie_metadata.csv'.

    It builds a text representation for each movie from:
      - genres
      - plot_keywords
      - director_name
      - main cast names

    Then uses TF–IDF + cosine similarity to find similar movies.
    """

    def __init__(self, csv_path: str = "data/movie_metadata.csv"):
        self.csv_path = csv_path
        self.movies_df: pd.DataFrame | None = None
        self.tfidf_matrix = None
        self.similarity_matrix = None

        self._fit()

    def _load_and_clean(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        # Keep only rows with a movie title and something to work with
        df = df.dropna(subset=["movie_title", "genres"])

        # Normalize and strip text fields
        for col in [
            "movie_title", "director_name", "actor_1_name",
            "actor_2_name", "actor_3_name", "genres", "plot_keywords",
            "language", "country", "content_rating"
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()

        # Convert pipe-separated fields to space-separated tokens
        # e.g. "Action|Adventure|Sci-Fi" -> "Action Adventure Sci-Fi"
        if "genres" in df.columns:
            df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
        else:
            df["genres_clean"] = ""

        if "plot_keywords" in df.columns:
            df["plot_keywords_clean"] = df["plot_keywords"].str.replace("|", " ", regex=False)
        else:
            df["plot_keywords_clean"] = ""

        # Some numeric fields that may be useful to display later
        if "duration" in df.columns:
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

        if "title_year" in df.columns:
            df["title_year"] = pd.to_numeric(df["title_year"], errors="coerce")

        if "imdb_score" in df.columns:
            df["imdb_score"] = pd.to_numeric(df["imdb_score"], errors="coerce")

        # Drop rows that completely lack language (optional)
        if "language" in df.columns:
            df["language"] = df["language"].replace("", "Unknown")

        return df

    def _build_text_features(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine multiple descriptive text fields into one "bag of words"
        representation used for TF–IDF.
        """
        parts = [
            df["genres_clean"],
            df["plot_keywords_clean"],
            df["director_name"],
            df["actor_1_name"],
            df["actor_2_name"],
            df["actor_3_name"],
        ]

        # Concatenate with spaces between parts
        text_features = parts[0]
        for p in parts[1:]:
            text_features = text_features + " " + p

        return text_features

    def _fit(self) -> None:
        """Load data, build text features, TF–IDF matrix, and similarity matrix."""
        df = self._load_and_clean()
        df = df.reset_index(drop=True)

        # Build text features
        df["text_features"] = self._build_text_features(df)

        # Save cleaned df
        self.movies_df = df

        # Vectorize with TF–IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = vectorizer.fit_transform(df["text_features"])

        # Precompute cosine similarity (OK for a few thousand movies)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix,
                                                   self.tfidf_matrix)

    def get_items(self) -> List[str]:
        """Return a sorted list of unique movie titles."""
        # movie_title in this dataset often has trailing spaces or weird chars; we've stripped.
        titles = self.movies_df["movie_title"].tolist()
        # Optionally deduplicate while preserving order
        seen = set()
        unique_titles: List[str] = []
        for t in titles:
            if t not in seen:
                seen.add(t)
                unique_titles.append(t)
        return sorted(unique_titles)

    def recommend(self, item_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Recommend top_n movies similar to the given title.
        Returns a DataFrame with:
          - movie_title
          - genres
          - imdb_score (if available)
          - language
          - country
          - content_rating
          - explanation
          - score (similarity)
        """
        if item_title not in self.movies_df["movie_title"].values:
            raise ValueError(f"Title '{item_title}' not found in dataset.")

        # Index of the selected movie
        idx = self.movies_df.index[self.movies_df["movie_title"] == item_title][0]

        # Similarity scores vs all movies
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity (descending) and skip the first (same movie)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

        movie_indices = [i for i, _ in sim_scores]
        scores = [score for _, score in sim_scores]

        results = self.movies_df.iloc[movie_indices].copy()
        results["score"] = scores

        # Build simple HCI-friendly explanations based on overlapping genres
        selected_genres_raw = self.movies_df.loc[idx, "genres"]
        selected_genres = set(
            g.strip() for g in selected_genres_raw.split("|") if g.strip()
        )

        explanations: list[str] = []
        for _, row in results.iterrows():
            movie_genres_raw = row["genres"]
            movie_genres = set(
                g.strip() for g in movie_genres_raw.split("|") if g.strip()
            )
            common = selected_genres.intersection(movie_genres)
            if common:
                reason = f"Shares genres: {', '.join(sorted(common))}"
            else:
                reason = "Similar themes/keywords based on description."
            explanations.append(reason)

        results["explanation"] = explanations

        # Select a subset of columns to return
        cols = ["movie_title", "genres"]
        if "imdb_score" in results.columns:
            cols.append("imdb_score")
        if "language" in results.columns:
            cols.append("language")
        if "country" in results.columns:
            cols.append("country")
        if "content_rating" in results.columns:
            cols.append("content_rating")

        cols.extend(["score", "explanation"])

        return results[cols]
