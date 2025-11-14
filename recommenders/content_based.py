# recommenders/content_based.py

from __future__ import annotations

from typing import List, Optional
from pathlib import Path
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


# ---------------------------------------------------------------------
# Tiny logger to a dedicated file; quiet by default in console.
# ---------------------------------------------------------------------
def _get_logger() -> logging.Logger:
    logger = logging.getLogger("recommender_core")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on Streamlit reruns
    if logger.handlers:
        return logger

    Path("logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler("logs/content_based.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    return logger


log = _get_logger()


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
        self.movies_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self._title_to_index: dict[str, int] = {}

        self._fit()

    # -----------------------------
    # Data loading & cleaning
    # -----------------------------
    def _load_and_clean(self) -> pd.DataFrame:
        p = Path(self.csv_path)
        
        if not p.exists():
            msg = f"Dataset not found at: {p.resolve()}"
            log.error(msg)
            raise FileNotFoundError(msg)

        try:
            df = pd.read_csv(self.csv_path)
        
        except Exception as e:
            log.exception("Failed to read dataset: %s", self.csv_path)
            raise RuntimeError(f"Dataset could not be read: {e}") from e

        # Keep only rows with a movie title and something to work with
        df = df.dropna(subset=["movie_title", "genres"])

        # Normalize and strip text fields
        for col in [
            "movie_title", "director_name", "actor_1_name",
            "actor_2_name", "actor_3_name", "genres", "plot_keywords",
            "language", "country", "content_rating",
        ]:
            if col in df.columns:
                # strip, coerce to str, and normalize NBSP \xa0 which appears in this dataset
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.replace("\xa0", " ", regex=False)
                    .str.strip()
                )

        # Convert pipe-separated fields to space-separated tokens
        # e.g. "Action|Adventure|Sci-Fi" -> "Action Adventure Sci-Fi"
        df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False) if "genres" in df.columns else ""
        df["plot_keywords_clean"] = (
            df["plot_keywords"].str.replace("|", " ", regex=False) if "plot_keywords" in df.columns else ""
        )

        # Numeric fields we may display or filter on
        for col in ("duration", "title_year", "imdb_score"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "language" in df.columns:
            df["language"] = df["language"].replace("", "Unknown")

        log.info("Loaded %d movies from %s", len(df), p.name)
        return df.reset_index(drop=True)

    # -----------------------------
    # Feature engineering
    # -----------------------------
    def _build_text_features(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine multiple descriptive text fields into one "bag of words"
        representation used for TF–IDF.
        """
        parts = [
            df.get("genres_clean", pd.Series("", index=df.index)),
            df.get("plot_keywords_clean", pd.Series("", index=df.index)),
            df.get("director_name", pd.Series("", index=df.index)),
            df.get("actor_1_name", pd.Series("", index=df.index)),
            df.get("actor_2_name", pd.Series("", index=df.index)),
            df.get("actor_3_name", pd.Series("", index=df.index)),
        ]

        text_features = parts[0]
        for p in parts[1:]:
            text_features = text_features + " " + p

        # Ensure string dtype and strip (guard against all-empty)
        text_features = text_features.fillna("").astype(str).str.strip()
        return text_features

    # -----------------------------
    # Model build
    # -----------------------------
    def _fit(self) -> None:
        """Load data, build text features, TF–IDF matrix, and similarity matrix."""
        df = self._load_and_clean()

        # Build text features and drop rows with no usable text (rare)
        df["text_features"] = self._build_text_features(df)
        df = df[df["text_features"] != ""].reset_index(drop=True)

        if df.empty:
            msg = "After preprocessing, no movies had usable text features."
            log.error(msg)
            raise ValueError(msg)

        self.movies_df = df

        # Vectorize with TF–IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = vectorizer.fit_transform(df["text_features"])

        # Precompute cosine similarity (OK for a few thousand movies)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Fast lookup: title -> row index (last occurrence wins if duplicates)
        self._title_to_index = {title: i for i, title in enumerate(df["movie_title"])}

        log.info(
            "TF–IDF shape: %s | Similarity matrix: %s | Titles indexed: %d",
            tuple(self.tfidf_matrix.shape),
            tuple(self.similarity_matrix.shape),
            len(self._title_to_index),
        )

    # -----------------------------
    # Public API
    # -----------------------------
    def get_items(self) -> List[str]:

        """Return a sorted list of unique movie titles."""
        if self.movies_df is None:
            return []

        titles = self.movies_df["movie_title"].tolist()

        # Deduplicate while preserving first occurrence
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
        if self.movies_df is None or self.similarity_matrix is None:
            raise RuntimeError("Recommender not initialized. Call constructor first.")

        if item_title not in self._title_to_index:
            raise ValueError(f"Title '{item_title}' not found in dataset.")

        idx = self._title_to_index[item_title]

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
        selected_genres = set(g.strip() for g in selected_genres_raw.split("|") if g.strip())

        explanations: list[str] = []
        for _, row in results.iterrows():
            movie_genres_raw = row["genres"]
            movie_genres = set(g.strip() for g in movie_genres_raw.split("|") if g.strip())
            common = selected_genres.intersection(movie_genres)

            if common:
                reason = f"Shares genres: {', '.join(sorted(common))}"
            
            else:
                reason = "Similar themes/keywords based on description."
            
            explanations.append(reason)

        results["explanation"] = explanations

        # Select a subset of columns to return
        cols = ["movie_title", "genres"]

        for c in ("imdb_score", "language", "country", "content_rating"):
            if c in results.columns:
                cols.append(c)
        
        cols.extend(["score", "explanation"])

        log.info("Generated %d recommendations for '%s'", len(results), item_title)
        return results[cols]
