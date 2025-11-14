# app.py
# IMDb Content-Based Movie Recommender (AI/HCI)
# Minimal logging + basic defensive checks
# Author: Rikrish Shrestha

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from recommenders.content_based import ImdbContentBasedRecommender


# -----------------------------
# Settings
# -----------------------------
SETTINGS = {
    "DATA_PATH": "data/movie_metadata.csv",
    "DEFAULT_TOP_N": 5,
    "MAX_RECOMMENDATIONS": 20,
    "PAGE_TITLE": "IMDb Content-Based Movie Recommender",
    "LAYOUT": "wide",
    "MIN_IMDB_DEFAULT": 0.0,
    "LOG_DIR": "logs",
    "LOG_FILE": "logs/app.log",
}


# -----------------------------
# Minimal logger (file + console)
# -----------------------------
def setup_logger() -> logging.Logger:
    """
    Minimal logger:
    - INFO-level messages go to logs/app.log
    - Only ERRORs appear in the Streamlit terminal
    """
    Path(SETTINGS["LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("recommender_app")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on Streamlit reruns
    if logger.handlers:
        return logger

    # File handler — keeps all info-level logs
    file_handler = logging.FileHandler(SETTINGS["LOG_FILE"], encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # Console handler — shows only errors/warnings
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


log = setup_logger()


# -----------------------------
# Cached recommender factory
# -----------------------------
@st.cache_resource
def get_recommender() -> ImdbContentBasedRecommender:
    """
    Build and cache the content-based recommender.
    Expensive work runs once per session; subsequent UI interactions reuse it.
    """
    t0 = time.time()
    rec = ImdbContentBasedRecommender(SETTINGS["DATA_PATH"])
    elapsed = time.time() - t0
    log.info("Recommender initialized in %.2fs", elapsed)
    return rec


# -----------------------------
# Small dataset sanity check
# -----------------------------
def verify_dataset(path_str: str) -> None:
    p = Path(path_str)
    if not p.exists():
        msg = f"Dataset not found at: {p.resolve()}"
        log.error(msg)
        st.error(msg)
        st.stop()
    if p.stat().st_size == 0:
        msg = f"Dataset file is empty: {p.resolve()}"
        log.error(msg)
        st.error(msg)
        st.stop()


# -----------------------------
# App entry
# -----------------------------
def main() -> None:
    # Page config
    st.set_page_config(page_title=SETTINGS["PAGE_TITLE"], layout=SETTINGS["LAYOUT"])
    st.title("🎬 IMDb Content-Based Movie Recommender")

    # Basic data check before heavy work
    verify_dataset(SETTINGS["DATA_PATH"])

    # Build (cached) recommender
    try:
        recommender = get_recommender()
    except Exception as e:
        log.exception("Failed to initialize recommender")
        st.error(f"An error occurred while initializing the recommender: {e}")
        st.stop()

    # --- Sidebar: controls ---
    st.sidebar.header("Controls")

    try:
        all_titles = recommender.get_items()
    except Exception as e:
        log.exception("Failed to load movie list")
        st.error(f"Failed to load the movie list: {e}")
        st.stop()

    # Optional filter to handle long dropdowns
    search_query = st.sidebar.text_input(
        "Filter titles (optional):",
        help="Type part of a movie title to narrow down the list.",
    )

    if search_query:
        sq_lower = search_query.lower()
        filtered_titles = [t for t in all_titles if sq_lower in t.lower()]
        if not filtered_titles:
            st.sidebar.warning("No titles match that filter. Showing all titles.")
            filtered_titles = all_titles
    else:
        filtered_titles = all_titles

    selected_title = st.sidebar.selectbox("Choose a movie you like:", filtered_titles)

    top_n = st.sidebar.slider(
        "Number of recommendations",
        min_value=1,
        max_value=SETTINGS["MAX_RECOMMENDATIONS"],
        value=SETTINGS["DEFAULT_TOP_N"],
        help="How many similar movies to show.",
    )

    min_imdb = st.sidebar.slider(
        "Minimum IMDb score",
        min_value=0.0,
        max_value=10.0,
        value=SETTINGS["MIN_IMDB_DEFAULT"],
        step=0.1,
        help="Filter out low-rated recommendations.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This system uses content-based filtering over genres, plot keywords, "
        "director, and main cast to find similar movies."
    )

    # --- Main content area ---
    with st.expander("ℹ️ How this recommender works", expanded=True):
        st.markdown(
            """
            This recommender is **content-based**:
            - Uses the IMDb `movie_metadata.csv` dataset (about 5,000 movies).
            - Combines **Genres**, **Plot keywords**, **Director**, and **Main cast**.
            - Transforms these fields into TF–IDF vectors.
            - Uses cosine similarity to find similar movies.
            - Explains results via overlapping genres or similar text features.
            """
        )

    st.subheader("Pick a movie and explore similar titles")
    st.markdown(
        f"**Selected movie:** `{selected_title}`  \n"
        f"Use the controls on the left to adjust the number of recommendations "
        f"and the minimum IMDb score."
    )

    if st.button("Get recommendations"):
        try:
            t0 = time.time()
            results = recommender.recommend(selected_title, top_n=top_n)
            elapsed = time.time() - t0
            log.info("recommend('%s', top_n=%d) completed in %.3fs", selected_title, top_n, elapsed)

            # Optional IMDb filter
            if "imdb_score" in results.columns:
                before = len(results)
                results = results[results["imdb_score"].fillna(0) >= min_imdb]
                after = len(results)
                if after < before:
                    log.info("IMDb filter kept %d/%d results (threshold=%.1f)", after, before, min_imdb)

            if results.empty:
                st.warning(
                    "No recommendations met the filter criteria. "
                    "Try lowering the minimum IMDb score or increasing the number of recommendations."
                )
                return

            st.subheader(f"Recommendations similar to: *{selected_title}*")

            for _, row in results.iterrows():
                with st.container():
                    st.markdown(f"### {row['movie_title']}")
                    st.markdown(f"**Genres:** {row['genres']}")

                    # Optional metadata for richer context
                    meta_bits = []
                    if "title_year" in row and not pd.isna(row["title_year"]):
                        try:
                            meta_bits.append(f"Year: {int(row['title_year'])}")
                        except Exception:
                            pass
                    if "imdb_score" in row and not pd.isna(row["imdb_score"]):
                        meta_bits.append(f"IMDb: {row['imdb_score']}")
                    if "content_rating" in row and isinstance(row["content_rating"], str) and row["content_rating"]:
                        meta_bits.append(f"Rating: {row['content_rating']}")
                    if "language" in row and isinstance(row["language"], str) and row["language"]:
                        meta_bits.append(f"Language: {row['language']}")
                    if "country" in row and isinstance(row["country"], str) and row["country"]:
                        meta_bits.append(f"Country: {row['country']}")

                    if meta_bits:
                        st.markdown("**Details:** " + " · ".join(meta_bits))

                    st.markdown(f"**Why recommended:** {row['explanation']}")
                    st.caption(f"Similarity score (cosine): `{row['score']:.3f}`")
                    st.markdown("---")

        except Exception as e:
            log.exception("Error during recommendation generation")
            st.error(f"An error occurred while generating recommendations: {e}")


if __name__ == "__main__":
    main()
