# app.py

import pandas as pd
import streamlit as st
from recommenders.content_based import ImdbContentBasedRecommender


# Centralized configuration/settings
SETTINGS = {
    "DATA_PATH": "data/movie_metadata.csv",
    "DEFAULT_TOP_N": 5,
    "MAX_RECOMMENDATIONS": 20,
    "PAGE_TITLE": "IMDb Content-Based Movie Recommender",
    "LAYOUT": "wide",
    "MIN_IMDB_DEFAULT": 0.0,
}


@st.cache_resource
def get_recommender() -> ImdbContentBasedRecommender:
    """
    Build and cache the content-based recommender.

    This ensures that expensive work (loading CSV, TF–IDF fitting,
    similarity matrix computation) is done only once per session,
    not on every Streamlit rerun.
    """
    return ImdbContentBasedRecommender(SETTINGS["DATA_PATH"])


def main() -> None:
    # Page-level config
    st.set_page_config(
        page_title=SETTINGS["PAGE_TITLE"],
        layout=SETTINGS["LAYOUT"],
    )

    st.title("🎬 IMDb Content-Based Movie Recommender")

    # Load the recommender (cached)
    recommender = get_recommender()

    # --- Sidebar: controls ---
    st.sidebar.header("Controls")

    all_titles = recommender.get_items()

    # Optional title filter to handle long dropdowns efficiently
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

    selected_title = st.sidebar.selectbox(
        "Choose a movie you like:",
        filtered_titles,
    )

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

            - It uses the IMDb `movie_metadata.csv` dataset (about 5,000 movies).
            - For each movie, it combines:
              - **Genres** (`genres`)
              - **Plot keywords** (`plot_keywords`)
              - **Director** (`director_name`)
              - **Main cast** (`actor_1_name`, `actor_2_name`, `actor_3_name`)
            - These fields are transformed into TF–IDF vectors.
            - Cosine similarity between vectors is used to find similar movies.
            - Explanations are based on overlapping genres or similar text features.
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
            results = recommender.recommend(selected_title, top_n=top_n)

            # Optional filter: IMDb score threshold
            if "imdb_score" in results.columns:
                results = results[results["imdb_score"].fillna(0) >= min_imdb]

            if results.empty:
                st.warning(
                    "No recommendations met the filter criteria. "
                    "Try lowering the minimum IMDb score or increasing the "
                    "number of recommendations."
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
            st.error(f"An error occurred while generating recommendations: {e}")


if __name__ == "__main__":
    main()
