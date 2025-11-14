# test_recommender.py
# Simple offline test for the IMDb Content-Based Recommender
# This script validates dataset loading and recommendation generation.
# Author: Rikrish Shrestha

import logging
from recommenders.content_based import ImdbContentBasedRecommender


def setup_logger():
    """Set up a simple logger for console + file (optional)."""
    logging.basicConfig(
        filename="logs/test_recommender.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    return logging.getLogger("test_recommender")


def main():
    log = setup_logger()
    log.info("Starting recommender test...")

    try:
        rec = ImdbContentBasedRecommender("data/movie_metadata.csv")
        print(f"✅ Loaded {len(rec.movies_df)} movies.")
        log.info("Loaded %d movies successfully.", len(rec.movies_df))
    
    except Exception as e:
        log.exception("Failed to initialize recommender.")
        print(f"❌ Error loading dataset: {e}")
        return

    titles = rec.get_items()
    print("\nSample titles:")
    for t in titles[:10]:
        print("-", t)

    test_title = "Avatar"
    print(f"\n🎥 Recommendations for: {test_title}\n")

    try:
        results = rec.recommend(test_title, top_n=5)
        for _, row in results.iterrows():
            print(f"Title: {row['movie_title']}")
            print(f"Genres: {row['genres']}")

            if "imdb_score" in row:
                print(f"IMDb score: {row['imdb_score']}")
            
            print(f"Explanation: {row['explanation']}")
            print(f"Similarity score: {row['score']:.3f}")
            print("-" * 60)
        
        log.info("Test completed successfully for '%s'.", test_title)

    except Exception as e:
        log.exception("Error during recommendation test.")
        print(f"❌ Error generating recommendations: {e}")


if __name__ == "__main__":
    main()
