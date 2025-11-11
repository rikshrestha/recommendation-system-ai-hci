# test_recommender.py

from recommenders.content_based import ImdbContentBasedRecommender


def main():
    rec = ImdbContentBasedRecommender("data/movie_metadata.csv")

    print("Loaded movies:", len(rec.movies_df))
    print()

    titles = rec.get_items()
    print("Sample titles:")
    for t in titles[:20]:
        print("-", t)

    # Pick a known title from the dataset, e.g. "Avatar" or "The Dark Knight Rises"
    test_title = "Avatar"
    print(f"\nRecommendations for: {test_title}\n")

    results = rec.recommend(test_title, top_n=5)
    for _, row in results.iterrows():
        print(f"Title: {row['movie_title']}")
        print(f"Genres: {row['genres']}")
        if 'imdb_score' in row:
            print(f"IMDb score: {row['imdb_score']}")
        print(f"Explanation: {row['explanation']}")
        print(f"Similarity score: {row['score']:.3f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
