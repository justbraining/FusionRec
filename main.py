# main.py

import pandas as pd
from modules.hybrid_recommender import HybridRecommender
from modules.visualizer import plot_genre_distribution, plot_ratings_hist, plot_hybrid_scores

# === CONFIGURATION ===
user_id = 5
input_title = "Naruto"
top_n = 10
content_path = "data/content_dataset.csv"
ratings_path = "data/rating.csv"

# === LOAD DATA AND MODELS ===
print("Initializing Hybrid Recommender...")
hybrid = HybridRecommender(content_df_path=content_path, ratings_csv_path=ratings_path)
print("Loaded pre-trained model\n")

# === HYBRID RECOMMENDATIONS ===
print(f"Hybrid Recommendations for user {user_id} based on '{input_title}':\n")
results = hybrid.get_recommendations(user_id, input_title, top_n=top_n)

# === DISPLAY RESULTS ===
for media_type, rec_list in results.items():
    print(f"\n=== {media_type.upper()} ===")
    for idx, rec in enumerate(rec_list, 1):
        print(f"{idx}. {rec['title']} "
              f"(Hybrid Score: {rec['final_score']}, "
              f"Original Rating: {rec['rating']}, "
              f"Predicted Rating: {rec['predicted_rating']})")

# === PLOT ZONE ===
try:
    print("\n Generating plots...")
    content_df = pd.read_csv(content_path)

    plot_genre_distribution(content_df)
    plot_ratings_hist(content_df)

    # Flatten recommendations to a single list for plotting hybrid scores
    flat_results = [r for recs in results.values() for r in recs]
    plot_hybrid_scores(flat_results)

    print("Plots saved in /plots folder.\n")
except Exception as e:
    print(f"Plotting failed: {e}")
