'''
from modules.content_recommender import ContentRecommender
import pandas as pd

# Load your unified dataset
df = pd.read_csv("data/content_dataset.csv")  # Or the merged final CSV

# Initialize the recommender
recommender = ContentRecommender(df)

# Try fetching recommendations
query = "nartuo"  # Intentional typo for fuzzy testing
results = recommender.get_recommendations(query, top_n=5)

# Print results
for category, items in results.items():
    if items:
        print(f"\n=== {category.upper()} ===")
        for item in items:
            print(f"{item['title']} | Score: {item['score']} | Rating: {item['rating']}")

'''

'''

from modules.collaborative_recommender import CollaborativeRecommender
import pandas as pd
import os

# Load datasets
ratings_path = "data/rating.csv"
anime_df = pd.read_csv("data/anime.csv")
# print(anime_df.columns)


# Init and train
recommender = CollaborativeRecommender(ratings_path)
recommender.train_model()

# Get user recommendations
user_id = 5  # try any ID that exists
recommendations = recommender.get_user_recommendations(user_id, anime_df)

# Display
print(f"\n🎯 Recommendations for user {user_id}:")
for rec in recommendations:
    print(f"{rec['title']} (Predicted Rating: {rec['predicted_rating']})")

recommender.save_model("models/svd_model.pkl")


model_path = "models/svd_model.pkl"
if os.path.exists(model_path):
    recommender.load_model(model_path)
else:
    recommender.train_model()
    recommender.save_model(model_path)

'''

'''
# test_hybrid.py

from modules.hybrid_recommender import HybridRecommender

# Inputs
user_id = 5
input_title = "Naruto"
top_n = 10

# Load hybrid recommender
hybrid = HybridRecommender(content_df_path=(r"D:\FusionRec\data\content_dataset.csv"), ratings_csv_path=(r"D:\FusionRec\data\rating.csv"))
print("Loaded pre-trained model\n")

# Get hybrid recommendations
print(f"🎯 Hybrid Recommendations for user {user_id} based on '{input_title}':\n")
results = hybrid.get_recommendations(user_id, input_title, top_n=top_n)

# Display recommendations grouped by media_type
for media_type, rec_list in results.items():
    print(f"\n=== {media_type.upper()} ===")
    for idx, rec in enumerate(rec_list, 1):
        print(f"{idx}. {rec['title']} "
              f"(Hybrid Score: {rec['final_score']}, "
              f"Original Rating: {rec['rating']}, "
              f"Predicted Rating: {rec['predicted_rating']})")
              
              '''