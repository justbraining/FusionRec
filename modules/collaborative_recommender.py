# modules/collaborative_recommender.py

import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

class CollaborativeRecommender:
    def __init__(self, ratings_csv_path):
        self.df = pd.read_csv(ratings_csv_path)
        self.model = None

    def train_model(self, sample_size=10000):
        reader = Reader(rating_scale=(1, 10))

        # Sample smaller data for faster training
        small_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        data = Dataset.load_from_df(small_df[['user_id', 'anime_id', 'rating']], reader)

        trainset = data.build_full_trainset()
        self.model = SVD()
        self.model.fit(trainset)

        # Optional: Evaluate with RMSE
        _, testset = train_test_split(data, test_size=0.2, random_state=42)
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=True)

        print(f"Model trained with RMSE: {rmse:.4f}")

    def get_user_recommendations(self, user_id, anime_df, top_n=10):
        rated_ids = self.df[self.df['user_id'] == user_id]['anime_id'].tolist()

        candidates = anime_df[~anime_df['anime_id'].isin(rated_ids)]
        candidates = candidates.sample(n=min(500, len(candidates)), random_state=42)

        predictions = [
            (anime['name'], self.model.predict(user_id, anime['anime_id']).est)
            for _, anime in candidates.iterrows()
        ]

        top_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        return [{'title': title, 'predicted_rating': round(score, 2)} for title, score in top_preds]

    def save_model(self, path='models/svd_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path='models/svd_model.pkl'):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print("Loaded pre-trained model")