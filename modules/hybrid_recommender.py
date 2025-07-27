from modules.content_recommender import ContentRecommender
from modules.collaborative_recommender import CollaborativeRecommender
import pandas as pd

class HybridRecommender:
    def __init__(self, content_df_path, ratings_csv_path):
        self.content_df = pd.read_csv(content_df_path)
        self.ratings_path = ratings_csv_path

        self.content_rec = ContentRecommender(self.content_df)
        self.collab_rec = CollaborativeRecommender(ratings_csv_path)
        self.collab_rec.load_model('models/svd_model.pkl')  # Load pre-trained model

    def get_recommendations(self, user_id, input_title, top_n=10):
        print("Content DF columns:", self.content_df.columns.tolist())

        content_recs_by_type = self.content_rec.get_recommendations(input_title, top_n=50)

        flat_recs = []
        for media_type, rec_list in content_recs_by_type.items():
            for rec in rec_list:
                if isinstance(rec, dict) and 'title' in rec:
                    rec['media_type'] = media_type
                    flat_recs.append(rec)

        results = []
        for rec in flat_recs:
            #print("DEBUG rec:", rec, "| type:", type(rec))
            title = rec['title'].strip().lower()
            anime_row = self.content_df[self.content_df['title'].str.lower() == title]

            if anime_row.empty:
                continue

            anime_id = anime_row.iloc[0]['anime_id']
            pred_rating = self.collab_rec.model.predict(user_id, anime_id).est
            final_score = (rec['score'] + pred_rating) / 2

            results.append({
                'title': rec['title'],
                'rating': rec['rating'],
                'media_type': rec['media_type'],
                'predicted_rating': round(pred_rating, 2),
                'final_score': round(final_score, 2)
            })

        # Grouping top_n results by media_type
        grouped = {
            'anime_series': [],
            'anime_movie': [],
            'manga': [],
            'manhwa': [],
            'manhua': [],
            'anime_other': []
        }

        results = sorted(results, key=lambda x: x['final_score'], reverse=True)

        for r in results:
            key = r['media_type']
            if key in grouped and len(grouped[key]) < top_n:
                grouped[key].append(r)
            elif len(grouped['anime_other']) < top_n:
                grouped['anime_other'].append(r)

        return grouped
