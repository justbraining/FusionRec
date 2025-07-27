import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

class ContentRecommender:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df['combined_text'] = self.df['tags'].fillna('') + ' ' + self.df['description'].fillna('')
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_text'])
        self.title_to_index = pd.Series(self.df.index, index = self.df['title'].str.lower()).to_dict()


    def suggest_titles(self, input_title, n=3, cutoff = 0.7):
        return difflib.get_close_matches(
            input_title.lower(), self.df['title'].str.lower().tolist(), n=n, cutoff=cutoff
        )
    
    def get_recommendations(self, title, top_n = 10):
        title = title.lower().strip()

        if title not in self.title_to_index:
            suggestions = self.suggest_titles(title)
            if suggestions:
                print(f"Title not found. Did you mean:\n - " + "\n - ".join(suggestions))
            else:
                print("Title not found, and no close matches were detected.")
            return {}
        
        idx = self.title_to_index[title]
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[::-1][1:top_n+1]

        recommendations = {
            'anime_series':[], 'anime_movie':[],
            'manga':[], 'manhwa':[], 'manhua':[], 'anime_other':[]
        }

        for i in sim_indices:
            row = self.df.iloc[i]
            entry = {
                'title': row['title'].title(),
                'rating': row['rating'],
                'tags': row['tags'],
                'score': round(sim_scores[i], 3)
            }
            recommendations[row['media_type']].append(entry)

        return recommendations