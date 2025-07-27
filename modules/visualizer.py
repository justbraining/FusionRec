import matplotlib.pyplot as plt
import seaborn as sns

def plot_genre_distribution(df):
    tags_series = df['tags'].dropna().str.lower().str.split(', ')
    all_tags = tags_series.explode().value_counts().head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=all_tags.values, y=all_tags.index, palette='viridis')
    plt.title('Top 20 Genres/Tags')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig('plots/genre_distribution.png')
    plt.show()

def plot_ratings_hist(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['rating'].dropna(), bins=30, kde=True, color='skyblue')
    plt.title('Ratings Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/ratings_hist.png')
    plt.show()

def plot_hybrid_scores(results):
    titles = [r['title'] for r in results]
    scores = [r['final_score'] for r in results]
    plt.figure(figsize=(10,6))
    sns.barplot(x=scores, y=titles, palette='mako')
    plt.xlabel('Hybrid Score')
    plt.title('Top Hybrid Recommendations')
    plt.tight_layout()
    plt.savefig('plots/hybrid_recs.png')
    plt.show()
