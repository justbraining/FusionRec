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




def plot_hybrid_scores(results, show=True):
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['final_score'], reverse=True)

    titles = [r['title'] for r in sorted_results]
    scores = [r['final_score'] for r in sorted_results]

    plt.figure(figsize=(10, max(6, len(titles) * 0.5)))  # dynamic height

    ax = sns.barplot(x=scores, y=titles, palette="coolwarm")

    # Add labels on bars
    for i, v in enumerate(scores):
        ax.text(v + 0.05, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

    plt.title(" Hybrid Score Rankings", fontsize=14)
    plt.xlabel("Hybrid Score", fontsize=12)
    plt.ylabel("Titles", fontsize=12)
    plt.xlim(0, max(scores) + 1)

    plt.tight_layout()

    if show:
        plt.show()
    return plt.gcf()


import pandas as pd

def plot_genre_donut(df):
    tags_series = df['tags'].dropna().str.lower().str.split(', ')
    all_tags = tags_series.explode().value_counts().head(10)

    # 🎨 Custom color palette (feel free to tweak)
    colors = [
        '#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1',
        '#955251', '#B565A7', '#009B77', '#DD4124', '#45B8AC'
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        all_tags.values,
        labels=all_tags.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4)
    )

    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title("Top 10 Genres (Donut Chart)")
    return fig


def plot_rating_heatmap(df):
    df_filtered = df[['rating', 'media_type']].dropna()
    df_filtered['rating_bin'] = pd.cut(
        df_filtered['rating'], bins=[0, 2, 4, 6, 8, 10],
        labels=["0–2", "2–4", "4–6", "6–8", "8–10"]
    )
    pivot = pd.pivot_table(df_filtered, values='rating', index='media_type', columns='rating_bin',
                           aggfunc='count', fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='rocket', ax=ax)
    ax.set_title("Ratings Distribution Heatmap")
    return fig
