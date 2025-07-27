import streamlit as st
import pandas as pd
from modules.hybrid_recommender import HybridRecommender
from modules.visualizer import plot_hybrid_scores, plot_genre_donut, plot_rating_heatmap
from modules.jikan_helper import fetch_anime_info

# Load data paths
CONTENT_PATH = 'data/content_dataset.csv'
RATINGS_PATH = 'data/rating.csv'

# Init recommender
recommender = HybridRecommender(CONTENT_PATH, RATINGS_PATH)

# Page Title
st.set_page_config(page_title="FusionRec", layout="wide")
st.title("🎬 FusionRec - Anime & Manga Recommender")

# Sidebar Inputs
st.sidebar.header("🔍 Input Panel")
user_id = st.sidebar.number_input("User ID", min_value=1, value=5, step=1)
input_title = st.sidebar.text_input("Enter a title", value="Naruto")
top_n = st.sidebar.slider("Top N Recommendations", min_value=5, max_value=20, value=10)

# Trigger Recommendation
if st.sidebar.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        results = recommender.get_recommendations(user_id, input_title, top_n)

        if any(results.values()):
            st.success(f"Recommendations for user {user_id} based on '{input_title}':")

            # Flatten for visualizations
            flat_list = [item for media in results.values() for item in media]
            content_df = pd.read_csv(CONTENT_PATH)

            # === MAIN TABS ===
            main_tabs = st.tabs(["🎯 Recommendations", "📊 Visualizations"])
            with main_tabs[0]:
                media_tab_titles = ['Anime Series', 'Anime Movie', 'Manga', 'Manhwa', 'Manhua', 'Other']
                media_keys = ['anime_series', 'anime_movie', 'manga', 'manhwa', 'manhua', 'anime_other']
                media_tabs = st.tabs(media_tab_titles)

                for tab, key in zip(media_tabs, media_keys):
                    with tab:
                        items = results.get(key, [])
                        if not items:
                            st.info("🚫 No results found in this category.")
                            continue

                        for idx, item in enumerate(items, 1):
                            st.markdown(f"### {idx}. {item['title']}")
                            st.markdown(
                                f"- **Hybrid Score:** `{item['final_score']}`  \n"
                                f"- **Original Rating:** `{item['rating']}`  \n"
                                f"- **Predicted Rating:** `{item['predicted_rating']}`"
                            )

                            anime_info = fetch_anime_info(item['title'])
                            cols = st.columns([1, 4])
                            if anime_info["image_url"]:
                                with cols[0]:
                                    st.image(anime_info["image_url"], width=100)
                            with cols[1]:
                                st.markdown(f"*{anime_info['synopsis']}*")
                                st.markdown(f"[📎 More Info]({anime_info['url']})")
                            st.markdown("---")

            # === VISUALIZATION TAB ===
            with main_tabs[1]:
                st.markdown("### 📈 Explore Visual Insights")
                viz_tabs = st.tabs(["Hybrid Score", "Genre Donut", "Ratings Heatmap"])

                with viz_tabs[0]:
                    if flat_list:
                        fig = plot_hybrid_scores(flat_list, show=False)
                        st.pyplot(fig)
                    else:
                        st.warning("No data to visualize hybrid scores.")

                with viz_tabs[1]:
                    fig = plot_genre_donut(content_df)
                    st.pyplot(fig)

                with viz_tabs[2]:
                    fig = plot_rating_heatmap(content_df)
                    st.pyplot(fig)

        else:
            st.warning("No recommendations found. Try a different title or user ID.")
