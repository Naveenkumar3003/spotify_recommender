"""
Spotify Song Recommender - Streamlit App
Interactive web interface for song recommendations.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from recommender import SpotifySongRecommender

# ═══════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main header */
    .main-title {
        text-align: center;
        color: #1DB954;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #b3b3b3;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Song cards */
    .song-card {
        background-color: #282828;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1DB954;
    }
    
    /* Match badge */
    .match-badge {
        background-color: #1DB954;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    
    /* Genre tag */
    .genre-tag {
        background-color: #333;
        color: #1DB954;
        padding: 0.15rem 0.5rem;
        border-radius: 8px;
        font-size: 0.75rem;
        border: 1px solid #1DB954;
    }
    
    /* Divider */
    .custom-divider {
        border-top: 1px solid #333;
        margin: 1.5rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Load Model (Cached)
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def load_recommender():
    """Load the recommender model (cached across sessions)"""
    return SpotifySongRecommender(data_path="data/spotify_processed.csv")


# ═══════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════

def create_radar_chart(input_song, recommendations, feature_columns):
    """Create radar chart comparing input song with recommendations"""
    
    radar_features = [
        'danceability', 'energy', 'speechiness',
        'acousticness', 'instrumentalness', 'valence'
    ]
    
    fig = go.Figure()
    
    # Input song
    values = [input_song[f] for f in radar_features]
    values.append(values[0])  # Close polygon
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=radar_features + [radar_features[0]],
        fill='toself',
        name=f"🎵 {input_song['track_name'][:25]}",
        line=dict(color='#1DB954', width=3),
        fillcolor='rgba(29, 185, 84, 0.15)'
    ))
    
    # Top 3 recommendations
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, rec in enumerate(recommendations[:3]):
        values = [rec[f] for f in radar_features]
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_features + [radar_features[0]],
            fill='toself',
            name=f"{i+1}. {rec['track_name'][:25]}",
            line=dict(color=colors[i], width=2),
            fillcolor=f'rgba(0,0,0,0)'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title=dict(text="Audio Feature Comparison", font=dict(size=16)),
        height=450,
        margin=dict(t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_feature_bars(input_song, recommendation, feature_columns):
    """Create side-by-side bar chart"""
    
    features = ['danceability', 'energy', 'acousticness', 'valence', 'speechiness', 'liveness']
    
    input_values = [input_song[f] for f in features]
    rec_values = [recommendation[f] for f in features]
    
    fig = go.Figure(data=[
        go.Bar(
            name=input_song['track_name'][:20],
            x=features,
            y=input_values,
            marker_color='#1DB954'
        ),
        go.Bar(
            name=recommendation['track_name'][:20],
            x=features,
            y=rec_values,
            marker_color='#FF6B6B'
        )
    ])
    
    fig.update_layout(
        barmode='group',
        title=dict(text="Feature Comparison", font=dict(size=16)),
        yaxis_title="Value (0-1)",
        height=350,
        margin=dict(t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_scatter_map(df, input_index, rec_indices):
    """Create 2D scatter plot of songs"""
    
    fig = go.Figure()
    
    # Sample background songs (for performance)
    sample_size = min(3000, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    
    # All songs (background)
    fig.add_trace(go.Scatter(
        x=sample_df['energy'],
        y=sample_df['valence'],
        mode='markers',
        marker=dict(size=3, color='#555', opacity=0.3),
        name='All Songs',
        hoverinfo='skip'
    ))
    
    # Recommended songs
    rec_df = df.iloc[rec_indices]
    fig.add_trace(go.Scatter(
        x=rec_df['energy'],
        y=rec_df['valence'],
        mode='markers',
        marker=dict(size=14, color='#FF6B6B', symbol='diamond',
                   line=dict(width=1, color='white')),
        name='Recommendations',
        text=[f"{row['track_name']} - {row['artists']}" for _, row in rec_df.iterrows()],
        hovertemplate='%{text}<br>Energy: %{x:.2f}<br>Valence: %{y:.2f}<extra></extra>'
    ))
    
    # Input song
    input_row = df.iloc[input_index]
    fig.add_trace(go.Scatter(
        x=[input_row['energy']],
        y=[input_row['valence']],
        mode='markers',
        marker=dict(size=20, color='#1DB954', symbol='star',
                   line=dict(width=2, color='white')),
        name='Your Song',
        text=[f"{input_row['track_name']} - {input_row['artists']}"],
        hovertemplate='%{text}<br>Energy: %{x:.2f}<br>Valence: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Energy →",
        yaxis_title="Happiness (Valence) →",
        title=dict(text="Song Map: Energy vs Mood", font=dict(size=16)),
        height=450,
        margin=dict(t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_heatmap(recommendations):
    """Create heatmap of recommendation features"""
    
    features = ['danceability', 'energy', 'acousticness', 'valence', 'speechiness', 'liveness']
    
    data = []
    labels = []
    
    for rec in recommendations[:8]:
        values = [rec[f] for f in features]
        data.append(values)
        labels.append(f"{rec['track_name'][:20]}")
    
    fig = px.imshow(
        data,
        x=features,
        y=labels,
        color_continuous_scale='Viridis',
        title="Recommendation Feature Heatmap"
    )
    
    fig.update_layout(
        height=350,
        margin=dict(t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


# ═══════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<p class="main-title">🎵 Spotify Song Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Find songs similar to your favorites based on audio features</p>', unsafe_allow_html=True)
    
    # Load recommender
    recommender = load_recommender()
    
    # ─────────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Number of recommendations
        n_recs = st.slider(
            "Number of recommendations",
            min_value=3, max_value=20, value=10
        )
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Algorithm",
            ["KNN (K-Nearest Neighbors)", "Cosine Similarity"]
        )
        
        # Genre filter
        st.markdown("---")
        st.markdown("## 🎭 Genre Filter")
        
        available_genres = recommender.get_available_genres()
        genre_filter = st.selectbox(
            "Filter by genre (optional)",
            ["All Genres"] + available_genres
        )
        
        # Feature weights
        st.markdown("---")
        st.markdown("## ⚖️ Feature Importance")
        st.caption("Coming in v2.0")
        
        # Dataset info
        st.markdown("---")
        st.markdown("## 📊 Dataset Info")
        st.markdown(f"- **Songs:** {len(recommender.df):,}")
        st.markdown(f"- **Artists:** {recommender.df['artists'].nunique():,}")
        st.markdown(f"- **Genres:** {recommender.df['track_genre'].nunique()}")
    
    # ─────────────────────────────────────────────
    # Main Content - Tab Selection
    # ─────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Search by Song",
        "🎛️ Search by Mood",
        "📋 Playlist Mode"
    ])
    
    # ═══════════════════════════════════════════════
    # TAB 1: Search by Song Name
    # ═══════════════════════════════════════════════
    with tab1:
        st.markdown("### Find songs similar to one you love")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            song_query = st.text_input(
                "🔍 Song name",
                placeholder="e.g., Blinding Lights, Bohemian Rhapsody, Shape of You...",
                key="song_search"
            )
        
        with col2:
            artist_query = st.text_input(
                "🎤 Artist (optional)",
                placeholder="e.g., The Weeknd",
                key="artist_search"
            )
        
        if song_query:
            # Search for the song
            song_index, matches = recommender.find_song(
                song_query,
                artist_query if artist_query else None
            )
            
            if song_index is None:
                st.error(f"❌ Song '{song_query}' not found in database.")
                
                # Show suggestions
                partial = recommender.df[
                    recommender.df['track_name'].str.lower().str.contains(
                        song_query.lower()[:4], na=False
                    )
                ].head(5)
                
                if not partial.empty:
                    st.info("💡 Did you mean:")
                    for _, row in partial.iterrows():
                        st.write(f"  • **{row['track_name']}** - {row['artists']}")
            
            else:
                # Display input song info
                input_song = recommender.get_song_info(song_index)
                
                st.markdown("---")
                st.markdown("### 🎵 Your Song")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**{input_song['track_name']}**")
                    st.markdown(f"*{input_song['artists']}*")
                    st.caption(f"Album: {input_song.get('album_name', 'N/A')}")
                
                with col2:
                    st.caption(f"Genre: {input_song.get('track_genre', 'N/A')}")
                    st.caption(f"Popularity: {input_song.get('popularity', 'N/A')}/100")
                
                with col3:
                    if 'popularity' in input_song:
                        st.metric("Popularity", f"{input_song['popularity']}")
                
                # Show multiple matches if found
                if matches is not None and len(matches) > 1:
                    with st.expander(f"📋 Found {len(matches)} matches — click to see all"):
                        display_matches = matches[['track_name', 'artists', 'track_genre', 'popularity']].head(10)
                        st.dataframe(display_matches, use_container_width=True)
                
                st.markdown("---")
                
                # Get recommendations
                if genre_filter != "All Genres":
                    recommendations = recommender.recommend_by_genre(
                        song_index, genre=genre_filter, n_recommendations=n_recs
                    )
                elif "KNN" in algorithm:
                    recommendations = recommender.recommend_by_knn(song_index, n_recs)
                else:
                    recommendations = recommender.recommend_by_cosine(song_index, n_recs)
                
                if not recommendations:
                    st.warning("No recommendations found. Try a different genre filter.")
                else:
                    # Display recommendations
                    st.markdown(f"### 🎶 Recommended Songs ({len(recommendations)})")
                    
                    for i, rec in enumerate(recommendations, 1):
                        col1, col2, col3, col4 = st.columns([0.3, 2.5, 1.5, 0.7])
                        
                        with col1:
                            st.markdown(f"**#{i}**")
                        
                        with col2:
                            st.markdown(f"**{rec['track_name']}**")
                            st.caption(f"{rec['artists']} • {rec.get('genre', 'N/A')}")
                        
                        with col3:
                            # Mini feature preview
                            st.caption(
                                f"🎵 Dance: {rec['danceability']:.2f} | "
                                f"⚡ Energy: {rec['energy']:.2f} | "
                                f"😊 Mood: {rec['valence']:.2f}"
                            )
                        
                        with col4:
                            score_pct = int(rec['similarity_score'] * 100)
                            st.markdown(
                                f'<span class="match-badge">{score_pct}% match</span>',
                                unsafe_allow_html=True
                            )
                        
                        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                    
                    # ─────────────────────────────────────────
                    # Visualizations
                    # ─────────────────────────────────────────
                    st.markdown("### 📊 Visual Analysis")
                    
                    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                        "🕸️ Radar Chart",
                        "📊 Feature Bars",
                        "🗺️ Song Map",
                        "🔥 Heatmap"
                    ])
                    
                    with viz_tab1:
                        radar_fig = create_radar_chart(
                            input_song, recommendations, recommender.feature_columns
                        )
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    with viz_tab2:
                        # Let user select which recommendation to compare
                        compare_idx = st.selectbox(
                            "Compare with:",
                            range(len(recommendations)),
                            format_func=lambda x: f"{recommendations[x]['track_name']} - {recommendations[x]['artists']}",
                            key="compare_select"
                        )
                        bar_fig = create_feature_bars(
                            input_song, recommendations[compare_idx], recommender.feature_columns
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)
                    
                    with viz_tab3:
                        rec_indices = [r['index'] for r in recommendations]
                        scatter_fig = create_scatter_map(
                            recommender.df, song_index, rec_indices
                        )
                        st.plotly_chart(scatter_fig, use_container_width=True)
                    
                    with viz_tab4:
                        heatmap_fig = create_heatmap(recommendations)
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # ─────────────────────────────────────────
                    # Explainability
                    # ─────────────────────────────────────────
                    st.markdown("### 🧠 Why This Recommendation?")
                    
                    explain_idx = st.selectbox(
                        "Select a song to explain:",
                        range(len(recommendations)),
                        format_func=lambda x: f"{recommendations[x]['track_name']} - {recommendations[x]['artists']}",
                        key="explain_select"
                    )
                    
                    explanation = recommender.explain_recommendation(
                        song_index,
                        recommendations[explain_idx]['index']
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(
                            f"**Most similar in:** {', '.join(explanation['most_similar_features'])}"
                        )
                    
                    with col2:
                        st.warning(
                            f"**Differs most in:** {', '.join(explanation['most_different_features'])}"
                        )
                    
                    st.info(
                        f"**Overall similarity:** {explanation['overall_similarity']*100:.1f}%"
                    )
    
    # ═══════════════════════════════════════════════
    # TAB 2: Search by Mood
    # ═══════════════════════════════════════════════
    with tab2:
        st.markdown("### Describe your mood and find matching songs")
        
        # Preset moods
        st.markdown("#### 🎭 Quick Presets")
        
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        with preset_col1:
            party_preset = st.button("🎉 Party", use_container_width=True)
        with preset_col2:
            chill_preset = st.button("🌙 Chill", use_container_width=True)
        with preset_col3:
            workout_preset = st.button("💪 Workout", use_container_width=True)
        with preset_col4:
            sad_preset = st.button("😢 Sad", use_container_width=True)
        
        # Set default values based on preset
        if party_preset:
            default_dance, default_energy, default_valence = 0.9, 0.9, 0.9
            default_acoustic, default_tempo, default_speech = 0.1, 0.7, 0.1
        elif chill_preset:
            default_dance, default_energy, default_valence = 0.3, 0.2, 0.4
            default_acoustic, default_tempo, default_speech = 0.9, 0.3, 0.05
        elif workout_preset:
            default_dance, default_energy, default_valence = 0.7, 1.0, 0.6
            default_acoustic, default_tempo, default_speech = 0.0, 0.8, 0.1
        elif sad_preset:
            default_dance, default_energy, default_valence = 0.3, 0.3, 0.1
            default_acoustic, default_tempo, default_speech = 0.7, 0.3, 0.05
        else:
            default_dance, default_energy, default_valence = 0.5, 0.5, 0.5
            default_acoustic, default_tempo, default_speech = 0.5, 0.5, 0.1
        
        st.markdown("#### 🎛️ Fine-tune your preferences")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            custom_dance = st.slider(
                "💃 Danceability", 0.0, 1.0, default_dance, 0.05, key="mood_dance"
            )
            custom_energy = st.slider(
                "⚡ Energy", 0.0, 1.0, default_energy, 0.05, key="mood_energy"
            )
        
        with col2:
            custom_valence = st.slider(
                "😊 Happiness", 0.0, 1.0, default_valence, 0.05, key="mood_valence"
            )
            custom_acoustic = st.slider(
                "🎸 Acousticness", 0.0, 1.0, default_acoustic, 0.05, key="mood_acoustic"
            )
        
        with col3:
            custom_tempo = st.slider(
                "🏃 Tempo", 0.0, 1.0, default_tempo, 0.05, key="mood_tempo"
            )
            custom_speech = st.slider(
                "🗣️ Speechiness", 0.0, 1.0, default_speech, 0.05, key="mood_speech"
            )
        
        if st.button("🔍 Find Songs For This Mood", type="primary", use_container_width=True):
            custom_features = {
                'danceability': custom_dance,
                'energy': custom_energy,
                'valence': custom_valence,
                'acousticness': custom_acoustic,
                'tempo': custom_tempo,
                'speechiness': custom_speech,
                'instrumentalness': 0.2,
                'liveness': 0.2,
                'loudness': custom_energy * 0.8  # Approximate
            }
            
            mood_recs = recommender.recommend_by_features(custom_features, n_recs)
            
            st.markdown("---")
            st.markdown(f"### 🎶 Songs Matching Your Mood ({len(mood_recs)})")
            
            for i, rec in enumerate(mood_recs, 1):
                col1, col2, col3 = st.columns([0.3, 3, 0.7])
                
                with col1:
                    st.markdown(f"**#{i}**")
                
                with col2:
                    st.markdown(f"**{rec['track_name']}** - {rec['artists']}")
                    st.caption(f"Genre: {rec.get('genre', 'N/A')}")
                
                with col3:
                    score_pct = int(rec['similarity_score'] * 100)
                    st.markdown(f"**{score_pct}%**")
                
                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            
            # Radar chart for mood results
            if mood_recs:
                # Create a "virtual" input song from custom features
                virtual_input = {
                    'track_name': 'Your Mood',
                    **custom_features
                }
                radar_fig = create_radar_chart(
                    virtual_input, mood_recs, recommender.feature_columns
                )
                st.plotly_chart(radar_fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════
    # TAB 3: Playlist Mode
    # ═══════════════════════════════════════════════
    with tab3:
        st.markdown("### Add multiple songs and get recommendations based on all of them")
        
        st.markdown("#### 🎵 Add Songs to Your Playlist")
        
        # Multi-song input
        playlist_input = st.text_area(
            "Enter song names (one per line):",
            placeholder="Blinding Lights\nShape of You\nBohemian Rhapsody\nLose Yourself",
            height=150,
            key="playlist_input"
        )
        
        if st.button("🎶 Get Playlist Recommendations", type="primary", use_container_width=True):
            if playlist_input.strip():
                song_names = [s.strip() for s in playlist_input.strip().split('\n') if s.strip()]
                
                # Find all songs
                found_indices = []
                found_songs = []
                not_found = []
                
                for name in song_names:
                    idx, _ = recommender.find_song(name)
                    if idx is not None:
                        found_indices.append(idx)
                        info = recommender.get_song_info(idx)
                        found_songs.append(info)
                    else:
                        not_found.append(name)
                
                # Show what was found
                if found_songs:
                    st.markdown("#### ✅ Songs Found:")
                    for song in found_songs:
                        st.write(f"  • **{song['track_name']}** - {song['artists']}")
                
                if not_found:
                    st.warning(f"❌ Not found: {', '.join(not_found)}")
                
                if len(found_indices) >= 1:
                    st.markdown("---")
                    
                    # Get playlist-based recommendations
                    playlist_recs = recommender.recommend_by_playlist(
                        found_indices, n_recs
                    )
                    
                    st.markdown(f"### 🎶 Recommended Based on Your Playlist ({len(playlist_recs)})")
                    
                    for i, rec in enumerate(playlist_recs, 1):
                        col1, col2, col3 = st.columns([0.3, 3, 0.7])
                        
                        with col1:
                            st.markdown(f"**#{i}**")
                        
                        with col2:
                            st.markdown(f"**{rec['track_name']}** - {rec['artists']}")
                            st.caption(f"Genre: {rec.get('genre', 'N/A')}")
                        
                        with col3:
                            score_pct = int(rec['similarity_score'] * 100)
                            st.markdown(f"**{score_pct}%**")
                        
                        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                    
                    # Visualization
                    if playlist_recs:
                        rec_indices = [r['index'] for r in playlist_recs]
                        scatter_fig = create_scatter_map(
                            recommender.df, found_indices[0], rec_indices
                        )
                        st.plotly_chart(scatter_fig, use_container_width=True)
                else:
                    st.error("No songs found. Please check the song names.")
            else:
                st.warning("Please enter at least one song name.")
    
    # ─────────────────────────────────────────────
    # Footer
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.85rem;">'
        'Built with Streamlit • Data from Spotify/Kaggle • '
        'ML: KNN + Cosine Similarity • 89,000+ songs indexed'
        '</p>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════
# Run App
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()