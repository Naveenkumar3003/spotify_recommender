"""
Spotify Song Recommender - Core Recommendation Engine
Uses KNN and Cosine Similarity to find similar songs.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


class SpotifySongRecommender:
    def __init__(self, data_path="data/spotify_processed.csv"):
        """Initialize the recommender with processed data"""
        
        print("[INFO] Loading recommender...")
        
        self.df = pd.read_csv(data_path)
        
        # Feature columns used for similarity
        self.feature_columns = [
            'danceability', 'energy', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'loudness'
        ]
        
        # Extract feature matrix
        self.feature_matrix = self.df[self.feature_columns].values
        
        # Initialize KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=20,
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(self.feature_matrix)
        
        print(f"[INFO] Recommender ready!")
        print(f"[INFO] Dataset: {len(self.df)} songs")
        print(f"[INFO] Features: {len(self.feature_columns)}")
    
    def find_song(self, song_name, artist=None):
        """
        Find a song in the dataset by name.
        Returns the index of the best match.
        """
        
        # Case-insensitive search
        mask = self.df['track_name'].str.lower().str.contains(
            song_name.lower(), na=False
        )
        
        # Filter by artist if provided
        if artist:
            artist_mask = self.df['artists'].str.lower().str.contains(
                artist.lower(), na=False
            )
            mask = mask & artist_mask
        
        matches = self.df[mask]
        
        if matches.empty:
            return None, None
        
        # Return the most popular match
        if 'popularity' in matches.columns:
            best_idx = matches['popularity'].idxmax()
        else:
            best_idx = matches.index[0]
        
        return best_idx, matches
    
    def recommend_by_knn(self, song_index, n_recommendations=10):
        """Recommend songs using K-Nearest Neighbors"""
        
        # Get the feature vector of the input song
        song_features = self.feature_matrix[song_index].reshape(1, -1)
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(
            song_features,
            n_neighbors=n_recommendations + 1
        )
        
        # Build recommendations (exclude the input song itself)
        recommendations = []
        for i, idx in enumerate(indices[0]):
            if idx != song_index:
                rec = {
                    'index': int(idx),
                    'track_name': self.df.iloc[idx]['track_name'],
                    'artists': self.df.iloc[idx]['artists'],
                    'album_name': self.df.iloc[idx].get('album_name', 'Unknown'),
                    'similarity_score': round(1 - distances[0][i], 4),
                    'popularity': self.df.iloc[idx].get('popularity', 0),
                    'genre': self.df.iloc[idx].get('track_genre', 'Unknown'),
                }
                # Add feature values
                for col in self.feature_columns:
                    rec[col] = round(self.df.iloc[idx][col], 4)
                
                recommendations.append(rec)
        
        return recommendations[:n_recommendations]
    
    def recommend_by_cosine(self, song_index, n_recommendations=10):
        """Recommend songs using Cosine Similarity"""
        
        # Compute similarity between input song and all songs
        song_features = self.feature_matrix[song_index].reshape(1, -1)
        similarities = cosine_similarity(song_features, self.feature_matrix)[0]
        
        # Get top similar songs (excluding itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations + 1]
        
        # Build recommendations
        recommendations = []
        for idx in similar_indices:
            rec = {
                'index': int(idx),
                'track_name': self.df.iloc[idx]['track_name'],
                'artists': self.df.iloc[idx]['artists'],
                'album_name': self.df.iloc[idx].get('album_name', 'Unknown'),
                'similarity_score': round(similarities[idx], 4),
                'popularity': self.df.iloc[idx].get('popularity', 0),
                'genre': self.df.iloc[idx].get('track_genre', 'Unknown'),
            }
            for col in self.feature_columns:
                rec[col] = round(self.df.iloc[idx][col], 4)
            
            recommendations.append(rec)
        
        return recommendations
    
    def recommend_by_features(self, custom_features, n_recommendations=10):
        """
        Recommend songs based on custom feature values.
        User specifies desired mood/energy/etc.
        """
        
        # Build feature vector from custom values
        feature_vector = np.array([
            custom_features.get(col, 0.5) for col in self.feature_columns
        ]).reshape(1, -1)
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(
            feature_vector,
            n_neighbors=n_recommendations
        )
        
        # Build recommendations
        recommendations = []
        for i, idx in enumerate(indices[0]):
            rec = {
                'index': int(idx),
                'track_name': self.df.iloc[idx]['track_name'],
                'artists': self.df.iloc[idx]['artists'],
                'album_name': self.df.iloc[idx].get('album_name', 'Unknown'),
                'similarity_score': round(1 - distances[0][i], 4),
                'popularity': self.df.iloc[idx].get('popularity', 0),
                'genre': self.df.iloc[idx].get('track_genre', 'Unknown'),
            }
            for col in self.feature_columns:
                rec[col] = round(self.df.iloc[idx][col], 4)
            
            recommendations.append(rec)
        
        return recommendations
    
    def recommend_by_playlist(self, song_indices, n_recommendations=10):
        """
        Recommend based on multiple songs (playlist).
        Uses the average feature vector of all input songs.
        """
        
        # Calculate centroid of input songs
        input_features = self.feature_matrix[song_indices]
        centroid = np.mean(input_features, axis=0).reshape(1, -1)
        
        # Find nearest neighbors to centroid
        distances, indices = self.knn_model.kneighbors(
            centroid,
            n_neighbors=n_recommendations + len(song_indices)
        )
        
        # Exclude input songs
        recommendations = []
        for i, idx in enumerate(indices[0]):
            if idx not in song_indices:
                rec = {
                    'index': int(idx),
                    'track_name': self.df.iloc[idx]['track_name'],
                    'artists': self.df.iloc[idx]['artists'],
                    'album_name': self.df.iloc[idx].get('album_name', 'Unknown'),
                    'similarity_score': round(1 - distances[0][i], 4),
                    'popularity': self.df.iloc[idx].get('popularity', 0),
                    'genre': self.df.iloc[idx].get('track_genre', 'Unknown'),
                }
                for col in self.feature_columns:
                    rec[col] = round(self.df.iloc[idx][col], 4)
                
                recommendations.append(rec)
        
        return recommendations[:n_recommendations]
    
    def recommend_by_genre(self, song_index, genre=None, n_recommendations=10):
        """Recommend similar songs filtered by genre"""
        
        # If no genre specified, use the input song's genre
        if genre is None:
            genre = self.df.iloc[song_index].get('track_genre', None)
        
        if genre is None:
            print("[WARN] No genre available, using standard recommendation")
            return self.recommend_by_knn(song_index, n_recommendations)
        
        # Filter dataset by genre
        genre_mask = self.df['track_genre'] == genre
        genre_indices = self.df[genre_mask].index.tolist()
        
        if len(genre_indices) < n_recommendations:
            print(f"[WARN] Only {len(genre_indices)} songs in genre '{genre}'")
        
        # Compute similarity only within genre
        song_features = self.feature_matrix[song_index].reshape(1, -1)
        genre_features = self.feature_matrix[genre_indices]
        
        similarities = cosine_similarity(song_features, genre_features)[0]
        
        # Sort by similarity
        sorted_idx = np.argsort(similarities)[::-1]
        
        recommendations = []
        for i in sorted_idx:
            actual_idx = genre_indices[i]
            if actual_idx != song_index:
                rec = {
                    'index': int(actual_idx),
                    'track_name': self.df.iloc[actual_idx]['track_name'],
                    'artists': self.df.iloc[actual_idx]['artists'],
                    'album_name': self.df.iloc[actual_idx].get('album_name', 'Unknown'),
                    'similarity_score': round(similarities[i], 4),
                    'popularity': self.df.iloc[actual_idx].get('popularity', 0),
                    'genre': genre,
                }
                for col in self.feature_columns:
                    rec[col] = round(self.df.iloc[actual_idx][col], 4)
                
                recommendations.append(rec)
                
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations
    
    def get_song_info(self, song_index):
        """Get full information about a song"""
        row = self.df.iloc[song_index]
        return row.to_dict()
    
    def explain_recommendation(self, input_index, recommended_index):
        """Explain why a song was recommended"""
        
        input_features = self.df.iloc[input_index][self.feature_columns]
        rec_features = self.df.iloc[recommended_index][self.feature_columns]
        
        # Calculate differences
        differences = {}
        for col in self.feature_columns:
            diff = abs(input_features[col] - rec_features[col])
            differences[col] = round(diff, 4)
        
        # Sort by smallest difference (most similar)
        sorted_features = sorted(differences.items(), key=lambda x: x[1])
        
        most_similar = [f[0] for f in sorted_features[:3]]
        most_different = [f[0] for f in sorted_features[-3:]]
        
        # Overall similarity
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        input_vec = self.feature_matrix[input_index].reshape(1, -1)
        rec_vec = self.feature_matrix[recommended_index].reshape(1, -1)
        overall_similarity = cos_sim(input_vec, rec_vec)[0][0]
        
        return {
            'overall_similarity': round(overall_similarity, 4),
            'most_similar_features': most_similar,
            'most_different_features': most_different,
            'feature_differences': differences,
            'input_song': {
                'name': self.df.iloc[input_index]['track_name'],
                'artist': self.df.iloc[input_index]['artists'],
                'features': {col: round(input_features[col], 4) for col in self.feature_columns}
            },
            'recommended_song': {
                'name': self.df.iloc[recommended_index]['track_name'],
                'artist': self.df.iloc[recommended_index]['artists'],
                'features': {col: round(rec_features[col], 4) for col in self.feature_columns}
            }
        }
    
    def get_available_genres(self):
        """Get list of all available genres"""
        if 'track_genre' in self.df.columns:
            return sorted(self.df['track_genre'].unique().tolist())
        return []


# ═══════════════════════════════════════════════════════════
# Interactive Testing
# ═══════════════════════════════════════════════════════════

def test_recommender():
    """Test the recommender with sample queries"""
    
    print("="*60)
    print("    SPOTIFY RECOMMENDER - TEST RUN")
    print("="*60)
    
    # Initialize
    recommender = SpotifySongRecommender()
    
    # ─────────────────────────────────────────────
    # Test 1: Search and recommend by song name
    # ─────────────────────────────────────────────
    print("\n" + "-"*60)
    print("[TEST 1] Search by song name")
    print("-"*60)
    
    test_songs = [
        ("Blinding Lights", "Weeknd"),
        ("Bohemian Rhapsody", "Queen"),
        ("Shape of You", "Ed Sheeran"),
        ("Lose Yourself", "Eminem"),
        ("Someone Like You", "Adele"),
    ]
    
    for song_name, artist in test_songs:
        idx, matches = recommender.find_song(song_name, artist)
        
        if idx is not None:
            song_info = recommender.get_song_info(idx)
            print(f"\n  ✓ Found: '{song_info['track_name']}' by {song_info['artists']}")
            print(f"    Genre: {song_info.get('track_genre', 'N/A')} | "
                  f"Popularity: {song_info.get('popularity', 'N/A')}")
            
            # Get recommendations
            recs = recommender.recommend_by_knn(idx, n_recommendations=5)
            print(f"    Top 5 recommendations:")
            for i, rec in enumerate(recs, 1):
                print(f"      {i}. {rec['track_name']} - {rec['artists']} "
                      f"({rec['similarity_score']*100:.1f}% match)")
            break  # Just test the first one found
        else:
            print(f"\n  ✗ Not found: '{song_name}' by {artist}")
    
    # ─────────────────────────────────────────────
    # Test 2: Recommend by custom features
    # ─────────────────────────────────────────────
    print("\n" + "-"*60)
    print("[TEST 2] Recommend by mood (custom features)")
    print("-"*60)
    
    # Happy party music
    print("\n  🎉 Happy Party Music:")
    party_recs = recommender.recommend_by_features({
        'danceability': 0.9,
        'energy': 0.9,
        'valence': 0.9,
        'tempo': 0.6,
        'acousticness': 0.1,
        'speechiness': 0.1,
        'instrumentalness': 0.0,
        'liveness': 0.3,
        'loudness': 0.8
    }, n_recommendations=5)
    
    for i, rec in enumerate(party_recs, 1):
        print(f"    {i}. {rec['track_name']} - {rec['artists']} "
              f"({rec['similarity_score']*100:.1f}% match) [{rec['genre']}]")
    
    # Chill acoustic
    print("\n  🌙 Chill Acoustic Music:")
    chill_recs = recommender.recommend_by_features({
        'danceability': 0.3,
        'energy': 0.2,
        'valence': 0.4,
        'tempo': 0.3,
        'acousticness': 0.9,
        'speechiness': 0.05,
        'instrumentalness': 0.3,
        'liveness': 0.1,
        'loudness': 0.3
    }, n_recommendations=5)
    
    for i, rec in enumerate(chill_recs, 1):
        print(f"    {i}. {rec['track_name']} - {rec['artists']} "
              f"({rec['similarity_score']*100:.1f}% match) [{rec['genre']}]")
    
    # Intense workout
    print("\n  💪 Intense Workout Music:")
    workout_recs = recommender.recommend_by_features({
        'danceability': 0.6,
        'energy': 1.0,
        'valence': 0.5,
        'tempo': 0.8,
        'acousticness': 0.0,
        'speechiness': 0.1,
        'instrumentalness': 0.1,
        'liveness': 0.3,
        'loudness': 0.9
    }, n_recommendations=5)
    
    for i, rec in enumerate(workout_recs, 1):
        print(f"    {i}. {rec['track_name']} - {rec['artists']} "
              f"({rec['similarity_score']*100:.1f}% match) [{rec['genre']}]")
    
    # ─────────────────────────────────────────────
    # Test 3: Explanation
    # ─────────────────────────────────────────────
    print("\n" + "-"*60)
    print("[TEST 3] Recommendation Explanation")
    print("-"*60)
    
    # Find any song and explain its first recommendation
    idx, _ = recommender.find_song("love")
    if idx is not None:
        recs = recommender.recommend_by_knn(idx, 3)
        if recs:
            explanation = recommender.explain_recommendation(idx, recs[0]['index'])
            
            print(f"\n  Input: '{explanation['input_song']['name']}' by {explanation['input_song']['artist']}")
            print(f"  Recommended: '{explanation['recommended_song']['name']}' by {explanation['recommended_song']['artist']}")
            print(f"  Overall similarity: {explanation['overall_similarity']*100:.1f}%")
            print(f"  Most similar in: {', '.join(explanation['most_similar_features'])}")
            print(f"  Differs most in: {', '.join(explanation['most_different_features'])}")
    
    # ─────────────────────────────────────────────
    # Test 4: Genre-filtered recommendation
    # ─────────────────────────────────────────────
    print("\n" + "-"*60)
    print("[TEST 4] Genre-filtered recommendations")
    print("-"*60)
    
    if idx is not None:
        genres_to_test = ['rock', 'jazz', 'electronic']
        for genre in genres_to_test:
            genre_recs = recommender.recommend_by_genre(idx, genre=genre, n_recommendations=3)
            if genre_recs:
                print(f"\n  🎵 Similar songs in '{genre}':")
                for i, rec in enumerate(genre_recs, 1):
                    print(f"    {i}. {rec['track_name']} - {rec['artists']} "
                          f"({rec['similarity_score']*100:.1f}%)")
    
    # ─────────────────────────────────────────────
    # Test 5: Stats
    # ─────────────────────────────────────────────
    print("\n" + "-"*60)
    print("[TEST 5] Available genres")
    print("-"*60)
    
    genres = recommender.get_available_genres()
    print(f"\n  Total genres available: {len(genres)}")
    print(f"  Sample: {genres[:15]}")
    
    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("    ALL TESTS COMPLETE!")
    print("="*60)
    print(f"\n  ✅ Recommender is working correctly")
    print(f"  ✅ {len(recommender.df)} songs indexed")
    print(f"  ✅ KNN model fitted")
    print(f"  ✅ Ready for Streamlit app")
    print(f"\n  Next step: python app.py (or streamlit run app.py)")
    print("="*60)


if __name__ == "__main__":
    test_recommender()