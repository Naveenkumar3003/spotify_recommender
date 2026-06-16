"""
Spotify Song Recommender - Data Collection
Uses Kaggle dataset (no API needed, no SSL issues).
Optionally connects to Spotify API if available.
"""

import os
import sys
import pandas as pd
import numpy as np
import zipfile
import urllib.request

# Create data directory
os.makedirs("data", exist_ok=True)


# ═══════════════════════════════════════════════════════════
# Option 1: Download from Kaggle (RECOMMENDED)
# ═══════════════════════════════════════════════════════════

def download_kaggle_dataset():
    """
    Download Spotify dataset from Kaggle.
    
    MANUAL STEPS (one-time):
    1. Go to https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
    2. Download the dataset (click "Download" button)
    3. Place the CSV file in: data/dataset.csv
    
    Alternative datasets:
    - https://www.kaggle.com/datasets/lehaknarnauli/spotify-datasets
    - https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset
    """
    
    print("[INFO] Kaggle dataset instructions:")
    print("="*60)
    print()
    print("  1. Go to this URL in your browser:")
    print("     https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
    print()
    print("  2. Click 'Download' (you need a free Kaggle account)")
    print()
    print("  3. Extract the ZIP file")
    print()
    print("  4. Place the CSV file here:")
    print(f"     {os.path.abspath('data/dataset.csv')}")
    print()
    print("  5. Re-run this script")
    print()
    print("="*60)


def load_kaggle_dataset(filepath=None):
    """
    Load and clean the Kaggle Spotify dataset.
    Handles multiple common Kaggle dataset formats.
    """
    
    # Try multiple possible file locations
    possible_paths = [
        filepath,
        "data/dataset.csv",
        "data/spotify_tracks.csv",
        "data/tracks.csv",
        "data/spotify-tracks-dataset/dataset.csv",
        "data/spotify_dataset.csv",
    ]
    
    # Find the first existing file
    actual_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            actual_path = path
            break
    
    if actual_path is None:
        print("[ERROR] No dataset file found!")
        print("[INFO] Looked in these locations:")
        for path in possible_paths:
            if path:
                print(f"  - {path}")
        print()
        download_kaggle_dataset()
        return None
    
    print(f"[INFO] Loading dataset from: {actual_path}")
    df = pd.read_csv(actual_path)
    print(f"[INFO] Raw dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"[INFO] Columns found: {list(df.columns)}")
    
    return df


def clean_kaggle_dataset(df):
    """
    Clean and standardize the Kaggle dataset.
    Handles different column naming conventions.
    """
    
    if df is None:
        return None
    
    # ─────────────────────────────────────────────
    # Standardize column names
    # ─────────────────────────────────────────────
    
    # Common column name mappings across different Kaggle datasets
    column_mappings = {
        # Track info
        'track_id': ['track_id', 'id', 'song_id'],
        'track_name': ['track_name', 'name', 'song_name', 'title'],
        'artists': ['artists', 'artist', 'artist_name', 'artist_names'],
        'album_name': ['album_name', 'album', 'album_title'],
        'popularity': ['popularity', 'pop'],
        
        # Audio features
        'danceability': ['danceability'],
        'energy': ['energy'],
        'key': ['key'],
        'loudness': ['loudness'],
        'mode': ['mode'],
        'speechiness': ['speechiness'],
        'acousticness': ['acousticness'],
        'instrumentalness': ['instrumentalness'],
        'liveness': ['liveness'],
        'valence': ['valence'],
        'tempo': ['tempo'],
        'duration_ms': ['duration_ms', 'duration'],
        'time_signature': ['time_signature'],
        
        # Optional
        'track_genre': ['track_genre', 'genre', 'playlist_genre'],
    }
    
    # Rename columns to standard names
    rename_map = {}
    for standard_name, possible_names in column_mappings.items():
        for possible in possible_names:
            if possible in df.columns and standard_name not in df.columns:
                rename_map[possible] = standard_name
                break
    
    df = df.rename(columns=rename_map)
    print(f"[INFO] Renamed columns: {rename_map}")
    
    # ─────────────────────────────────────────────
    # Generate track_id if missing
    # ─────────────────────────────────────────────
    if 'track_id' not in df.columns:
        df['track_id'] = [f"track_{i}" for i in range(len(df))]
        print("[INFO] Generated track_id column")
    
    # ─────────────────────────────────────────────
    # Clean data
    # ─────────────────────────────────────────────
    
    # Required audio feature columns
    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo'
    ]
    
    # Check which features exist
    existing_features = [f for f in audio_features if f in df.columns]
    missing_features = [f for f in audio_features if f not in df.columns]
    
    if missing_features:
        print(f"[WARN] Missing audio features: {missing_features}")
    
    # Drop rows with null audio features
    before_count = len(df)
    df = df.dropna(subset=existing_features)
    dropped = before_count - len(df)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows with null audio features")
    
    # Remove duplicates
    if 'track_id' in df.columns:
        before_count = len(df)
        df = df.drop_duplicates(subset=['track_id'])
        dropped = before_count - len(df)
        if dropped > 0:
            print(f"[INFO] Removed {dropped} duplicate tracks")
    
    # Remove rows with invalid tempo
    if 'tempo' in df.columns:
        df = df[(df['tempo'] > 30) & (df['tempo'] < 300)]
    
    # Remove very short/long songs
    if 'duration_ms' in df.columns:
        df = df[(df['duration_ms'] > 30000) & (df['duration_ms'] < 600000)]
    
    # Clean artist names
    if 'artists' in df.columns:
        # Some datasets have artists as "['artist1', 'artist2']"
        df['artists'] = df['artists'].astype(str)
        df['artists'] = df['artists'].str.replace(r"[\[\]']", "", regex=True)
        df['artists'] = df['artists'].str.split(',').str[0].str.strip()
    
    # Clean track names
    if 'track_name' in df.columns:
        df['track_name'] = df['track_name'].astype(str)
        df = df[df['track_name'] != 'nan']
        df = df[df['track_name'].str.len() > 0]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"[INFO] Cleaned dataset: {len(df)} tracks")
    
    return df


# ═══════════════════════════════════════════════════════════
# Option 2: Generate Sample Dataset (For Testing)
# ═══════════════════════════════════════════════════════════

def generate_sample_dataset(n_songs=2000):
    """
    Generate a realistic sample dataset for testing.
    Use this if you can't download from Kaggle either.
    """
    
    print(f"[INFO] Generating sample dataset with {n_songs} songs...")
    
    np.random.seed(42)
    
    # Realistic song names and artists
    genres = {
        'pop': {
            'artists': ['Taylor Swift', 'Ed Sheeran', 'Dua Lipa', 'The Weeknd', 
                       'Harry Styles', 'Billie Eilish', 'Post Malone', 'Ariana Grande',
                       'Bruno Mars', 'Adele'],
            'danceability': (0.6, 0.15),
            'energy': (0.65, 0.15),
            'valence': (0.55, 0.2),
            'tempo': (120, 20),
            'acousticness': (0.15, 0.1),
            'speechiness': (0.08, 0.05),
            'instrumentalness': (0.01, 0.02),
            'liveness': (0.15, 0.1),
            'loudness': (-6, 2),
        },
        'rock': {
            'artists': ['Foo Fighters', 'Arctic Monkeys', 'Imagine Dragons', 
                       'Coldplay', 'Linkin Park', 'Green Day', 'Nirvana',
                       'Red Hot Chili Peppers', 'Muse', 'The Killers'],
            'danceability': (0.45, 0.15),
            'energy': (0.8, 0.12),
            'valence': (0.45, 0.2),
            'tempo': (130, 25),
            'acousticness': (0.1, 0.1),
            'speechiness': (0.05, 0.03),
            'instrumentalness': (0.05, 0.08),
            'liveness': (0.2, 0.15),
            'loudness': (-5, 2),
        },
        'hiphop': {
            'artists': ['Drake', 'Kendrick Lamar', 'Travis Scott', 'J. Cole',
                       'Kanye West', 'Eminem', 'Lil Nas X', 'Tyler The Creator',
                       'Juice WRLD', '21 Savage'],
            'danceability': (0.75, 0.12),
            'energy': (0.65, 0.15),
            'valence': (0.45, 0.2),
            'tempo': (140, 30),
            'acousticness': (0.1, 0.1),
            'speechiness': (0.2, 0.12),
            'instrumentalness': (0.01, 0.02),
            'liveness': (0.15, 0.1),
            'loudness': (-6, 2),
        },
        'electronic': {
            'artists': ['Calvin Harris', 'Marshmello', 'Avicii', 'Deadmau5',
                       'Skrillex', 'Daft Punk', 'Tiesto', 'Martin Garrix',
                       'Kygo', 'Zedd'],
            'danceability': (0.7, 0.12),
            'energy': (0.8, 0.1),
            'valence': (0.5, 0.2),
            'tempo': (128, 15),
            'acousticness': (0.05, 0.05),
            'speechiness': (0.07, 0.05),
            'instrumentalness': (0.3, 0.25),
            'liveness': (0.12, 0.08),
            'loudness': (-5, 2),
        },
        'acoustic': {
            'artists': ['John Mayer', 'Jack Johnson', 'Bon Iver', 'Iron & Wine',
                       'Damien Rice', 'Nick Drake', 'James Taylor', 'Cat Stevens',
                       'Fleet Foxes', 'Sufjan Stevens'],
            'danceability': (0.4, 0.15),
            'energy': (0.3, 0.15),
            'valence': (0.4, 0.2),
            'tempo': (100, 20),
            'acousticness': (0.8, 0.12),
            'speechiness': (0.04, 0.02),
            'instrumentalness': (0.1, 0.15),
            'liveness': (0.12, 0.08),
            'loudness': (-12, 4),
        },
        'jazz': {
            'artists': ['Miles Davis', 'John Coltrane', 'Bill Evans', 'Thelonious Monk',
                       'Dave Brubeck', 'Chet Baker', 'Herbie Hancock', 'Charles Mingus',
                       'Kamasi Washington', 'Robert Glasper'],
            'danceability': (0.45, 0.15),
            'energy': (0.35, 0.2),
            'valence': (0.5, 0.2),
            'tempo': (120, 30),
            'acousticness': (0.7, 0.2),
            'speechiness': (0.04, 0.03),
            'instrumentalness': (0.5, 0.3),
            'liveness': (0.2, 0.15),
            'loudness': (-14, 5),
        },
        'classical': {
            'artists': ['Ludwig van Beethoven', 'Mozart', 'Bach', 'Chopin',
                       'Debussy', 'Vivaldi', 'Tchaikovsky', 'Brahms',
                       'Schubert', 'Liszt'],
            'danceability': (0.25, 0.12),
            'energy': (0.25, 0.2),
            'valence': (0.3, 0.2),
            'tempo': (100, 30),
            'acousticness': (0.9, 0.08),
            'speechiness': (0.04, 0.02),
            'instrumentalness': (0.85, 0.1),
            'liveness': (0.1, 0.08),
            'loudness': (-20, 6),
        },
        'latin': {
            'artists': ['Bad Bunny', 'J Balvin', 'Shakira', 'Daddy Yankee',
                       'Ozuna', 'Maluma', 'Rosalia', 'Karol G',
                       'Rauw Alejandro', 'Nicky Jam'],
            'danceability': (0.75, 0.1),
            'energy': (0.7, 0.12),
            'valence': (0.65, 0.15),
            'tempo': (100, 20),
            'acousticness': (0.15, 0.12),
            'speechiness': (0.1, 0.08),
            'instrumentalness': (0.02, 0.03),
            'liveness': (0.15, 0.1),
            'loudness': (-6, 2),
        },
    }
    
    # Song name templates
    song_templates = [
        "Midnight {}", "Golden {}", "Electric {}", "Fading {}",
        "Lost in {}", "Dancing {}", "Broken {}", "Neon {}",
        "Silent {}", "Burning {}", "Crystal {}", "Shadow {}",
        "Endless {}", "Velvet {}", "Thunder {}", "Ocean {}",
        "Starlight {}", "Echoes of {}", "Beyond {}", "Falling {}",
    ]
    
    song_words = [
        "Dreams", "Love", "Fire", "Rain", "Night", "Heart",
        "Stars", "Waves", "Light", "Time", "Soul", "Eyes",
        "Roads", "Sky", "Wind", "Hope", "Memory", "Silence",
        "Paradise", "Gravity", "Horizon", "Desire", "Freedom",
    ]
    
    # Generate songs
    songs = []
    genre_names = list(genres.keys())
    songs_per_genre = n_songs // len(genre_names)
    
    for genre_name, genre_data in genres.items():
        for i in range(songs_per_genre):
            artist = np.random.choice(genre_data['artists'])
            
            # Generate song name
            template = np.random.choice(song_templates)
            word = np.random.choice(song_words)
            song_name = template.format(word)
            
            # Generate features with genre-specific distributions
            def clip_feature(value, min_val=0, max_val=1):
                return max(min_val, min(max_val, value))
            
            song = {
                'track_id': f"{genre_name}_{i:04d}",
                'track_name': song_name,
                'artists': artist,
                'album_name': f"Album {np.random.randint(1, 20)}",
                'track_genre': genre_name,
                'popularity': int(np.clip(np.random.normal(50, 20), 1, 100)),
                'danceability': clip_feature(np.random.normal(*genre_data['danceability'])),
                'energy': clip_feature(np.random.normal(*genre_data['energy'])),
                'valence': clip_feature(np.random.normal(*genre_data['valence'])),
                'tempo': max(40, np.random.normal(*genre_data['tempo'])),
                'acousticness': clip_feature(np.random.normal(*genre_data['acousticness'])),
                'speechiness': clip_feature(np.random.normal(*genre_data['speechiness'])),
                'instrumentalness': clip_feature(np.random.normal(*genre_data['instrumentalness'])),
                'liveness': clip_feature(np.random.normal(*genre_data['liveness'])),
                'loudness': np.clip(np.random.normal(*genre_data['loudness']), -60, 0),
                'key': np.random.randint(0, 12),
                'mode': np.random.randint(0, 2),
                'duration_ms': int(np.random.normal(210000, 40000)),
                'time_signature': np.random.choice([3, 4, 5], p=[0.1, 0.85, 0.05]),
            }
            
            songs.append(song)
    
    df = pd.DataFrame(songs)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[INFO] Generated {len(df)} sample songs across {len(genres)} genres")
    
    return df


# ═══════════════════════════════════════════════════════════
# Data Collection Pipeline
# ═══════════════════════════════════════════════════════════

def collect_dataset():
    """
    Main data collection function.
    Tries sources in order:
    1. Kaggle dataset (if file exists)
    2. Generated sample dataset (fallback)
    """
    
    print("[INFO] Starting data collection...")
    print("-"*60)
    
    # ─────────────────────────────────────────────
    # Try 1: Load Kaggle dataset
    # ─────────────────────────────────────────────
    print("\n[ATTEMPT 1] Looking for Kaggle dataset...")
    
    df = load_kaggle_dataset()
    
    if df is not None:
        print("[INFO] Kaggle dataset found! Cleaning...")
        df = clean_kaggle_dataset(df)
        
        if df is not None and len(df) > 100:
            output_path = "data/spotify_dataset.csv"
            df.to_csv(output_path, index=False)
            print(f"\n[SUCCESS] Dataset saved to: {output_path}")
            print(f"[INFO] Total songs: {len(df)}")
            return df
    
    # ─────────────────────────────────────────────
    # Try 2: Generate sample dataset
    # ─────────────────────────────────────────────
    print("\n[ATTEMPT 2] Generating sample dataset...")
    print("[INFO] This creates realistic synthetic data for development")
    print("[INFO] You can replace with real Kaggle data later\n")
    
    df = generate_sample_dataset(n_songs=2000)
    
    output_path = "data/spotify_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Sample dataset saved to: {output_path}")
    print(f"[INFO] Total songs: {len(df)}")
    print(f"[INFO] Genres: {df['track_genre'].nunique()}")
    
    return df


# ═══════════════════════════════════════════════════════════
# Data Verification
# ═══════════════════════════════════════════════════════════

def verify_collected_data(filepath="data/spotify_dataset.csv"):
    """
    Verify the collected dataset is complete and usable.
    """
    
    print("\n" + "="*60)
    print("        DATA COLLECTION VERIFICATION REPORT")
    print("="*60)
    
    # ─────────────────────────────────────────────
    # CHECK 1: File exists
    # ─────────────────────────────────────────────
    print("\n[CHECK 1] File Existence")
    if not os.path.exists(filepath):
        print(f"  ✗ FAILED: File '{filepath}' not found!")
        return False
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  ✓ File exists: {filepath}")
    print(f"  ✓ File size: {file_size:.2f} MB")
    
    # ─────────────────────────────────────────────
    # CHECK 2: Load and basic shape
    # ─────────────────────────────────────────────
    print("\n[CHECK 2] Dataset Shape")
    df = pd.read_csv(filepath)
    print(f"  ✓ Rows: {len(df)}")
    print(f"  ✓ Columns: {len(df.columns)}")
    print(f"  ✓ Column names: {list(df.columns)}")
    
    if len(df) < 100:
        print(f"  ⚠ WARNING: Only {len(df)} tracks. Aim for 1000+")
    elif len(df) < 1000:
        print(f"  ⚠ ACCEPTABLE: {len(df)} tracks")
    else:
        print(f"  ✓ GOOD: {len(df)} tracks is sufficient")
    
    # ─────────────────────────────────────────────
    # CHECK 3: Required columns
    # ─────────────────────────────────────────────
    print("\n[CHECK 3] Required Columns")
    
    required_columns = [
        'track_id', 'track_name', 'artists',
        'danceability', 'energy', 'valence', 'tempo',
        'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness'
    ]
    
    missing_columns = []
    for col in required_columns:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            missing_columns.append(col)
            print(f"  ✗ {col} — MISSING")
    
    if not missing_columns:
        print(f"\n  ✓ All required columns present!")
    
    # ─────────────────────────────────────────────
    # CHECK 4: Null values
    # ─────────────────────────────────────────────
    print("\n[CHECK 4] Missing Values")
    
    audio_features = [
        'danceability', 'energy', 'valence', 'tempo',
        'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness'
    ]
    
    existing_features = [f for f in audio_features if f in df.columns]
    null_report = df[existing_features].isnull().sum()
    total_nulls = null_report.sum()
    
    if total_nulls == 0:
        print(f"  ✓ No missing values in audio features!")
    else:
        print(f"  ⚠ Found {total_nulls} missing values:")
        for col, count in null_report[null_report > 0].items():
            print(f"    - {col}: {count} nulls")
    
    # ─────────────────────────────────────────────
    # CHECK 5: Value ranges
    # ─────────────────────────────────────────────
    print("\n[CHECK 5] Value Ranges")
    
    expected_ranges = {
        'danceability': (0, 1),
        'energy': (0, 1),
        'valence': (0, 1),
        'speechiness': (0, 1),
        'acousticness': (0, 1),
        'instrumentalness': (0, 1),
        'liveness': (0, 1),
        'tempo': (30, 300),
        'loudness': (-60, 5)
    }
    
    range_issues = []
    for col, (exp_min, exp_max) in expected_ranges.items():
        if col not in df.columns:
            continue
        actual_min = df[col].min()
        actual_max = df[col].max()
        in_range = actual_min >= exp_min - 0.01 and actual_max <= exp_max + 0.01
        
        if in_range:
            print(f"  ✓ {col:20s}: [{actual_min:.3f}, {actual_max:.3f}]")
        else:
            range_issues.append(col)
            print(f"  ✗ {col:20s}: [{actual_min:.3f}, {actual_max:.3f}] ⚠ OUT OF RANGE")
    
    # ─────────────────────────────────────────────
    # CHECK 6: Duplicates
    # ─────────────────────────────────────────────
    print("\n[CHECK 6] Duplicates")
    
    if 'track_id' in df.columns:
        duplicates = df['track_id'].duplicated().sum()
        if duplicates > 0:
            print(f"  ⚠ {duplicates} duplicate track_ids")
        else:
            print(f"  ✓ No duplicate tracks!")
    
    # ─────────────────────────────────────────────
    # CHECK 7: Feature distributions
    # ─────────────────────────────────────────────
    print("\n[CHECK 7] Feature Distributions")
    
    low_variance_cols = []
    for col in existing_features:
        mean = df[col].mean()
        std = df[col].std()
        print(f"  {col:20s}: mean={mean:.3f}, std={std:.3f}")
        if std < 0.01:
            low_variance_cols.append(col)
    
    if low_variance_cols:
        print(f"\n  ⚠ Low variance: {low_variance_cols}")
    else:
        print(f"\n  ✓ All features have healthy variance")
    
    # ─────────────────────────────────────────────
    # CHECK 8: Sample data
    # ─────────────────────────────────────────────
    print("\n[CHECK 8] Sample Data (First 5 Rows)")
    print("-"*60)
    
    display_cols = ['track_name', 'artists', 'danceability', 'energy', 'valence']
    existing_display = [c for c in display_cols if c in df.columns]
    print(df[existing_display].head().to_string(index=False))
    
    # ─────────────────────────────────────────────
    # CHECK 9: Diversity
    # ─────────────────────────────────────────────
    print("\n\n[CHECK 9] Dataset Diversity")
    
    if 'artists' in df.columns:
        unique_artists = df['artists'].nunique()
        print(f"  Unique artists: {unique_artists}")
        print(f"  Songs per artist (avg): {len(df)/unique_artists:.1f}")
        
        top_artists = df['artists'].value_counts().head(5)
        print(f"\n  Top 5 artists:")
        for artist, count in top_artists.items():
            print(f"    - {artist}: {count} songs")
    
    if 'track_genre' in df.columns:
        print(f"\n  Genres: {df['track_genre'].nunique()}")
        genre_dist = df['track_genre'].value_counts()
        print(f"  Genre distribution:")
        for genre, count in genre_dist.items():
            print(f"    - {genre}: {count} songs")
    
    # ─────────────────────────────────────────────
    # CHECK 10: Popularity
    # ─────────────────────────────────────────────
    if 'popularity' in df.columns:
        print(f"\n[CHECK 10] Popularity Distribution")
        print(f"  Min: {df['popularity'].min()}")
        print(f"  Max: {df['popularity'].max()}")
        print(f"  Mean: {df['popularity'].mean():.1f}")
        print(f"  Median: {df['popularity'].median():.1f}")
    
    # ─────────────────────────────────────────────
    # FINAL VERDICT
    # ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("        FINAL VERDICT")
    print("="*60)
    
    issues = []
    if len(df) < 100:
        issues.append("Too few tracks (need 100+)")
    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")
    if total_nulls > len(df) * 0.1:
        issues.append("Too many null values (>10%)")
    if range_issues:
        issues.append(f"Value range issues: {range_issues}")
    if low_variance_cols:
        issues.append(f"Low variance features: {low_variance_cols}")
    
    if not issues:
        print("\n  ✅ ALL CHECKS PASSED!")
        print("  → Dataset is ready for preprocessing and model training")
        print(f"  → {len(df)} tracks with {len(existing_features)} audio features")
        print(f"\n  Next step: python preprocessing.py")
    else:
        print(f"\n  ⚠ {len(issues)} ISSUE(S) FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print("\n  → Fix these issues before proceeding")
    
    print("="*60)
    
    return len(issues) == 0


# ═══════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("    SPOTIFY SONG RECOMMENDER - DATA COLLECTION")
    print("="*60)
    
    # Step 1: Collect data
    print("\n[STEP 1] Collecting data...")
    print("-"*60)
    dataset = collect_dataset()
    
    # Step 2: Verify
    print("\n[STEP 2] Verifying collected data...")
    print("-"*60)
    is_valid = verify_collected_data("data/spotify_dataset.csv")
    
    # Final message
    if is_valid:
        print("\n✅ SUCCESS! Data collection complete.")
        print("   Next: python preprocessing.py")
    else:
        print("\n❌ Issues found. Check the report above.")