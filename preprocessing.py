"""
Spotify Song Recommender - Data Preprocessing
Cleans, normalizes, and prepares data for the recommendation model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

warnings.filterwarnings('ignore')

# Create output directory
os.makedirs("data", exist_ok=True)


class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            'danceability', 'energy', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'loudness'
        ]
    
    def load_data(self, filepath="data/spotify_dataset.csv"):
        """Load the collected dataset"""
        print(f"[INFO] Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded {len(df)} tracks")
        return df
    
    def clean_data(self, df):
        """Remove unwanted columns and clean data"""
        
        print("\n[STEP 1] Cleaning data...")
        
        # Drop unnamed index column if exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            print("  ✓ Dropped 'Unnamed: 0' column")
        
        # Drop rows with missing features
        before = len(df)
        df = df.dropna(subset=self.feature_columns)
        dropped = before - len(df)
        if dropped > 0:
            print(f"  ✓ Dropped {dropped} rows with missing features")
        else:
            print(f"  ✓ No missing values in features")
        
        # Drop duplicates by track_id
        before = len(df)
        df = df.drop_duplicates(subset=['track_id'])
        dropped = before - len(df)
        if dropped > 0:
            print(f"  ✓ Removed {dropped} duplicate tracks")
        else:
            print(f"  ✓ No duplicates found")
        
        # Remove outliers in tempo
        before = len(df)
        df = df[(df['tempo'] > 40) & (df['tempo'] < 250)]
        dropped = before - len(df)
        print(f"  ✓ Removed {dropped} tracks with extreme tempo")
        
        # Remove very short or very long songs
        if 'duration_ms' in df.columns:
            before = len(df)
            df = df[(df['duration_ms'] > 30000) & (df['duration_ms'] < 600000)]
            dropped = before - len(df)
            print(f"  ✓ Removed {dropped} tracks with extreme duration")
        
        # Clean string columns
        if 'track_name' in df.columns:
            df['track_name'] = df['track_name'].astype(str).str.strip()
            df = df[df['track_name'] != 'nan']
            df = df[df['track_name'].str.len() > 0]
        
        if 'artists' in df.columns:
            df['artists'] = df['artists'].astype(str).str.strip()
            df = df[df['artists'] != 'nan']
        
        df = df.reset_index(drop=True)
        print(f"  ✓ Final cleaned size: {len(df)} tracks")
        
        return df
    
    def add_derived_features(self, df):
        """Create additional useful features"""
        
        print("\n[STEP 2] Adding derived features...")
        
        df = df.copy()
        
        # Mood score (happy + energetic)
        df['mood_score'] = (df['valence'] + df['energy']) / 2
        print("  ✓ Added 'mood_score' (valence + energy)")
        
        # Acoustic vs Electronic spectrum
        df['acoustic_electronic'] = df['acousticness'] - df['energy']
        print("  ✓ Added 'acoustic_electronic' spectrum")
        
        # Party score (danceable + energetic + happy)
        df['party_score'] = (df['danceability'] + df['energy'] + df['valence']) / 3
        print("  ✓ Added 'party_score' (dance + energy + valence)")
        
        # Vocal presence (speech + not instrumental)
        df['vocal_presence'] = (df['speechiness'] + (1 - df['instrumentalness'])) / 2
        print("  ✓ Added 'vocal_presence'")
        
        # Intensity (energy + loudness normalized)
        loudness_norm = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())
        df['intensity'] = (df['energy'] + loudness_norm) / 2
        print("  ✓ Added 'intensity' (energy + loudness)")
        
        return df
    
    def normalize_features(self, df):
        """Normalize audio features to 0-1 range"""
        
        print("\n[STEP 3] Normalizing features...")
        
        df = df.copy()
        
        # Fit and transform
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        
        print(f"  ✓ Normalized {len(self.feature_columns)} features to [0, 1] range")
        print(f"  ✓ Features normalized: {self.feature_columns}")
        
        return df
    
    def select_final_columns(self, df):
        """Select and order final columns for the model"""
        
        print("\n[STEP 4] Selecting final columns...")
        
        # Core columns to keep
        core_columns = ['track_id', 'track_name', 'artists', 'album_name', 'popularity']
        
        # Audio feature columns
        audio_columns = self.feature_columns.copy()
        
        # Derived feature columns
        derived_columns = ['mood_score', 'acoustic_electronic', 'party_score', 
                          'vocal_presence', 'intensity']
        
        # Optional columns
        optional_columns = ['track_genre', 'duration_ms', 'key', 'mode', 'time_signature']
        
        # Build final column list
        final_columns = []
        
        for col in core_columns + audio_columns + derived_columns + optional_columns:
            if col in df.columns:
                final_columns.append(col)
        
        df = df[final_columns]
        print(f"  ✓ Selected {len(final_columns)} columns")
        print(f"  ✓ Columns: {final_columns}")
        
        return df
    
    def print_summary(self, df):
        """Print preprocessing summary"""
        
        print("\n" + "="*60)
        print("        PREPROCESSING SUMMARY")
        print("="*60)
        
        print(f"\n  Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\n  Audio Features (normalized 0-1):")
        
        for col in self.feature_columns:
            if col in df.columns:
                print(f"    {col:20s}: min={df[col].min():.3f}, "
                      f"max={df[col].max():.3f}, mean={df[col].mean():.3f}")
        
        print(f"\n  Derived Features:")
        derived = ['mood_score', 'party_score', 'vocal_presence', 'intensity']
        for col in derived:
            if col in df.columns:
                print(f"    {col:20s}: min={df[col].min():.3f}, "
                      f"max={df[col].max():.3f}, mean={df[col].mean():.3f}")
        
        if 'track_genre' in df.columns:
            print(f"\n  Genres: {df['track_genre'].nunique()}")
        
        if 'artists' in df.columns:
            print(f"  Unique artists: {df['artists'].nunique()}")
        
        print(f"\n  Null values: {df.isnull().sum().sum()}")
        print("="*60)
    
    def process(self, input_path="data/spotify_dataset.csv", 
                output_path="data/spotify_processed.csv"):
        """Run the full preprocessing pipeline"""
        
        print("="*60)
        print("    SPOTIFY RECOMMENDER - DATA PREPROCESSING")
        print("="*60)
        
        # Load
        df = self.load_data(input_path)
        
        # Clean
        df = self.clean_data(df)
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Normalize
        df = self.normalize_features(df)
        
        # Select columns
        df = self.select_final_columns(df)
        
        # Summary
        self.print_summary(df)
        
        # Save
        df.to_csv(output_path, index=False)
        print(f"\n[SAVED] Processed dataset: {output_path}")
        print(f"[INFO] File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return df


# ═══════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════

def verify_preprocessing(filepath="data/spotify_processed.csv"):
    """Verify the preprocessed dataset"""
    
    print("\n" + "="*60)
    print("        PREPROCESSING VERIFICATION")
    print("="*60)
    
    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return False
    
    df = pd.read_csv(filepath)
    
    feature_columns = [
        'danceability', 'energy', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'loudness'
    ]
    
    issues = []
    
    # Check normalization
    print("\n[CHECK] Feature normalization (should be 0-1):")
    for col in feature_columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            is_normalized = min_val >= -0.01 and max_val <= 1.01
            
            status = "✓" if is_normalized else "✗"
            print(f"  {status} {col:20s}: [{min_val:.4f}, {max_val:.4f}]")
            
            if not is_normalized:
                issues.append(f"{col} not normalized")
    
    # Check no nulls
    null_count = df.isnull().sum().sum()
    print(f"\n[CHECK] Null values: {null_count}")
    if null_count > 0:
        issues.append(f"{null_count} null values found")
        print(f"  ✗ Found nulls in: {df.columns[df.isnull().any()].tolist()}")
    else:
        print(f"  ✓ No null values")
    
    # Check required columns
    required = ['track_id', 'track_name', 'artists'] + feature_columns
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        print(f"\n  ✗ Missing columns: {missing}")
    else:
        print(f"\n  ✓ All required columns present")
    
    # Final verdict
    print("\n" + "-"*60)
    if not issues:
        print("  ✅ PREPROCESSING VERIFIED! Ready for model building.")
        print(f"     Next step: python recommender.py")
    else:
        print(f"  ⚠ {len(issues)} issues found:")
        for issue in issues:
            print(f"    - {issue}")
    
    print("="*60)
    return len(issues) == 0


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = DataPreprocessor()
    df = preprocessor.process(
        input_path="data/spotify_dataset.csv",
        output_path="data/spotify_processed.csv"
    )
    
    # Verify
    verify_preprocessing("data/spotify_processed.csv")