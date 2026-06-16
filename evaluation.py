"""
Spotify Song Recommender - Evaluation
Measures recommendation quality using multiple metrics.
Run with: python evaluation.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

from recommender import SpotifySongRecommender


class RecommenderEvaluator:
    def __init__(self, recommender=None):
        """Initialize evaluator with a recommender instance"""
        
        if recommender is None:
            self.recommender = SpotifySongRecommender()
        else:
            self.recommender = recommender
        
        self.df = self.recommender.df
        self.feature_columns = self.recommender.feature_columns
        self.feature_matrix = self.recommender.feature_matrix
        self.results = {}
    
    # ═══════════════════════════════════════════════════════════
    # Metric 1: Intra-List Diversity
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_diversity(self, n_samples=200, n_recs=10):
        """
        Measure how diverse the recommendations are.
        High diversity = recommendations cover different styles.
        Low diversity = all recommendations sound the same.
        
        Score range: 0 (identical) to 1+ (very diverse)
        """
        
        print("\n[METRIC 1] Intra-List Diversity")
        print("-"*50)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        diversity_scores = []
        
        for idx in sample_indices:
            recs = self.recommender.recommend_by_knn(idx, n_recs)
            
            if len(recs) < 2:
                continue
            
            # Get feature vectors of recommendations
            rec_features = np.array([
                [r[col] for col in self.feature_columns] for r in recs
            ])
            
            # Calculate average pairwise distance between recommendations
            distances = euclidean_distances(rec_features)
            n = len(recs)
            
            # Mean of upper triangle (excluding diagonal)
            upper_triangle = distances[np.triu_indices(n, k=1)]
            diversity = upper_triangle.mean()
            diversity_scores.append(diversity)
        
        mean_diversity = np.mean(diversity_scores)
        std_diversity = np.std(diversity_scores)
        
        print(f"  Mean diversity score: {mean_diversity:.4f}")
        print(f"  Std deviation: {std_diversity:.4f}")
        print(f"  Min diversity: {np.min(diversity_scores):.4f}")
        print(f"  Max diversity: {np.max(diversity_scores):.4f}")
        
        # Interpret
        if mean_diversity > 0.3:
            print(f"  ✓ GOOD: Recommendations are diverse")
        elif mean_diversity > 0.15:
            print(f"  ⚠ MODERATE: Some diversity, could be improved")
        else:
            print(f"  ✗ LOW: Recommendations are too similar to each other")
        
        self.results['diversity'] = {
            'mean': mean_diversity,
            'std': std_diversity,
            'min': np.min(diversity_scores),
            'max': np.max(diversity_scores)
        }
        
        return mean_diversity
    
    # ═══════════════════════════════════════════════════════════
    # Metric 2: Catalog Coverage
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_coverage(self, n_samples=500, n_recs=10):
        """
        What percentage of the catalog gets recommended?
        Low coverage = popularity bias (same songs always recommended).
        High coverage = system explores the full catalog.
        
        Score range: 0% to 100%
        """
        
        print("\n[METRIC 2] Catalog Coverage")
        print("-"*50)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        recommended_songs = set()
        recommendation_counts = Counter()
        
        for idx in sample_indices:
            recs = self.recommender.recommend_by_knn(idx, n_recs)
            for r in recs:
                recommended_songs.add(r['index'])
                recommendation_counts[r['index']] += 1
        
        coverage = len(recommended_songs) / len(self.df) * 100
        
        # Gini coefficient (measure of inequality in recommendations)
        counts = list(recommendation_counts.values())
        if counts:
            sorted_counts = sorted(counts)
            n = len(sorted_counts)
            cumulative = np.cumsum(sorted_counts)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n + 1) / n
        else:
            gini = 0
        
        # Most recommended songs
        top_recommended = recommendation_counts.most_common(5)
        
        print(f"  Unique songs recommended: {len(recommended_songs):,} / {len(self.df):,}")
        print(f"  Coverage: {coverage:.2f}%")
        print(f"  Gini coefficient: {gini:.4f} (0=equal, 1=concentrated)")
        print(f"\n  Most frequently recommended:")
        for idx, count in top_recommended:
            song = self.df.iloc[idx]
            print(f"    - {song['track_name']} by {song['artists']} ({count} times)")
        
        # Interpret
        if coverage > 50:
            print(f"\n  ✓ GOOD: Wide catalog coverage")
        elif coverage > 20:
            print(f"\n  ⚠ MODERATE: Decent coverage but some bias")
        else:
            print(f"\n  ✗ LOW: Strong popularity bias detected")
        
        self.results['coverage'] = {
            'percentage': coverage,
            'unique_songs': len(recommended_songs),
            'total_songs': len(self.df),
            'gini': gini
        }
        
        return coverage
    
    # ═══════════════════════════════════════════════════════════
    # Metric 3: Genre Coherence
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_genre_coherence(self, n_samples=300, n_recs=10):
        """
        Do recommendations match the genre of the input song?
        High coherence = recommendations stay in similar genres.
        
        Score range: 0% to 100%
        """
        
        print("\n[METRIC 3] Genre Coherence")
        print("-"*50)
        
        if 'track_genre' not in self.df.columns:
            print("  ⚠ No genre column available, skipping...")
            return None
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        same_genre_ratios = []
        genre_distribution = defaultdict(list)
        
        for idx in sample_indices:
            input_genre = self.df.iloc[idx]['track_genre']
            recs = self.recommender.recommend_by_knn(idx, n_recs)
            
            if not recs:
                continue
            
            # Count how many recommendations are same genre
            same_genre = sum(1 for r in recs if r.get('genre') == input_genre)
            ratio = same_genre / len(recs)
            same_genre_ratios.append(ratio)
            genre_distribution[input_genre].append(ratio)
        
        mean_coherence = np.mean(same_genre_ratios)
        std_coherence = np.std(same_genre_ratios)
        
        print(f"  Mean genre coherence: {mean_coherence*100:.2f}%")
        print(f"  Std deviation: {std_coherence*100:.2f}%")
        
        # Best and worst genres for coherence
        genre_means = {
            genre: np.mean(ratios) 
            for genre, ratios in genre_distribution.items() 
            if len(ratios) >= 3
        }
        
        if genre_means:
            sorted_genres = sorted(genre_means.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n  Top 5 most coherent genres:")
            for genre, score in sorted_genres[:5]:
                print(f"    - {genre}: {score*100:.1f}%")
            
            print(f"\n  Bottom 5 least coherent genres:")
            for genre, score in sorted_genres[-5:]:
                print(f"    - {genre}: {score*100:.1f}%")
        
        # Interpret
        if mean_coherence > 0.5:
            print(f"\n  ✓ GOOD: Strong genre coherence")
        elif mean_coherence > 0.25:
            print(f"\n  ⚠ MODERATE: Some genre coherence")
        else:
            print(f"\n  ✗ LOW: Recommendations cross genres frequently")
            print(f"       (This may be fine — audio features transcend genres)")
        
        self.results['genre_coherence'] = {
            'mean': mean_coherence,
            'std': std_coherence,
            'by_genre': genre_means
        }
        
        return mean_coherence
    
    # ═══════════════════════════════════════════════════════════
    # Metric 4: Similarity Score Distribution
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_similarity_distribution(self, n_samples=200, n_recs=10):
        """
        Analyze the distribution of similarity scores.
        Are recommendations actually similar or just random?
        
        Compare with random baseline.
        """
        
        print("\n[METRIC 4] Similarity Score Distribution")
        print("-"*50)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        # Collect actual recommendation scores
        actual_scores = []
        for idx in sample_indices:
            recs = self.recommender.recommend_by_knn(idx, n_recs)
            for r in recs:
                actual_scores.append(r['similarity_score'])
        
        # Generate random baseline scores
        random_scores = []
        for idx in sample_indices:
            random_indices = np.random.choice(len(self.df), n_recs, replace=False)
            song_features = self.feature_matrix[idx].reshape(1, -1)
            
            for rand_idx in random_indices:
                rand_features = self.feature_matrix[rand_idx].reshape(1, -1)
                sim = cosine_similarity(song_features, rand_features)[0][0]
                random_scores.append(sim)
        
        # Statistics
        actual_mean = np.mean(actual_scores)
        actual_std = np.std(actual_scores)
        random_mean = np.mean(random_scores)
        random_std = np.std(random_scores)
        
        improvement = ((actual_mean - random_mean) / random_mean) * 100
        
        print(f"  Recommendation scores:")
        print(f"    Mean: {actual_mean:.4f}")
        print(f"    Std:  {actual_std:.4f}")
        print(f"    Min:  {np.min(actual_scores):.4f}")
        print(f"    Max:  {np.max(actual_scores):.4f}")
        
        print(f"\n  Random baseline scores:")
        print(f"    Mean: {random_mean:.4f}")
        print(f"    Std:  {random_std:.4f}")
        print(f"    Min:  {np.min(random_scores):.4f}")
        print(f"    Max:  {np.max(random_scores):.4f}")
        
        print(f"\n  Improvement over random: {improvement:.1f}%")
        
        # Interpret
        if improvement > 30:
            print(f"  ✓ GOOD: Recommendations are significantly better than random")
        elif improvement > 10:
            print(f"  ⚠ MODERATE: Some improvement over random")
        else:
            print(f"  ✗ LOW: Barely better than random recommendations")
        
        self.results['similarity_distribution'] = {
            'actual_mean': actual_mean,
            'actual_std': actual_std,
            'random_mean': random_mean,
            'random_std': random_std,
            'improvement_pct': improvement
        }
        
        return improvement
    
    # ═══════════════════════════════════════════════════════════
    # Metric 5: Method Comparison (KNN vs Cosine)
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_method_comparison(self, n_samples=100, n_recs=10):
        """
        Compare KNN and Cosine Similarity methods.
        Measure overlap and differences.
        """
        
        print("\n[METRIC 5] Method Comparison (KNN vs Cosine)")
        print("-"*50)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        overlap_scores = []
        knn_scores = []
        cosine_scores = []
        
        for idx in sample_indices:
            knn_recs = self.recommender.recommend_by_knn(idx, n_recs)
            cos_recs = self.recommender.recommend_by_cosine(idx, n_recs)
            
            knn_set = set(r['index'] for r in knn_recs)
            cos_set = set(r['index'] for r in cos_recs)
            
            # Jaccard overlap
            if knn_set or cos_set:
                overlap = len(knn_set & cos_set) / len(knn_set | cos_set)
                overlap_scores.append(overlap)
            
            # Average similarity scores
            knn_avg = np.mean([r['similarity_score'] for r in knn_recs]) if knn_recs else 0
            cos_avg = np.mean([r['similarity_score'] for r in cos_recs]) if cos_recs else 0
            knn_scores.append(knn_avg)
            cosine_scores.append(cos_avg)
        
        mean_overlap = np.mean(overlap_scores)
        mean_knn = np.mean(knn_scores)
        mean_cosine = np.mean(cosine_scores)
        
        print(f"  Jaccard overlap between methods: {mean_overlap*100:.1f}%")
        print(f"  KNN avg similarity score: {mean_knn:.4f}")
        print(f"  Cosine avg similarity score: {mean_cosine:.4f}")
        
        if mean_overlap > 0.8:
            print(f"\n  ✓ Methods strongly agree (>80% overlap)")
        elif mean_overlap > 0.5:
            print(f"\n  ⚠ Methods moderately agree (50-80% overlap)")
        else:
            print(f"\n  ℹ Methods produce different results (<50% overlap)")
            print(f"    This can be useful for diversity — offer both options")
        
        self.results['method_comparison'] = {
            'overlap': mean_overlap,
            'knn_avg_score': mean_knn,
            'cosine_avg_score': mean_cosine
        }
        
        return mean_overlap
    
    # ═══════════════════════════════════════════════════════════
    # Metric 6: Novelty Score
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_novelty(self, n_samples=200, n_recs=10):
        """
        Do recommendations include lesser-known songs?
        High novelty = recommending hidden gems, not just popular songs.
        Low novelty = always recommending the same popular tracks.
        
        Score: average inverse popularity of recommendations.
        """
        
        print("\n[METRIC 6] Novelty Score")
        print("-"*50)
        
        if 'popularity' not in self.df.columns:
            print("  ⚠ No popularity column, skipping...")
            return None
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        rec_popularities = []
        input_popularities = []
        
        for idx in sample_indices:
            input_pop = self.df.iloc[idx]['popularity']
            input_popularities.append(input_pop)
            
            recs = self.recommender.recommend_by_knn(idx, n_recs)
            for r in recs:
                rec_popularities.append(r['popularity'])
        
        mean_rec_pop = np.mean(rec_popularities)
        mean_input_pop = np.mean(input_popularities)
        catalog_mean_pop = self.df['popularity'].mean()
        
        # Novelty = how much less popular are recs compared to catalog average
        novelty_score = 1 - (mean_rec_pop / 100)  # 0-1 scale
        
        print(f"  Average input song popularity: {mean_input_pop:.1f}/100")
        print(f"  Average recommendation popularity: {mean_rec_pop:.1f}/100")
        print(f"  Catalog average popularity: {catalog_mean_pop:.1f}/100")
        print(f"  Novelty score: {novelty_score:.3f} (higher = more novel)")
        
        # Popularity distribution of recommendations
        pop_bins = {
            'Very Popular (80-100)': sum(1 for p in rec_popularities if p >= 80),
            'Popular (60-80)': sum(1 for p in rec_popularities if 60 <= p < 80),
            'Moderate (40-60)': sum(1 for p in rec_popularities if 40 <= p < 60),
            'Low (20-40)': sum(1 for p in rec_popularities if 20 <= p < 40),
            'Very Low (0-20)': sum(1 for p in rec_popularities if p < 20),
        }
        
        total_recs = len(rec_popularities)
        print(f"\n  Recommendation popularity distribution:")
        for label, count in pop_bins.items():
            pct = count / total_recs * 100
            bar = '█' * int(pct / 2)
            print(f"    {label:25s}: {pct:5.1f}% {bar}")
        
        # Interpret
        if novelty_score > 0.7:
            print(f"\n  ✓ HIGH NOVELTY: Recommending hidden gems")
        elif novelty_score > 0.5:
            print(f"\n  ⚠ MODERATE: Mix of popular and lesser-known")
        else:
            print(f"\n  ✗ LOW NOVELTY: Popularity bias detected")
        
        self.results['novelty'] = {
            'score': novelty_score,
            'mean_rec_popularity': mean_rec_pop,
            'mean_input_popularity': mean_input_pop,
            'catalog_mean_popularity': catalog_mean_pop,
            'distribution': pop_bins
        }
        
        return novelty_score
    
    # ═══════════════════════════════════════════════════════════
    # Metric 7: Latency / Performance
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_performance(self, n_samples=100, n_recs=10):
        """
        Measure recommendation latency.
        How fast can we generate recommendations?
        """
        
        print("\n[METRIC 7] Performance (Latency)")
        print("-"*50)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        # KNN latency
        knn_times = []
        for idx in sample_indices:
            start = time.time()
            _ = self.recommender.recommend_by_knn(idx, n_recs)
            knn_times.append(time.time() - start)
        
        # Cosine latency
        cosine_times = []
        for idx in sample_indices:
            start = time.time()
            _ = self.recommender.recommend_by_cosine(idx, n_recs)
            cosine_times.append(time.time() - start)
        
        # Custom features latency
        custom_times = []
        for _ in range(n_samples):
            features = {col: np.random.random() for col in self.feature_columns}
            start = time.time()
            _ = self.recommender.recommend_by_features(features, n_recs)
            custom_times.append(time.time() - start)
        
        knn_mean = np.mean(knn_times) * 1000  # Convert to ms
        cosine_mean = np.mean(cosine_times) * 1000
        custom_mean = np.mean(custom_times) * 1000
        
        print(f"  KNN recommendation:")
        print(f"    Mean latency: {knn_mean:.2f} ms")
        print(f"    P95 latency:  {np.percentile(knn_times, 95)*1000:.2f} ms")
        print(f"    P99 latency:  {np.percentile(knn_times, 99)*1000:.2f} ms")
        
        print(f"\n  Cosine Similarity recommendation:")
        print(f"    Mean latency: {cosine_mean:.2f} ms")
        print(f"    P95 latency:  {np.percentile(cosine_times, 95)*1000:.2f} ms")
        print(f"    P99 latency:  {np.percentile(cosine_times, 99)*1000:.2f} ms")
        
        print(f"\n  Custom features recommendation:")
        print(f"    Mean latency: {custom_mean:.2f} ms")
        print(f"    P95 latency:  {np.percentile(custom_times, 95)*1000:.2f} ms")
        
        # Interpret
        if knn_mean < 50:
            print(f"\n  ✓ EXCELLENT: Sub-50ms latency (real-time ready)")
        elif knn_mean < 200:
            print(f"\n  ✓ GOOD: Under 200ms (acceptable for web apps)")
        elif knn_mean < 1000:
            print(f"\n  ⚠ MODERATE: Under 1s (may need optimization)")
        else:
            print(f"\n  ✗ SLOW: Over 1s (needs optimization)")
        
        # Throughput
        throughput = 1000 / knn_mean  # recommendations per second
        print(f"\n  Throughput: ~{throughput:.0f} recommendations/second")
        
        self.results['performance'] = {
            'knn_mean_ms': knn_mean,
            'cosine_mean_ms': cosine_mean,
            'custom_mean_ms': custom_mean,
            'throughput_per_sec': throughput
        }
        
        return knn_mean
    
    # ═══════════════════════════════════════════════════════════
    # Metric 8: Stability Test
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_stability(self, n_samples=50, n_recs=10):
        """
        Are recommendations stable? 
        Same input should always give same output (deterministic).
        """
        
        print("\n[METRIC 8] Stability (Determinism)")
        print("-"*50)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        stable_count = 0
        total_tests = 0
        
        for idx in sample_indices:
            # Run recommendation twice
            recs1 = self.recommender.recommend_by_knn(idx, n_recs)
            recs2 = self.recommender.recommend_by_knn(idx, n_recs)
            
            # Check if results are identical
            set1 = set(r['index'] for r in recs1)
            set2 = set(r['index'] for r in recs2)
            
            if set1 == set2:
                stable_count += 1
            total_tests += 1
        
        stability_rate = stable_count / total_tests * 100
        
        print(f"  Stability rate: {stability_rate:.1f}%")
        print(f"  ({stable_count}/{total_tests} queries returned identical results)")
        
        if stability_rate == 100:
            print(f"  ✓ PERFECT: Fully deterministic recommendations")
        elif stability_rate > 95:
            print(f"  ✓ GOOD: Highly stable")
        else:
            print(f"  ⚠ UNSTABLE: Results vary between runs")
        
        self.results['stability'] = {
            'rate': stability_rate,
            'stable_count': stable_count,
            'total_tests': total_tests
        }
        
        return stability_rate
    
    # ═══════════════════════════════════════════════════════════
    # Metric 9: Artist Diversity
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_artist_diversity(self, n_samples=200, n_recs=10):
        """
        Do recommendations include diverse artists?
        Or do they always recommend the same artist?
        """
        
        print("\n[METRIC 9] Artist Diversity")
        print("-"*50)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        unique_artist_ratios = []
        same_artist_counts = []
        
        for idx in sample_indices:
            input_artist = self.df.iloc[idx]['artists']
            recs = self.recommender.recommend_by_knn(idx, n_recs)
            
            if not recs:
                continue
            
            # Count unique artists in recommendations
            rec_artists = [r['artists'] for r in recs]
            unique_ratio = len(set(rec_artists)) / len(rec_artists)
            unique_artist_ratios.append(unique_ratio)
            
            # Count same-artist recommendations
            same_artist = sum(1 for a in rec_artists if a == input_artist)
            same_artist_counts.append(same_artist)
        
        mean_unique_ratio = np.mean(unique_artist_ratios)
        mean_same_artist = np.mean(same_artist_counts)
        
        print(f"  Unique artist ratio in recs: {mean_unique_ratio*100:.1f}%")
        print(f"  Avg same-artist recs: {mean_same_artist:.2f} out of {n_recs}")
        print(f"  Same-artist rate: {mean_same_artist/n_recs*100:.1f}%")
        
        if mean_unique_ratio > 0.8:
            print(f"\n  ✓ GOOD: High artist diversity")
        elif mean_unique_ratio > 0.5:
            print(f"\n  ⚠ MODERATE: Some artist repetition")
        else:
            print(f"\n  ✗ LOW: Too many same-artist recommendations")
        
        self.results['artist_diversity'] = {
            'unique_ratio': mean_unique_ratio,
            'same_artist_avg': mean_same_artist,
            'same_artist_rate': mean_same_artist / n_recs
        }
        
        return mean_unique_ratio
    
    # ═══════════════════════════════════════════════════════════
    # Metric 10: Cold Start Handling
    # ═══════════════════════════════════════════════════════════
    
    def evaluate_cold_start(self, n_tests=50, n_recs=10):
        """
        Test how well the system handles edge cases:
        - Extreme feature values
        - Rare genres
        - Very unpopular songs
        """
        
        print("\n[METRIC 10] Cold Start / Edge Cases")
        print("-"*50)
        
        edge_cases_passed = 0
        total_cases = 0
        
        # Test 1: Extreme feature values
        print("\n  Testing extreme feature values...")
        extreme_features = [
            {'danceability': 1.0, 'energy': 1.0, 'valence': 1.0, 'tempo': 1.0,
             'acousticness': 0.0, 'speechiness': 0.0, 'instrumentalness': 0.0,
             'liveness': 0.0, 'loudness': 1.0},
            {'danceability': 0.0, 'energy': 0.0, 'valence': 0.0, 'tempo': 0.0,
             'acousticness': 1.0, 'speechiness': 0.0, 'instrumentalness': 1.0,
             'liveness': 0.0, 'loudness': 0.0},
            {'danceability': 0.5, 'energy': 0.5, 'valence': 0.5, 'tempo': 0.5,
             'acousticness': 0.5, 'speechiness': 0.5, 'instrumentalness': 0.5,
             'liveness': 0.5, 'loudness': 0.5},
        ]
        
        for i, features in enumerate(extreme_features):
            total_cases += 1
            try:
                recs = self.recommender.recommend_by_features(features, n_recs)
                if len(recs) >= n_recs:
                    edge_cases_passed += 1
                    print(f"    ✓ Extreme case {i+1}: Got {len(recs)} recommendations")
                else:
                    print(f"    ⚠ Extreme case {i+1}: Only got {len(recs)} recommendations")
            except Exception as e:
                print(f"    ✗ Extreme case {i+1}: FAILED - {e}")
        
        # Test 2: Very unpopular songs
        print("\n  Testing unpopular songs...")
        if 'popularity' in self.df.columns:
            unpopular = self.df[self.df['popularity'] <= 5].head(10)
            
            for idx in unpopular.index[:5]:
                total_cases += 1
                try:
                    recs = self.recommender.recommend_by_knn(idx, n_recs)
                    if len(recs) >= n_recs:
                        edge_cases_passed += 1
                        print(f"    ✓ Unpopular song (pop={self.df.iloc[idx]['popularity']}): OK")
                    else:
                        print(f"    ⚠ Unpopular song: Only {len(recs)} recs")
                except Exception as e:
                    print(f"    ✗ Unpopular song: FAILED - {e}")
        
        # Test 3: Rare genres
        print("\n  Testing rare genres...")
        if 'track_genre' in self.df.columns:
            genre_counts = self.df['track_genre'].value_counts()
            rare_genres = genre_counts[genre_counts < 100].index.tolist()[:5]
            
            for genre in rare_genres:
                total_cases += 1
                genre_songs = self.df[self.df['track_genre'] == genre]
                if len(genre_songs) > 0:
                    idx = genre_songs.index[0]
                    try:
                        recs = self.recommender.recommend_by_knn(idx, n_recs)
                        if len(recs) >= n_recs:
                            edge_cases_passed += 1
                            print(f"    ✓ Rare genre '{genre}' ({len(genre_songs)} songs): OK")
                        else:
                            print(f"    ⚠ Rare genre '{genre}': Only {len(recs)} recs")
                    except Exception as e:
                        print(f"    ✗ Rare genre '{genre}': FAILED - {e}")
        
        pass_rate = edge_cases_passed / total_cases * 100 if total_cases > 0 else 0
        
        print(f"\n  Edge case pass rate: {pass_rate:.1f}% ({edge_cases_passed}/{total_cases})")
        
        if pass_rate == 100:
            print(f"  ✓ EXCELLENT: Handles all edge cases")
        elif pass_rate > 80:
            print(f"  ✓ GOOD: Handles most edge cases")
        else:
            print(f"  ⚠ NEEDS WORK: Some edge cases fail")
        
        self.results['cold_start'] = {
            'pass_rate': pass_rate,
            'passed': edge_cases_passed,
            'total': total_cases
        }
        
        return pass_rate
    
    # ═══════════════════════════════════════════════════════════
    # Full Evaluation Report
    # ═══════════════════════════════════════════════════════════
    
    def run_full_evaluation(self):
        """Run all evaluation metrics and generate a report"""
        
        print("\n" + "="*60)
        print("    SPOTIFY RECOMMENDER - FULL EVALUATION REPORT")
        print("="*60)
        print(f"    Dataset: {len(self.df):,} songs")
        print(f"    Features: {len(self.feature_columns)}")
        print(f"    Genres: {self.df['track_genre'].nunique() if 'track_genre' in self.df.columns else 'N/A'}")
        print("="*60)
        
        start_time = time.time()
        
        # Run all metrics
        diversity = self.evaluate_diversity()
        coverage = self.evaluate_coverage()
        genre_coherence = self.evaluate_genre_coherence()
        similarity = self.evaluate_similarity_distribution()
        method_comparison = self.evaluate_method_comparison()
        novelty = self.evaluate_novelty()
        performance = self.evaluate_performance()
        stability = self.evaluate_stability()
        artist_diversity = self.evaluate_artist_diversity()
        cold_start = self.evaluate_cold_start()
        
        total_time = time.time() - start_time
        
        # ─────────────────────────────────────────────
        # Summary Report Card
        # ─────────────────────────────────────────────
        print("\n\n" + "="*60)
        print("    📊 EVALUATION SUMMARY REPORT CARD")
        print("="*60)
        
        def grade(score, thresholds):
            """Assign letter grade based on thresholds"""
            if score >= thresholds[0]:
                return "A", "✓"
            elif score >= thresholds[1]:
                return "B", "✓"
            elif score >= thresholds[2]:
                return "C", "⚠"
            else:
                return "D", "✗"
        
        metrics_summary = [
            ("Diversity", diversity, [0.3, 0.2, 0.1]),
            ("Coverage", coverage, [50, 30, 15]),
            ("Genre Coherence", genre_coherence * 100 if genre_coherence else 0, [50, 30, 15]),
            ("Similarity vs Random", similarity, [30, 15, 5]),
            ("Method Agreement", method_comparison * 100, [80, 60, 40]),
            ("Novelty", novelty * 100 if novelty else 50, [70, 50, 30]),
            ("Latency (ms)", 100 - min(performance, 100), [50, 30, 10]),
            ("Stability", stability, [99, 95, 80]),
            ("Artist Diversity", artist_diversity * 100, [80, 60, 40]),
            ("Cold Start", cold_start, [95, 80, 60]),
        ]
        
        print(f"\n  {'Metric':<25s} {'Score':<12s} {'Grade':<8s} {'Status'}")
        print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*6}")
        
        grades = []
        for name, score, thresholds in metrics_summary:
            if score is not None:
                letter, status = grade(score, thresholds)
                grades.append(letter)
                score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                print(f"  {name:<25s} {score_str:<12s} {letter:<8s} {status}")
            else:
                print(f"  {name:<25s} {'N/A':<12s} {'-':<8s} -")
        
        # Overall grade
        grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        if grades:
            avg_grade = np.mean([grade_values[g] for g in grades])
            if avg_grade >= 3.5:
                overall = "A"
            elif avg_grade >= 2.5:
                overall = "B"
            elif avg_grade >= 1.5:
                overall = "C"
            else:
                overall = "D"
        else:
            overall = "N/A"
        
        print(f"\n  {'='*55}")
        print(f"  {'OVERALL GRADE':<25s} {'':12s} {overall}")
        print(f"  {'='*55}")
        
        print(f"\n  Total evaluation time: {total_time:.1f} seconds")
        
        # ─────────────────────────────────────────────
        # Recommendations for improvement
        # ─────────────────────────────────────────────
        print("\n" + "-"*60)
        print("  💡 RECOMMENDATIONS FOR IMPROVEMENT")
        print("-"*60)
        
        if diversity and diversity < 0.2:
            print("  • Add diversity injection: penalize recommendations too similar to each other")
        
        if coverage and coverage < 30:
            print("  • Address popularity bias: add randomization or explore/exploit strategy")
        
        if genre_coherence and genre_coherence < 0.3:
            print("  • Consider adding genre as a feature or filter option")
        
        if novelty and novelty < 0.5:
            print("  • Boost lesser-known songs: add novelty bonus to scoring")
        
        if performance and performance > 200:
            print("  • Optimize latency: use approximate nearest neighbors (Annoy/FAISS)")
        
        if artist_diversity and artist_diversity < 0.6:
            print("  • Add artist deduplication: limit max songs per artist in results")
        
        if not any([
            diversity and diversity < 0.2,
            coverage and coverage < 30,
            performance and performance > 200
        ]):
            print("  • System is performing well! Consider A/B testing with real users.")
        
        print("\n" + "="*60)
        print("    EVALUATION COMPLETE!")
        print("="*60)
        
        return self.results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    evaluator = RecommenderEvaluator()
    results = evaluator.run_full_evaluation()