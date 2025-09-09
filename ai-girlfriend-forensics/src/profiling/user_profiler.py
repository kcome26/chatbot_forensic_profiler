"""
Machine learning based user profiling module.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import json
import warnings
warnings.filterwarnings('ignore')


class UserProfiler:
    """ML-based user profiler for behavioral pattern recognition."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.anomaly_detector = None
        self.feature_names = []
        self.profile_clusters = {}
        
    def extract_features(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract numerical features from analysis results for ML processing.
        
        Args:
            analysis_results: Complete analysis results dictionary
            
        Returns:
            DataFrame with extracted features
        """
        features = {}
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(analysis_results.get('temporal_analysis', {}))
        features.update(temporal_features)
        
        # Extract content features
        content_features = self._extract_content_features(analysis_results.get('content_analysis', {}))
        features.update(content_features)
        
        # Extract behavioral features
        behavioral_features = self._extract_behavioral_features(analysis_results.get('behavioral_analysis', {}))
        features.update(behavioral_features)
        
        # Extract relationship features
        relationship_features = self._extract_relationship_features(analysis_results.get('relationship_analysis', {}))
        features.update(relationship_features)
        
        # Create DataFrame
        if features:
            self.feature_names = list(features.keys())
            return pd.DataFrame([features])
        else:
            return pd.DataFrame()
    
    def _extract_temporal_features(self, temporal_analysis: Dict) -> Dict[str, float]:
        """Extract numerical features from temporal analysis."""
        features = {}
        
        for analysis_name, analysis in temporal_analysis.items():
            prefix = f"temporal_{analysis_name[:10]}"  # Truncate for brevity
            
            # Activity patterns
            activity_patterns = analysis.get('activity_patterns', {})
            features[f"{prefix}_peak_hour"] = float(activity_patterns.get('peak_activity_hour', 12))
            features[f"{prefix}_peak_day"] = float(activity_patterns.get('peak_activity_day', 3))
            
            time_breakdown = activity_patterns.get('time_period_breakdown', {})
            for period, data in time_breakdown.items():
                features[f"{prefix}_{period}_pct"] = float(data.get('percentage', 0))
            
            # Usage intensity
            usage_intensity = analysis.get('usage_intensity', {})
            daily_stats = usage_intensity.get('daily_stats', {})
            features[f"{prefix}_avg_daily"] = float(daily_stats.get('average_daily_messages', 0))
            features[f"{prefix}_max_daily"] = float(daily_stats.get('max_daily_messages', 0))
            features[f"{prefix}_daily_variance"] = float(daily_stats.get('daily_variance', 0))
            
            # Circadian patterns
            circadian = analysis.get('circadian_patterns', {})
            chronotype_map = {'morning_person': 1, 'night_owl': 3, 'intermediate': 2}
            features[f"{prefix}_chronotype"] = float(chronotype_map.get(circadian.get('chronotype', 'intermediate'), 2))
            features[f"{prefix}_circadian_reg"] = float(circadian.get('circadian_regularity', 0))
            
            # Session analysis
            sessions = analysis.get('session_analysis', {})
            session_overview = sessions.get('session_overview', {})
            features[f"{prefix}_avg_session_dur"] = float(session_overview.get('avg_session_duration_minutes', 0))
            features[f"{prefix}_avg_msg_per_session"] = float(session_overview.get('avg_messages_per_session', 0))
            
            break  # Use only first temporal analysis to avoid too many features
        
        return features
    
    def _extract_content_features(self, content_analysis: Dict) -> Dict[str, float]:
        """Extract numerical features from content analysis."""
        features = {}
        
        # Aggregate across all tables and columns
        all_sentiment_scores = []
        all_emotional_expressions = {}
        all_linguistic_complexity = []
        all_conversation_stats = []
        
        for table_name, table_results in content_analysis.items():
            for column_name, column_results in table_results.items():
                # Sentiment analysis
                sentiment = column_results.get('sentiment_analysis', {})
                overall_sentiment = sentiment.get('overall_sentiment', {})
                if overall_sentiment:
                    all_sentiment_scores.append({
                        'positive': overall_sentiment.get('positive', 0),
                        'negative': overall_sentiment.get('negative', 0),
                        'neutral': overall_sentiment.get('neutral', 0),
                        'compound': overall_sentiment.get('compound', 0)
                    })
                
                # Emotional patterns
                emotional_patterns = column_results.get('emotional_patterns', {})
                emotional_expressions = emotional_patterns.get('emotional_expressions', {})
                for emotion, count in emotional_expressions.items():
                    all_emotional_expressions[emotion] = all_emotional_expressions.get(emotion, 0) + count
                
                # Linguistic analysis
                linguistic = column_results.get('linguistic_analysis', {})
                if 'linguistic_complexity' in linguistic:
                    all_linguistic_complexity.append(linguistic['linguistic_complexity'])
                
                # Conversation patterns
                conv_patterns = column_results.get('conversation_patterns', {})
                msg_stats = conv_patterns.get('message_statistics', {})
                if msg_stats:
                    all_conversation_stats.append(msg_stats)
        
        # Aggregate sentiment features
        if all_sentiment_scores:
            avg_sentiment = {
                'positive': np.mean([s['positive'] for s in all_sentiment_scores]),
                'negative': np.mean([s['negative'] for s in all_sentiment_scores]),
                'neutral': np.mean([s['neutral'] for s in all_sentiment_scores]),
                'compound': np.mean([s['compound'] for s in all_sentiment_scores])
            }
            
            features.update({
                'content_sentiment_positive': float(avg_sentiment['positive']),
                'content_sentiment_negative': float(avg_sentiment['negative']),
                'content_sentiment_neutral': float(avg_sentiment['neutral']),
                'content_sentiment_compound': float(avg_sentiment['compound'])
            })
        
        # Aggregate emotional features
        emotion_features = ['love', 'happiness', 'sadness', 'anger', 'fear', 'surprise']
        for emotion in emotion_features:
            features[f'content_emotion_{emotion}'] = float(all_emotional_expressions.get(emotion, 0))
        
        # Aggregate linguistic features
        if all_linguistic_complexity:
            features['content_linguistic_complexity'] = float(np.mean(all_linguistic_complexity))
        
        # Aggregate conversation features
        if all_conversation_stats:
            features['content_avg_message_length'] = float(np.mean([s.get('avg_message_length', 0) for s in all_conversation_stats]))
            features['content_avg_word_count'] = float(np.mean([s.get('avg_word_count', 0) for s in all_conversation_stats]))
            features['content_total_messages'] = float(sum([s.get('total_messages', 0) for s in all_conversation_stats]))
        
        return features
    
    def _extract_behavioral_features(self, behavioral_analysis: Dict) -> Dict[str, float]:
        """Extract numerical features from behavioral analysis."""
        features = {}
        
        for table_name, table_results in behavioral_analysis.items():
            prefix = f"behavior_{table_name[:10]}"  # Truncate for brevity
            
            # Communication style
            comm_style = table_results.get('communication_style', {})
            features[f"{prefix}_avg_msg_len"] = float(comm_style.get('avg_message_length', 0))
            features[f"{prefix}_question_freq"] = float(comm_style.get('question_frequency', 0))
            features[f"{prefix}_exclamation_freq"] = float(comm_style.get('exclamation_frequency', 0))
            features[f"{prefix}_capital_usage"] = float(comm_style.get('capital_usage', 0))
            
            # Interaction frequency
            interaction_freq = table_results.get('interaction_frequency', {})
            features[f"{prefix}_avg_daily_int"] = float(interaction_freq.get('avg_daily_interactions', 0))
            features[f"{prefix}_max_daily_int"] = float(interaction_freq.get('max_daily_interactions', 0))
            features[f"{prefix}_interaction_var"] = float(interaction_freq.get('interaction_variance', 0))
            
            # Engagement patterns
            engagement = table_results.get('engagement_patterns', {})
            features[f"{prefix}_engagement_score"] = float(engagement.get('engagement_score', 0))
            
            break  # Use only first behavioral analysis to avoid too many features
        
        return features
    
    def _extract_relationship_features(self, relationship_analysis: Dict) -> Dict[str, float]:
        """Extract numerical features from relationship analysis."""
        features = {}
        
        # Relationship metrics
        rel_metrics = relationship_analysis.get('relationship_metrics', {})
        features['relationship_emotional_intensity'] = float(rel_metrics.get('emotional_intensity', 0))
        features['relationship_intimacy_level'] = float(rel_metrics.get('intimacy_level', 0))
        features['relationship_dependency'] = float(rel_metrics.get('dependency_indicators', 0))
        
        # Attachment analysis
        attachment = relationship_analysis.get('attachment_analysis', {})
        attachment_map = {'secure': 1, 'anxious': 2, 'avoidant': 3, 'mixed': 4, 'unknown': 0}
        features['relationship_attachment_style'] = float(attachment_map.get(attachment.get('predicted_style', 'unknown'), 0))
        features['relationship_attachment_confidence'] = float(attachment.get('confidence', 0))
        
        # Emotional patterns totals
        emotional_patterns = relationship_analysis.get('emotional_patterns', {})
        features['relationship_total_emotions'] = float(sum(emotional_patterns.values()))
        
        relationship_indicators = relationship_analysis.get('relationship_indicators', {})
        features['relationship_total_indicators'] = float(sum(relationship_indicators.values()))
        
        return features
    
    def create_user_profile(self, features_df: pd.DataFrame, user_id: str = "unknown") -> Dict[str, Any]:
        """
        Create a comprehensive user profile using ML techniques.
        
        Args:
            features_df: DataFrame with extracted features
            user_id: User identifier
            
        Returns:
            Comprehensive user profile dictionary
        """
        if features_df.empty:
            return {'error': 'No features available for profiling'}
        
        profile = {
            'user_id': user_id,
            'profile_timestamp': pd.Timestamp.now().isoformat(),
            'feature_analysis': {},
            'behavioral_cluster': {},
            'anomaly_detection': {},
            'risk_scoring': {},
            'personality_indicators': {}
        }
        
        # Normalize features
        features_normalized = self._normalize_features(features_df)
        
        # Feature analysis
        profile['feature_analysis'] = self._analyze_features(features_df, features_normalized)
        
        # Behavioral clustering
        profile['behavioral_cluster'] = self._perform_clustering(features_normalized)
        
        # Anomaly detection
        profile['anomaly_detection'] = self._detect_anomalies(features_normalized)
        
        # Risk scoring
        profile['risk_scoring'] = self._calculate_risk_scores(features_df)
        
        # Personality indicators
        profile['personality_indicators'] = self._extract_personality_indicators(features_df)
        
        return profile
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for ML processing."""
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        return pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
    
    def _analyze_features(self, features_df: pd.DataFrame, features_normalized: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions and importance."""
        analysis = {
            'feature_count': len(features_df.columns),
            'feature_summary': {},
            'top_features': {},
            'feature_correlations': {}
        }
        
        # Basic statistics for each feature
        for col in features_df.columns:
            analysis['feature_summary'][col] = {
                'mean': float(features_df[col].mean()),
                'std': float(features_df[col].std()),
                'min': float(features_df[col].min()),
                'max': float(features_df[col].max())
            }
        
        # Identify top features by variance
        feature_variances = features_normalized.var().sort_values(ascending=False)
        analysis['top_features'] = {
            'highest_variance': feature_variances.head(5).to_dict(),
            'lowest_variance': feature_variances.tail(5).to_dict()
        }
        
        # Feature correlations (if multiple samples in future)
        if len(features_df) > 1:
            corr_matrix = features_df.corr()
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            analysis['feature_correlations']['high_correlations'] = high_corr_pairs
        
        return analysis
    
    def _perform_clustering(self, features_normalized: pd.DataFrame) -> Dict[str, Any]:
        """Perform behavioral clustering analysis."""
        if len(features_normalized) < 2:
            return {'message': 'Insufficient data for clustering', 'cluster_id': 0}
        
        # Use small number of clusters for interpretability
        n_clusters = min(3, len(features_normalized))
        
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clustering.fit_predict(features_normalized)
        
        # Store clustering model
        self.clustering_model = clustering
        
        cluster_result = {
            'cluster_id': int(cluster_labels[0]),
            'total_clusters': n_clusters,
            'cluster_centers': clustering.cluster_centers_.tolist(),
            'inertia': float(clustering.inertia_)
        }
        
        # Interpret cluster characteristics
        cluster_result['cluster_interpretation'] = self._interpret_cluster(
            cluster_labels[0], features_normalized, clustering.cluster_centers_
        )
        
        return cluster_result
    
    def _detect_anomalies(self, features_normalized: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalous behavioral patterns."""
        if len(features_normalized) < 2:
            return {'anomaly_score': 0.0, 'is_anomaly': False}
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features_normalized)
        anomaly_scores = iso_forest.score_samples(features_normalized)
        
        # Store anomaly detector
        self.anomaly_detector = iso_forest
        
        return {
            'anomaly_score': float(anomaly_scores[0]),
            'is_anomaly': bool(anomaly_labels[0] == -1),
            'anomaly_threshold': float(np.percentile(anomaly_scores, 10)),
            'anomaly_interpretation': self._interpret_anomaly(anomaly_scores[0], anomaly_labels[0])
        }
    
    def _calculate_risk_scores(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various risk scores based on features."""
        risk_scores = {
            'addiction_risk': 0.0,
            'social_isolation_risk': 0.0,
            'emotional_dependency_risk': 0.0,
            'sleep_disruption_risk': 0.0,
            'overall_risk': 0.0
        }
        
        # Addiction risk based on usage patterns
        if 'temporal_avg_daily' in features_df.columns:
            daily_usage = features_df['temporal_avg_daily'].iloc[0]
            risk_scores['addiction_risk'] = min(1.0, daily_usage / 200.0)  # Normalize by high usage threshold
        
        # Social isolation risk based on temporal patterns
        if 'temporal_night_pct' in features_df.columns:
            night_usage = features_df['temporal_night_pct'].iloc[0]
            risk_scores['social_isolation_risk'] = min(1.0, night_usage / 50.0)  # High night usage
        
        # Emotional dependency risk
        if 'relationship_dependency' in features_df.columns:
            dependency = features_df['relationship_dependency'].iloc[0]
            risk_scores['emotional_dependency_risk'] = min(1.0, dependency / 20.0)
        
        # Sleep disruption risk
        if 'temporal_night_pct' in features_df.columns:
            night_usage = features_df['temporal_night_pct'].iloc[0]
            risk_scores['sleep_disruption_risk'] = min(1.0, night_usage / 30.0)
        
        # Overall risk (weighted average)
        weights = [0.3, 0.2, 0.3, 0.2]  # Weights for each risk category
        risk_values = list(risk_scores.values())[:-1]  # Exclude overall_risk
        risk_scores['overall_risk'] = sum(w * v for w, v in zip(weights, risk_values))
        
        # Add risk level categorization
        overall_risk = risk_scores['overall_risk']
        if overall_risk > 0.7:
            risk_level = 'high'
        elif overall_risk > 0.4:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        risk_scores['risk_level'] = risk_level
        risk_scores['risk_factors'] = self._identify_risk_factors(features_df, risk_scores)
        
        return risk_scores
    
    def _extract_personality_indicators(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract personality indicators from behavioral features."""
        personality = {
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'conscientiousness': 0.5,
            'neuroticism': 0.5,
            'openness': 0.5,
            'dominant_traits': [],
            'personality_summary': ''
        }
        
        # Extraversion (based on social engagement)
        if 'content_total_messages' in features_df.columns:
            msg_count = features_df['content_total_messages'].iloc[0]
            personality['extraversion'] = min(1.0, msg_count / 1000.0)
        
        # Agreeableness (based on positive sentiment)
        if 'content_sentiment_positive' in features_df.columns:
            positive_sentiment = features_df['content_sentiment_positive'].iloc[0]
            personality['agreeableness'] = positive_sentiment
        
        # Conscientiousness (based on temporal regularity)
        if 'temporal_circadian_reg' in features_df.columns:
            regularity = features_df['temporal_circadian_reg'].iloc[0]
            personality['conscientiousness'] = regularity
        
        # Neuroticism (based on negative emotions and dependency)
        neuroticism_factors = []
        if 'content_sentiment_negative' in features_df.columns:
            neuroticism_factors.append(features_df['content_sentiment_negative'].iloc[0])
        if 'relationship_dependency' in features_df.columns:
            neuroticism_factors.append(min(1.0, features_df['relationship_dependency'].iloc[0] / 20.0))
        
        if neuroticism_factors:
            personality['neuroticism'] = np.mean(neuroticism_factors)
        
        # Openness (based on linguistic complexity and variety)
        if 'content_linguistic_complexity' in features_df.columns:
            complexity = features_df['content_linguistic_complexity'].iloc[0]
            personality['openness'] = complexity
        
        # Identify dominant traits
        trait_scores = {k: v for k, v in personality.items() if isinstance(v, float)}
        sorted_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)
        personality['dominant_traits'] = [trait for trait, score in sorted_traits[:2] if score > 0.6]
        
        # Create personality summary
        personality['personality_summary'] = self._create_personality_summary(personality)
        
        return personality
    
    def _interpret_cluster(self, cluster_id: int, features_normalized: pd.DataFrame, cluster_centers: np.ndarray) -> str:
        """Interpret the meaning of a behavioral cluster."""
        if len(cluster_centers) <= cluster_id:
            return "Unknown cluster"
        
        center = cluster_centers[cluster_id]
        
        # Simple interpretation based on feature values
        if np.mean(center) > 0.5:
            return "High engagement user with active behavioral patterns"
        elif np.mean(center) < -0.5:
            return "Low engagement user with minimal activity"
        else:
            return "Moderate engagement user with balanced behavioral patterns"
    
    def _interpret_anomaly(self, anomaly_score: float, anomaly_label: int) -> str:
        """Interpret anomaly detection results."""
        if anomaly_label == -1:
            return f"Unusual behavioral pattern detected (score: {anomaly_score:.3f})"
        else:
            return f"Normal behavioral pattern (score: {anomaly_score:.3f})"
    
    def _identify_risk_factors(self, features_df: pd.DataFrame, risk_scores: Dict[str, float]) -> List[str]:
        """Identify specific risk factors based on features and scores."""
        risk_factors = []
        
        if risk_scores['addiction_risk'] > 0.6:
            risk_factors.append('high_usage_frequency')
        
        if risk_scores['social_isolation_risk'] > 0.6:
            risk_factors.append('excessive_nighttime_usage')
        
        if risk_scores['emotional_dependency_risk'] > 0.6:
            risk_factors.append('strong_emotional_attachment')
        
        if risk_scores['sleep_disruption_risk'] > 0.6:
            risk_factors.append('sleep_pattern_disruption')
        
        # Additional feature-based risk factors
        if 'content_emotion_sadness' in features_df.columns:
            sadness = features_df['content_emotion_sadness'].iloc[0]
            if sadness > 10:
                risk_factors.append('elevated_sadness_expressions')
        
        if 'temporal_daily_variance' in features_df.columns:
            variance = features_df['temporal_daily_variance'].iloc[0]
            if variance > 100:
                risk_factors.append('irregular_usage_patterns')
        
        return risk_factors
    
    def _create_personality_summary(self, personality: Dict[str, Any]) -> str:
        """Create a human-readable personality summary."""
        traits = []
        
        if personality['extraversion'] > 0.7:
            traits.append("highly social")
        elif personality['extraversion'] < 0.3:
            traits.append("introverted")
        
        if personality['agreeableness'] > 0.7:
            traits.append("cooperative")
        elif personality['agreeableness'] < 0.3:
            traits.append("skeptical")
        
        if personality['conscientiousness'] > 0.7:
            traits.append("organized")
        elif personality['conscientiousness'] < 0.3:
            traits.append("spontaneous")
        
        if personality['neuroticism'] > 0.7:
            traits.append("emotionally sensitive")
        elif personality['neuroticism'] < 0.3:
            traits.append("emotionally stable")
        
        if personality['openness'] > 0.7:
            traits.append("intellectually curious")
        elif personality['openness'] < 0.3:
            traits.append("practical")
        
        if traits:
            return f"Personality appears to be {', '.join(traits)}"
        else:
            return "Balanced personality profile with moderate traits"
    
    def compare_profiles(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two user profiles to identify similarities and differences."""
        comparison = {
            'similarity_score': 0.0,
            'key_differences': [],
            'behavioral_similarity': {},
            'risk_comparison': {},
            'personality_comparison': {}
        }
        
        # Compare behavioral clusters
        cluster1 = profile1.get('behavioral_cluster', {}).get('cluster_id', 0)
        cluster2 = profile2.get('behavioral_cluster', {}).get('cluster_id', 0)
        comparison['behavioral_similarity']['same_cluster'] = cluster1 == cluster2
        
        # Compare risk scores
        risk1 = profile1.get('risk_scoring', {})
        risk2 = profile2.get('risk_scoring', {})
        
        if risk1 and risk2:
            risk_diff = abs(risk1.get('overall_risk', 0) - risk2.get('overall_risk', 0))
            comparison['risk_comparison'] = {
                'risk_difference': float(risk_diff),
                'similar_risk_level': risk_diff < 0.2
            }
        
        # Compare personality traits
        personality1 = profile1.get('personality_indicators', {})
        personality2 = profile2.get('personality_indicators', {})
        
        if personality1 and personality2:
            trait_similarities = []
            for trait in ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']:
                if trait in personality1 and trait in personality2:
                    diff = abs(personality1[trait] - personality2[trait])
                    trait_similarities.append(1 - diff)  # Convert difference to similarity
            
            if trait_similarities:
                comparison['personality_comparison']['average_similarity'] = float(np.mean(trait_similarities))
        
        # Calculate overall similarity
        similarities = []
        if 'same_cluster' in comparison['behavioral_similarity']:
            similarities.append(1.0 if comparison['behavioral_similarity']['same_cluster'] else 0.0)
        if 'similar_risk_level' in comparison['risk_comparison']:
            similarities.append(1.0 if comparison['risk_comparison']['similar_risk_level'] else 0.0)
        if 'average_similarity' in comparison['personality_comparison']:
            similarities.append(comparison['personality_comparison']['average_similarity'])
        
        if similarities:
            comparison['similarity_score'] = float(np.mean(similarities))
        
        return comparison
