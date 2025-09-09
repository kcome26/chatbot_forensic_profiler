"""
Machine Learning analyzer for behavioral profiling and forensic analysis.

This module provides comprehensive ML-based analysis of chat messages to create
behavioral profiles and identify psychological patterns.
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ML and NLP imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import LatentDirichletAllocation, PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLAnalyzer:
    """Advanced ML-based analyzer for behavioral profiling and forensic analysis."""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.vectorizer = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.clustering_model = None
        self.anomaly_detector = None
        
        # Initialize NLP tools
        self._initialize_nlp_tools()
        
        # Analysis results cache
        self.analysis_cache = {}
        
    def _initialize_nlp_tools(self):
        """Initialize NLP and ML tools."""
        if NLTK_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                logger.warning("NLTK VADER lexicon not found. Sentiment analysis will be limited.")
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Some ML features will be disabled.")
        
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available. Some NLP features will be disabled.")
    
    def analyze_extraction_results(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive ML analysis on extraction results.
        
        Args:
            extraction_results: Results from data extraction
            
        Returns:
            Dictionary containing ML analysis results
        """
        logger.info("Starting ML analysis of extraction results")
        
        # Initialize analysis results structure
        ml_analysis = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_version': '1.0.0',
                'ml_capabilities': {
                    'sklearn_available': SKLEARN_AVAILABLE,
                    'nltk_available': NLTK_AVAILABLE,
                    'sentiment_analysis': self.sentiment_analyzer is not None
                }
            },
            'behavioral_profile': {},
            'content_analysis': {},
            'temporal_analysis': {},
            'risk_assessment': {},
            'psychological_indicators': {},
            'anomaly_detection': {},
            'relationship_patterns': {}
        }
        
        # Handle both scan results and extraction results formats
        data_to_analyze = extraction_results.get('extracted_data', extraction_results.get('databases', {}))
        
        # Process each application's data
        for app_name, app_data in data_to_analyze.items():
            if not app_data:
                continue
                
            logger.info(f"Analyzing {app_name} data...")
            
            # Combine all messages from all databases for this app
            all_messages = []
            if isinstance(app_data, list):
                # Extraction results format
                for db_entry in app_data:
                    if 'data' in db_entry and 'messages' in db_entry['data']:
                        all_messages.extend(db_entry['data']['messages'])
            elif isinstance(app_data, dict):
                # Direct database format
                if 'messages' in app_data:
                    all_messages.extend(app_data['messages'])
            
            if not all_messages:
                continue
            
            # Convert to DataFrame for analysis
            messages_df = self._prepare_messages_dataframe(all_messages)
            
            if messages_df.empty:
                continue
            
            # Perform comprehensive analysis
            app_analysis = self._analyze_app_messages(messages_df, app_name)
            
            # Store results
            for analysis_type, results in app_analysis.items():
                if analysis_type not in ml_analysis:
                    ml_analysis[analysis_type] = {}
                ml_analysis[analysis_type][app_name] = results
        
        # Perform cross-application analysis
        ml_analysis['cross_app_analysis'] = self._cross_application_analysis(ml_analysis)
        
        # Generate final behavioral profile
        ml_analysis['final_profile'] = self._generate_behavioral_profile(ml_analysis)
        
        return ml_analysis
    
    def _prepare_messages_dataframe(self, messages: List[Dict]) -> pd.DataFrame:
        """Convert messages to a structured DataFrame."""
        if not messages:
            return pd.DataFrame()
        
        # Extract and standardize fields
        processed_messages = []
        
        for msg in messages:
            processed_msg = {
                'raw_message': str(msg),
                'text': '',
                'timestamp': None,
                'user_id': None,
                'conversation_id': None,
                'message_length': 0,
                'source_table': msg.get('_forensic_metadata', {}).get('source_table', msg.get('_table', 'unknown'))
            }
            
            # Extract text content from various fields
            text_extracted = False
            
            # Handle latest_msg JSON structure (UShow_iChat format)
            if 'latest_msg' in msg and msg['latest_msg']:
                try:
                    latest_msg_data = json.loads(str(msg['latest_msg']))
                    if 'text_data' in latest_msg_data and 'raw_content' in latest_msg_data['text_data']:
                        processed_msg['text'] = self._clean_text(latest_msg_data['text_data']['raw_content'])
                        processed_msg['message_length'] = len(processed_msg['text'])
                        text_extracted = True
                        # Extract timestamp from latest_msg
                        if 'send_time' in latest_msg_data:
                            try:
                                ts = latest_msg_data['send_time']
                                if ts > 1e12:  # milliseconds
                                    ts = ts / 1000
                                processed_msg['timestamp'] = datetime.fromtimestamp(ts)
                            except:
                                pass
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # Standard text fields if not already extracted
            if not text_extracted:
                text_fields = ['text', '_std_text', 'message', 'content', 'body', 'msg_data', 'raw_content']
                for field in text_fields:
                    if field in msg and msg[field]:
                        text = str(msg[field])
                        # Clean JSON-like structures in message data
                        if field == 'msg_data' and text.startswith('{'):
                            try:
                                json_data = json.loads(text)
                                if 'raw_content' in json_data:
                                    text = json_data['raw_content']
                            except:
                                pass
                        processed_msg['text'] = self._clean_text(text)
                        processed_msg['message_length'] = len(processed_msg['text'])
                        text_extracted = True
                        break
            
            # Extract timestamp
            time_fields = ['timestamp', '_std_timestamp', 'time', 'created', 'sent', 'send_time']
            for field in time_fields:
                if field in msg and msg[field]:
                    try:
                        ts = msg[field]
                        if isinstance(ts, (int, float)):
                            if ts > 1e12:  # milliseconds
                                ts = ts / 1000
                            processed_msg['timestamp'] = datetime.fromtimestamp(ts)
                        break
                    except:
                        continue
            
            # Extract user/conversation info
            user_fields = ['user_id', '_std_user_id', 'sender', 'from_user']
            for field in user_fields:
                if field in msg and msg[field]:
                    processed_msg['user_id'] = str(msg[field])
                    break
            
            conv_fields = ['conversation_id', '_std_conversation_id', 'chat_id', 'thread_id']
            for field in conv_fields:
                if field in msg and msg[field]:
                    processed_msg['conversation_id'] = str(msg[field])
                    break
            
            # Extract timestamp if not already extracted
            if not processed_msg['timestamp']:
                time_fields = ['timestamp', '_std_timestamp', 'time', 'created', 'sent', 'send_time', 'latest_update_time']
                for field in time_fields:
                    if field in msg and msg[field]:
                        try:
                            ts = msg[field]
                            if isinstance(ts, (int, float)):
                                if ts > 1e12:  # milliseconds
                                    ts = ts / 1000
                                processed_msg['timestamp'] = datetime.fromtimestamp(ts)
                                break
                        except:
                            continue
            
            if processed_msg['text']:  # Only add if we have text content
                processed_messages.append(processed_msg)
        
        return pd.DataFrame(processed_messages)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep emoticons and basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-*()"\']', ' ', text)
        return text.strip()
    
    def _analyze_app_messages(self, df: pd.DataFrame, app_name: str) -> Dict[str, Any]:
        """Perform comprehensive analysis on messages from a specific app."""
        analysis_results = {}
        
        # Content Analysis
        analysis_results['content_analysis'] = self._analyze_content(df)
        
        # Temporal Analysis
        analysis_results['temporal_analysis'] = self._analyze_temporal_patterns(df)
        
        # Behavioral Analysis
        analysis_results['behavioral_analysis'] = self._analyze_behavioral_patterns(df)
        
        # Sentiment Analysis
        if self.sentiment_analyzer:
            analysis_results['sentiment_analysis'] = self._analyze_sentiment(df)
        
        # Topic Analysis
        if SKLEARN_AVAILABLE:
            analysis_results['topic_analysis'] = self._analyze_topics(df)
        
        # Psychological Indicators
        analysis_results['psychological_indicators'] = self._analyze_psychological_patterns(df)
        
        # Communication Patterns
        analysis_results['communication_patterns'] = self._analyze_communication_patterns(df)
        
        return analysis_results
    
    def _analyze_content(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze message content characteristics."""
        if df.empty or 'text' not in df.columns:
            return {'error': 'No text data available'}
        
        texts = df['text'].astype(str)
        
        # Basic statistics
        message_lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        # Language patterns
        question_count = texts.str.count(r'\?').sum()
        exclamation_count = texts.str.count(r'!').sum()
        asterisk_actions = texts.str.count(r'\*[^*]*\*').sum()  # *action* patterns
        
        # Emotional expressions
        emotional_patterns = {
            'love_affection': len(texts[texts.str.contains(r'\b(love|adore|cherish|affection|heart)\b', case=False, na=False)]),
            'sexual_content': len(texts[texts.str.contains(r'\b(sexy|seductive|intimate|desire|pleasure)\b', case=False, na=False)]),
            'loneliness_indicators': len(texts[texts.str.contains(r'\b(lonely|alone|sad|depressed|empty)\b', case=False, na=False)]),
            'excitement': len(texts[texts.str.contains(r'\b(excited|amazing|wonderful|incredible)\b', case=False, na=False)])
        }
        
        return {
            'message_statistics': {
                'total_messages': len(texts),
                'avg_message_length': float(message_lengths.mean()),
                'avg_word_count': float(word_counts.mean()),
                'longest_message': int(message_lengths.max()),
                'shortest_message': int(message_lengths.min()),
                'message_length_std': float(message_lengths.std())
            },
            'language_patterns': {
                'questions_asked': int(question_count),
                'exclamations_used': int(exclamation_count),
                'action_descriptions': int(asterisk_actions),
                'question_frequency': float(question_count / len(texts)),
                'exclamation_frequency': float(exclamation_count / len(texts)),
                'action_frequency': float(asterisk_actions / len(texts))
            },
            'emotional_content': emotional_patterns,
            'content_themes': self._identify_content_themes(texts)
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal communication patterns."""
        if df.empty or 'timestamp' not in df.columns:
            return {'error': 'No timestamp data available'}
        
        # Filter out null timestamps
        df_with_time = df.dropna(subset=['timestamp'])
        if df_with_time.empty:
            return {'error': 'No valid timestamps found'}
        
        timestamps = pd.to_datetime(df_with_time['timestamp'])
        
        # Activity patterns by hour and day
        hours = timestamps.dt.hour
        days = timestamps.dt.dayofweek
        
        activity_patterns = {
            'peak_activity_hour': int(hours.mode().iloc[0]) if not hours.empty else 12,
            'peak_activity_day': int(days.mode().iloc[0]) if not days.empty else 0,
            'hourly_distribution': hours.value_counts().to_dict(),
            'daily_distribution': days.value_counts().to_dict()
        }
        
        # Usage intensity over time
        daily_counts = timestamps.dt.date.value_counts().sort_index()
        
        usage_intensity = {
            'total_active_days': len(daily_counts),
            'avg_daily_messages': float(daily_counts.mean()),
            'max_daily_messages': int(daily_counts.max()),
            'daily_variance': float(daily_counts.var()),
            'usage_consistency': float(1 / (1 + daily_counts.std())) if daily_counts.std() > 0 else 1.0
        }
        
        # Communication frequency analysis
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dt.total_seconds().dropna()
            frequency_analysis = {
                'avg_message_interval_hours': float(time_diffs.mean() / 3600),
                'median_message_interval_hours': float(time_diffs.median() / 3600),
                'rapid_fire_sessions': int((time_diffs < 60).sum()),  # Messages within 1 minute
                'long_gaps': int((time_diffs > 86400).sum())  # Gaps > 24 hours
            }
        else:
            frequency_analysis = {}
        
        return {
            'activity_patterns': activity_patterns,
            'usage_intensity': usage_intensity,
            'frequency_analysis': frequency_analysis,
            'temporal_span': {
                'first_message': timestamps.min().isoformat(),
                'last_message': timestamps.max().isoformat(),
                'total_days': (timestamps.max() - timestamps.min()).days
            }
        }
    
    def _analyze_behavioral_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze behavioral patterns in communication."""
        if df.empty:
            return {'error': 'No data available'}
        
        behavioral_indicators = {}
        
        # Response patterns (if we have conversation flow)
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            
            # Interaction style
            behavioral_indicators['interaction_style'] = {
                'initiator_behavior': self._analyze_initiator_patterns(texts),
                'response_style': self._analyze_response_patterns(texts),
                'conversation_depth': self._analyze_conversation_depth(texts)
            }
            
            # Emotional investment
            behavioral_indicators['emotional_investment'] = {
                'emotional_intensity': self._calculate_emotional_intensity(texts),
                'vulnerability_indicators': self._identify_vulnerability_patterns(texts),
                'attachment_signs': self._identify_attachment_patterns(texts)
            }
            
            # Communication preferences
            behavioral_indicators['communication_preferences'] = {
                'message_style': self._analyze_message_style(texts),
                'expression_patterns': self._analyze_expression_patterns(texts)
            }
        
        return behavioral_indicators
    
    def _analyze_sentiment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform sentiment analysis on messages."""
        if not self.sentiment_analyzer or df.empty or 'text' not in df.columns:
            return {'error': 'Sentiment analysis not available'}
        
        texts = df['text'].astype(str)
        sentiments = []
        
        for text in texts:
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiments.append(scores)
        
        sentiment_df = pd.DataFrame(sentiments)
        
        return {
            'overall_sentiment': {
                'positive': float(sentiment_df['pos'].mean()),
                'negative': float(sentiment_df['neg'].mean()),
                'neutral': float(sentiment_df['neu'].mean()),
                'compound': float(sentiment_df['compound'].mean())
            },
            'sentiment_distribution': {
                'positive_messages': int((sentiment_df['compound'] > 0.05).sum()),
                'negative_messages': int((sentiment_df['compound'] < -0.05).sum()),
                'neutral_messages': int((sentiment_df['compound'].between(-0.05, 0.05)).sum())
            },
            'emotional_volatility': float(sentiment_df['compound'].std()),
            'extreme_sentiments': {
                'most_positive_score': float(sentiment_df['compound'].max()),
                'most_negative_score': float(sentiment_df['compound'].min())
            }
        }
    
    def _analyze_topics(self, df: pd.DataFrame, n_topics: int = 5) -> Dict[str, Any]:
        """Extract conversation topics using topic modeling."""
        if not SKLEARN_AVAILABLE or df.empty or 'text' not in df.columns:
            return {'error': 'Topic analysis not available'}
        
        texts = df['text'].astype(str).tolist()
        
        if len(texts) < 10:
            return {'error': 'Insufficient data for topic modeling'}
        
        try:
            # Vectorize texts
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            text_vectors = vectorizer.fit_transform(texts)
            
            # Apply LDA
            n_components = min(n_topics, len(texts)//2, text_vectors.shape[1])
            lda = LatentDirichletAllocation(
                n_components=n_components,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(text_vectors)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
                topics.append({
                    'topic_id': topic_idx,
                    'keywords': top_words[:5],
                    'weight': float(topic.sum())
                })
            
            return {
                'topics': topics,
                'dominant_topic_words': topics[0]['keywords'] if topics else [],
                'topic_diversity': len(topics)
            }
            
        except Exception as e:
            return {'error': f'Topic analysis failed: {str(e)}'}
    
    def _analyze_psychological_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze psychological indicators in communication."""
        if df.empty or 'text' not in df.columns:
            return {'error': 'No text data for psychological analysis'}
        
        texts = df['text'].astype(str)
        
        # Psychological indicators
        indicators = {
            'dependency_indicators': self._assess_dependency_patterns(texts),
            'fantasy_engagement': self._assess_fantasy_engagement(texts),
            'reality_detachment': self._assess_reality_detachment(texts),
            'emotional_regulation': self._assess_emotional_regulation(texts),
            'social_connection_seeking': self._assess_social_connection_patterns(texts)
        }
        
        # Risk factors
        risk_assessment = {
            'isolation_risk': self._assess_isolation_risk(texts),
            'addiction_indicators': self._assess_addiction_indicators(df),
            'emotional_instability': self._assess_emotional_instability(texts)
        }
        
        return {
            'psychological_indicators': indicators,
            'risk_assessment': risk_assessment,
            'overall_psychological_health': self._calculate_psychological_health_score(indicators, risk_assessment)
        }
    
    def _analyze_communication_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze communication patterns and relationship dynamics."""
        if df.empty:
            return {'error': 'No data for communication analysis'}
        
        patterns = {}
        
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            
            # Communication style analysis
            patterns['communication_style'] = {
                'formality_level': self._assess_formality_level(texts),
                'intimacy_level': self._assess_intimacy_level(texts),
                'directness': self._assess_communication_directness(texts)
            }
            
            # Relationship progression
            if 'timestamp' in df.columns and not df['timestamp'].isna().all():
                patterns['relationship_progression'] = self._analyze_relationship_progression(df)
        
        return patterns
    
    # Helper methods for psychological analysis
    def _assess_dependency_patterns(self, texts: pd.Series) -> Dict[str, float]:
        """Assess dependency indicators in text."""
        dependency_keywords = r'\b(need|depend|can\'t live|desperate|must have|require)\b'
        return {
            'dependency_frequency': float(texts.str.contains(dependency_keywords, case=False, na=False).mean()),
            'dependency_intensity': float(texts.str.count(dependency_keywords).mean())
        }
    
    def _assess_fantasy_engagement(self, texts: pd.Series) -> Dict[str, float]:
        """Assess level of fantasy engagement."""
        fantasy_indicators = r'\b(imagine|pretend|fantasy|dream|wish|if only)\b'
        roleplay_indicators = r'\*[^*]*\*'
        
        return {
            'fantasy_language': float(texts.str.contains(fantasy_indicators, case=False, na=False).mean()),
            'roleplay_frequency': float(texts.str.contains(roleplay_indicators, na=False).mean())
        }
    
    def _assess_reality_detachment(self, texts: pd.Series) -> Dict[str, float]:
        """Assess indicators of reality detachment."""
        reality_confusion = r'\b(real|actually|truly|genuinely|for real)\b'
        questioning_reality = r'\b(are you real|is this real|feels real)\b'
        
        return {
            'reality_questioning': float(texts.str.contains(questioning_reality, case=False, na=False).mean()),
            'reality_emphasis': float(texts.str.contains(reality_confusion, case=False, na=False).mean())
        }
    
    def _assess_emotional_regulation(self, texts: pd.Series) -> Dict[str, float]:
        """Assess emotional regulation patterns."""
        emotional_extremes = r'\b(always|never|everything|nothing|completely|totally)\b'
        emotional_words = r'\b(feel|feeling|emotion|mood|sad|happy|angry|scared)\b'
        
        return {
            'emotional_extremes': float(texts.str.contains(emotional_extremes, case=False, na=False).mean()),
            'emotional_awareness': float(texts.str.contains(emotional_words, case=False, na=False).mean())
        }
    
    def _assess_social_connection_patterns(self, texts: pd.Series) -> Dict[str, float]:
        """Assess social connection and isolation patterns."""
        social_keywords = r'\b(friend|family|people|social|together|alone|lonely)\b'
        isolation_keywords = r'\b(alone|lonely|isolated|nobody|no one)\b'
        
        return {
            'social_references': float(texts.str.contains(social_keywords, case=False, na=False).mean()),
            'isolation_expressions': float(texts.str.contains(isolation_keywords, case=False, na=False).mean())
        }
    
    def _calculate_psychological_health_score(self, indicators: Dict, risk_assessment: Dict) -> Dict[str, float]:
        """Calculate overall psychological health indicators."""
        # This is a simplified scoring system - in practice, this would be more sophisticated
        health_score = 0.5  # Neutral baseline
        
        # Adjust based on indicators (simplified)
        if 'dependency_indicators' in indicators:
            health_score -= indicators['dependency_indicators'].get('dependency_frequency', 0) * 0.2
        
        if 'reality_detachment' in indicators:
            health_score -= indicators['reality_detachment'].get('reality_questioning', 0) * 0.3
        
        return {
            'overall_health_score': max(0.0, min(1.0, float(health_score))),
            'risk_level': 'high' if health_score < 0.3 else 'medium' if health_score < 0.7 else 'low'
        }
    
    # Additional helper methods (simplified implementations)
    def _identify_content_themes(self, texts: pd.Series) -> Dict[str, int]:
        """Identify major content themes."""
        themes = {
            'romantic': len(texts[texts.str.contains(r'\b(love|romantic|relationship|date)\b', case=False, na=False)]),
            'sexual': len(texts[texts.str.contains(r'\b(sexy|sexual|intimate|pleasure)\b', case=False, na=False)]),
            'emotional_support': len(texts[texts.str.contains(r'\b(comfort|support|understand|care)\b', case=False, na=False)]),
            'daily_life': len(texts[texts.str.contains(r'\b(work|school|day|today|yesterday)\b', case=False, na=False)])
        }
        return themes
    
    def _analyze_initiator_patterns(self, texts: pd.Series) -> Dict[str, float]:
        """Analyze conversation initiation patterns."""
        greetings = r'\b(hello|hi|hey|good morning|good evening)\b'
        return {
            'greeting_frequency': float(texts.str.contains(greetings, case=False, na=False).mean())
        }
    
    def _analyze_response_patterns(self, texts: pd.Series) -> Dict[str, float]:
        """Analyze response patterns."""
        return {
            'avg_response_length': float(texts.str.len().mean())
        }
    
    def _analyze_conversation_depth(self, texts: pd.Series) -> Dict[str, float]:
        """Analyze conversation depth and complexity."""
        return {
            'avg_complexity': float(texts.str.split().str.len().mean())
        }
    
    def _calculate_emotional_intensity(self, texts: pd.Series) -> float:
        """Calculate emotional intensity score."""
        emotional_words = texts.str.count(r'\b(love|hate|amazing|terrible|wonderful|awful)\b')
        return float(emotional_words.mean())
    
    def _identify_vulnerability_patterns(self, texts: pd.Series) -> Dict[str, float]:
        """Identify vulnerability indicators."""
        vulnerable_keywords = r'\b(vulnerable|hurt|pain|scared|afraid|insecure)\b'
        return {
            'vulnerability_frequency': float(texts.str.contains(vulnerable_keywords, case=False, na=False).mean())
        }
    
    def _identify_attachment_patterns(self, texts: pd.Series) -> Dict[str, float]:
        """Identify attachment behavior patterns."""
        attachment_keywords = r'\b(miss|think about|can\'t stop|always|forever)\b'
        return {
            'attachment_frequency': float(texts.str.contains(attachment_keywords, case=False, na=False).mean())
        }
    
    def _analyze_message_style(self, texts: pd.Series) -> Dict[str, float]:
        """Analyze message style characteristics."""
        return {
            'avg_message_length': float(texts.str.len().mean()),
            'punctuation_usage': float(texts.str.count(r'[.!?]').mean())
        }
    
    def _analyze_expression_patterns(self, texts: pd.Series) -> Dict[str, float]:
        """Analyze expression patterns."""
        return {
            'emoji_usage': float(texts.str.count(r'[😀-🿿]|:\)|:\(|:D').mean()),
            'caps_usage': float(texts.str.count(r'[A-Z]').mean())
        }
    
    def _assess_isolation_risk(self, texts: pd.Series) -> Dict[str, float]:
        """Assess isolation risk factors."""
        isolation_keywords = r'\b(alone|lonely|isolated|nobody|empty|void)\b'
        return {
            'isolation_language': float(texts.str.contains(isolation_keywords, case=False, na=False).mean())
        }
    
    def _assess_addiction_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Assess potential addiction indicators."""
        # Simple temporal analysis for addiction patterns
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            timestamps = pd.to_datetime(df['timestamp'])
            daily_counts = timestamps.dt.date.value_counts()
            return {
                'high_frequency_days': float((daily_counts > daily_counts.quantile(0.9)).mean())
            }
        return {}
    
    def _assess_emotional_instability(self, texts: pd.Series) -> Dict[str, float]:
        """Assess emotional instability indicators."""
        extreme_emotions = r'\b(ecstatic|devastated|furious|depressed|manic)\b'
        return {
            'extreme_emotion_frequency': float(texts.str.contains(extreme_emotions, case=False, na=False).mean())
        }
    
    def _assess_formality_level(self, texts: pd.Series) -> float:
        """Assess communication formality level."""
        formal_indicators = texts.str.contains(r'\b(please|thank you|sir|madam|certainly)\b', case=False, na=False)
        return float(formal_indicators.mean())
    
    def _assess_intimacy_level(self, texts: pd.Series) -> float:
        """Assess intimacy level in communication."""
        intimate_indicators = texts.str.contains(r'\b(baby|honey|darling|sweetheart|intimate|close)\b', case=False, na=False)
        return float(intimate_indicators.mean())
    
    def _assess_communication_directness(self, texts: pd.Series) -> float:
        """Assess directness of communication."""
        direct_indicators = texts.str.contains(r'\b(want|need|will|should|must)\b', case=False, na=False)
        return float(direct_indicators.mean())
    
    def _analyze_relationship_progression(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationship progression over time."""
        if 'timestamp' not in df.columns or 'text' not in df.columns:
            return {}
        
        # Sort by timestamp and analyze progression
        df_sorted = df.sort_values('timestamp')
        texts = df_sorted['text'].astype(str)
        
        # Simple progression analysis
        early_messages = texts.head(len(texts)//3)
        late_messages = texts.tail(len(texts)//3)
        
        early_intimacy = self._assess_intimacy_level(early_messages)
        late_intimacy = self._assess_intimacy_level(late_messages)
        
        return {
            'intimacy_progression': float(late_intimacy - early_intimacy),
            'relationship_development': 'increasing' if late_intimacy > early_intimacy else 'stable' if abs(late_intimacy - early_intimacy) < 0.1 else 'decreasing'
        }
    
    def _cross_application_analysis(self, ml_analysis: Dict) -> Dict[str, Any]:
        """Perform cross-application behavioral analysis."""
        cross_analysis = {
            'multi_app_usage': len([app for app in ml_analysis.get('content_analysis', {}) if app]),
            'behavioral_consistency': {},
            'platform_preferences': {}
        }
        
        # Analyze consistency across platforms
        if len(ml_analysis.get('sentiment_analysis', {})) > 1:
            sentiments = []
            for app_sentiment in ml_analysis['sentiment_analysis'].values():
                if 'overall_sentiment' in app_sentiment:
                    sentiments.append(app_sentiment['overall_sentiment']['compound'])
            
            if sentiments:
                cross_analysis['behavioral_consistency']['sentiment_variance'] = float(np.var(sentiments))
        
        return cross_analysis
    
    def _generate_behavioral_profile(self, ml_analysis: Dict) -> Dict[str, Any]:
        """Generate final behavioral profile summary."""
        profile = {
            'profile_timestamp': datetime.now().isoformat(),
            'user_characteristics': {},
            'risk_indicators': {},
            'psychological_assessment': {},
            'recommendations': []
        }
        
        # Aggregate characteristics from all analyses
        total_messages = 0
        apps_analyzed = []
        
        for app_name, content_analysis in ml_analysis.get('content_analysis', {}).items():
            if 'message_statistics' in content_analysis:
                total_messages += content_analysis['message_statistics']['total_messages']
                apps_analyzed.append(app_name)
        
        profile['user_characteristics'] = {
            'total_messages_analyzed': total_messages,
            'applications_used': apps_analyzed,
            'analysis_completeness': len(apps_analyzed)
        }
        
        # Generate recommendations based on analysis
        if total_messages > 100:
            profile['recommendations'].append("High usage detected - consider monitoring for addiction patterns")
        
        return profile
