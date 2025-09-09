"""
Text analysis module for extracting insights from conversation data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime, timedelta
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')


class TextAnalyzer:
    """Analyzer for extracting insights from text conversations."""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.nlp = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP tools."""
        try:
            # Initialize NLTK sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            print("NLTK VADER lexicon not found. Run: nltk.download('vader_lexicon')")
        
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    
    def analyze_conversations(self, conversations_df: pd.DataFrame, 
                            text_column: str = 'message', 
                            timestamp_column: str = 'timestamp',
                            user_column: str = 'user_id') -> Dict[str, Any]:
        """
        Comprehensive analysis of conversation data.
        
        Args:
            conversations_df: DataFrame containing conversation data
            text_column: Name of column containing message text
            timestamp_column: Name of column containing timestamps
            user_column: Name of column containing user identifier
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'sentiment_analysis': {},
            'topic_analysis': {},
            'conversation_patterns': {},
            'emotional_patterns': {},
            'linguistic_analysis': {},
            'temporal_analysis': {},
            'user_behavior': {}
        }
        
        # Prepare data
        df = conversations_df.copy()
        if text_column not in df.columns:
            print(f"Warning: Text column '{text_column}' not found")
            return results
        
        # Clean and preprocess text
        df['cleaned_text'] = df[text_column].astype(str).apply(self._clean_text)
        df = df[df['cleaned_text'].str.len() > 0]  # Remove empty messages
        
        if len(df) == 0:
            print("No valid text data found")
            return results
        
        # Sentiment Analysis
        results['sentiment_analysis'] = self._analyze_sentiment(df, 'cleaned_text')
        
        # Topic Analysis
        results['topic_analysis'] = self._analyze_topics(df, 'cleaned_text')
        
        # Conversation Patterns
        results['conversation_patterns'] = self._analyze_conversation_patterns(df, text_column)
        
        # Emotional Patterns
        results['emotional_patterns'] = self._analyze_emotional_patterns(df, 'cleaned_text')
        
        # Linguistic Analysis
        results['linguistic_analysis'] = self._analyze_linguistics(df, 'cleaned_text')
        
        # Temporal Analysis
        if timestamp_column in df.columns:
            results['temporal_analysis'] = self._analyze_temporal_patterns(df, timestamp_column)
        
        # User Behavior
        if user_column in df.columns:
            results['user_behavior'] = self._analyze_user_behavior(df, user_column, 'cleaned_text')
        
        return results
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()
    
    def _analyze_sentiment(self, df: pd.DataFrame, text_col: str) -> Dict[str, Any]:
        """Analyze sentiment patterns in conversations."""
        if self.sentiment_analyzer is None:
            return {'error': 'Sentiment analyzer not available'}
        
        sentiments = []
        for text in df[text_col]:
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
            'sentiment_trends': self._calculate_sentiment_trends(sentiment_df),
            'extreme_sentiments': {
                'most_positive': float(sentiment_df['compound'].max()),
                'most_negative': float(sentiment_df['compound'].min()),
                'sentiment_variance': float(sentiment_df['compound'].var())
            }
        }
    
    def _analyze_topics(self, df: pd.DataFrame, text_col: str, n_topics: int = 5) -> Dict[str, Any]:
        """Extract and analyze conversation topics using LDA."""
        texts = df[text_col].tolist()
        
        if len(texts) < 10:  # Need minimum documents for topic modeling
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
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(texts)//2),
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
                    'top_words': top_words,
                    'weight': float(topic.sum())
                })
            
            # Get document-topic distributions
            doc_topic_probs = lda.transform(text_vectors)
            topic_assignments = doc_topic_probs.argmax(axis=1)
            
            return {
                'topics': topics,
                'topic_distribution': [int(x) for x in np.bincount(topic_assignments)],
                'dominant_topics': [int(x) for x in Counter(topic_assignments).most_common(3)]
            }
            
        except Exception as e:
            return {'error': f'Topic analysis failed: {str(e)}'}
    
    def _analyze_conversation_patterns(self, df: pd.DataFrame, text_col: str) -> Dict[str, Any]:
        """Analyze conversation patterns and characteristics."""
        texts = df[text_col].astype(str)
        
        # Basic statistics
        message_lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        # Language patterns
        question_count = texts.str.count(r'\?').sum()
        exclamation_count = texts.str.count(r'!').sum()
        caps_count = texts.str.count(r'[A-Z]').sum()
        
        # Conversation flow
        conversation_starters = self._identify_conversation_starters(texts)
        conversation_enders = self._identify_conversation_enders(texts)
        
        return {
            'message_statistics': {
                'total_messages': len(texts),
                'avg_message_length': float(message_lengths.mean()),
                'avg_word_count': float(word_counts.mean()),
                'longest_message': int(message_lengths.max()),
                'shortest_message': int(message_lengths.min())
            },
            'language_patterns': {
                'questions_asked': int(question_count),
                'exclamations_used': int(exclamation_count),
                'capital_letters': int(caps_count),
                'question_frequency': float(question_count / len(texts)),
                'exclamation_frequency': float(exclamation_count / len(texts))
            },
            'conversation_flow': {
                'common_starters': conversation_starters,
                'common_enders': conversation_enders
            }
        }
    
    def _analyze_emotional_patterns(self, df: pd.DataFrame, text_col: str) -> Dict[str, Any]:
        """Analyze emotional expressions and patterns."""
        texts = df[text_col]
        
        # Emotional indicators
        emotion_patterns = {
            'love': r'\b(love|adore|cherish|affection)\b',
            'happiness': r'\b(happy|joy|excited|glad|wonderful)\b',
            'sadness': r'\b(sad|depressed|lonely|cry|tears)\b',
            'anger': r'\b(angry|mad|furious|annoyed|frustrated)\b',
            'fear': r'\b(scared|afraid|terrified|worried|anxious)\b',
            'surprise': r'\b(surprised|shocked|amazed|wow)\b'
        }
        
        emotion_counts = {}
        for emotion, pattern in emotion_patterns.items():
            emotion_counts[emotion] = texts.str.contains(pattern, case=False, regex=True).sum()
        
        # Relationship indicators
        relationship_terms = {
            'intimacy': r'\b(intimate|close|personal|private)\b',
            'commitment': r'\b(forever|always|never leave|stay together)\b',
            'affection': r'\b(kiss|hug|cuddle|touch|hold)\b',
            'future_planning': r'\b(future|plans|tomorrow|next|someday)\b'
        }
        
        relationship_counts = {}
        for term, pattern in relationship_terms.items():
            relationship_counts[term] = texts.str.contains(pattern, case=False, regex=True).sum()
        
        return {
            'emotional_expressions': emotion_counts,
            'relationship_indicators': relationship_counts,
            'emotional_intensity': self._calculate_emotional_intensity(texts),
            'emotional_progression': self._analyze_emotional_progression(df, text_col)
        }
    
    def _analyze_linguistics(self, df: pd.DataFrame, text_col: str) -> Dict[str, Any]:
        """Analyze linguistic patterns and complexity."""
        if self.nlp is None:
            return {'error': 'spaCy model not available'}
        
        texts = df[text_col].tolist()
        sample_size = min(100, len(texts))  # Analyze sample for performance
        sample_texts = texts[:sample_size]
        
        linguistic_features = {
            'avg_sentence_length': [],
            'lexical_diversity': [],
            'pos_tags': Counter(),
            'named_entities': Counter(),
            'dependency_patterns': Counter()
        }
        
        for text in sample_texts:
            if len(text.strip()) == 0:
                continue
                
            doc = self.nlp(text)
            
            # Sentence length
            sentences = list(doc.sents)
            if sentences:
                avg_sent_len = sum(len(sent.text.split()) for sent in sentences) / len(sentences)
                linguistic_features['avg_sentence_length'].append(avg_sent_len)
            
            # Lexical diversity (unique words / total words)
            words = [token.text.lower() for token in doc if token.is_alpha]
            if words:
                diversity = len(set(words)) / len(words)
                linguistic_features['lexical_diversity'].append(diversity)
            
            # POS tags
            for token in doc:
                linguistic_features['pos_tags'][token.pos_] += 1
            
            # Named entities
            for ent in doc.ents:
                linguistic_features['named_entities'][ent.label_] += 1
        
        return {
            'average_sentence_length': float(np.mean(linguistic_features['avg_sentence_length'])) if linguistic_features['avg_sentence_length'] else 0,
            'lexical_diversity': float(np.mean(linguistic_features['lexical_diversity'])) if linguistic_features['lexical_diversity'] else 0,
            'pos_distribution': dict(linguistic_features['pos_tags'].most_common(10)),
            'entity_types': dict(linguistic_features['named_entities'].most_common(10)),
            'linguistic_complexity': self._calculate_linguistic_complexity(linguistic_features)
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze temporal patterns in conversations."""
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except:
            return {'error': 'Could not parse timestamps'}
        
        df = df.sort_values(timestamp_col)
        
        # Basic temporal stats
        first_message = df[timestamp_col].min()
        last_message = df[timestamp_col].max()
        total_duration = (last_message - first_message).total_seconds() / (24 * 3600)  # days
        
        # Activity patterns
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['month'] = df[timestamp_col].dt.month
        
        hourly_activity = df['hour'].value_counts().sort_index()
        daily_activity = df['day_of_week'].value_counts().sort_index()
        monthly_activity = df['month'].value_counts().sort_index()
        
        # Conversation frequency
        daily_messages = df.groupby(df[timestamp_col].dt.date).size()
        
        return {
            'duration_stats': {
                'first_message': first_message.isoformat(),
                'last_message': last_message.isoformat(),
                'total_days': float(total_duration),
                'average_daily_messages': float(daily_messages.mean())
            },
            'activity_patterns': {
                'hourly_distribution': hourly_activity.to_dict(),
                'daily_distribution': daily_activity.to_dict(),
                'monthly_distribution': monthly_activity.to_dict(),
                'peak_hour': int(hourly_activity.idxmax()),
                'peak_day': int(daily_activity.idxmax())
            },
            'conversation_frequency': {
                'most_active_day': float(daily_messages.max()),
                'least_active_day': float(daily_messages.min()),
                'frequency_variance': float(daily_messages.var())
            }
        }
    
    def _analyze_user_behavior(self, df: pd.DataFrame, user_col: str, text_col: str) -> Dict[str, Any]:
        """Analyze user-specific behavior patterns."""
        user_stats = df.groupby(user_col).agg({
            text_col: ['count', lambda x: x.str.len().mean()],
        }).round(2)
        
        user_stats.columns = ['message_count', 'avg_message_length']
        
        return {
            'user_statistics': user_stats.to_dict('index'),
            'most_active_user': user_stats['message_count'].idxmax(),
            'user_engagement': self._calculate_user_engagement(df, user_col, text_col)
        }
    
    # Helper methods
    def _calculate_sentiment_trends(self, sentiment_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment trend metrics."""
        compound_scores = sentiment_df['compound'].values
        if len(compound_scores) < 2:
            return {'trend': 0.0, 'volatility': 0.0}
        
        # Simple linear trend
        x = np.arange(len(compound_scores))
        trend = np.polyfit(x, compound_scores, 1)[0]
        volatility = np.std(compound_scores)
        
        return {
            'trend': float(trend),
            'volatility': float(volatility)
        }
    
    def _identify_conversation_starters(self, texts: pd.Series) -> List[str]:
        """Identify common conversation starters."""
        starters = []
        for text in texts.head(50):  # Sample first 50 messages
            words = str(text).lower().split()
            if words:
                first_few = ' '.join(words[:3])
                starters.append(first_few)
        
        return [item for item, count in Counter(starters).most_common(5)]
    
    def _identify_conversation_enders(self, texts: pd.Series) -> List[str]:
        """Identify common conversation enders."""
        enders = []
        for text in texts.tail(50):  # Sample last 50 messages
            words = str(text).lower().split()
            if words:
                last_few = ' '.join(words[-3:])
                enders.append(last_few)
        
        return [item for item, count in Counter(enders).most_common(5)]
    
    def _calculate_emotional_intensity(self, texts: pd.Series) -> float:
        """Calculate overall emotional intensity."""
        # Count emotional punctuation and capitalization
        exclamations = texts.str.count('!').sum()
        caps_words = texts.str.findall(r'\b[A-Z]{2,}\b').str.len().sum()
        total_words = texts.str.split().str.len().sum()
        
        if total_words == 0:
            return 0.0
        
        intensity = (exclamations + caps_words) / total_words
        return float(intensity)
    
    def _analyze_emotional_progression(self, df: pd.DataFrame, text_col: str) -> Dict[str, Any]:
        """Analyze how emotions change over time."""
        if len(df) < 10:
            return {'progression': 'insufficient_data'}
        
        # Split into segments and analyze sentiment
        segment_size = len(df) // 5
        segments = []
        
        for i in range(0, len(df), segment_size):
            segment = df.iloc[i:i+segment_size]
            if len(segment) > 0:
                avg_sentiment = self._get_segment_sentiment(segment[text_col])
                segments.append(avg_sentiment)
        
        return {
            'progression': segments,
            'overall_trend': 'improving' if segments[-1] > segments[0] else 'declining'
        }
    
    def _get_segment_sentiment(self, texts: pd.Series) -> float:
        """Get average sentiment for a text segment."""
        if self.sentiment_analyzer is None:
            return 0.0
        
        sentiments = []
        for text in texts:
            score = self.sentiment_analyzer.polarity_scores(str(text))
            sentiments.append(score['compound'])
        
        return float(np.mean(sentiments)) if sentiments else 0.0
    
    def _calculate_linguistic_complexity(self, features: Dict) -> float:
        """Calculate linguistic complexity score."""
        if not features['avg_sentence_length'] or not features['lexical_diversity']:
            return 0.0
        
        avg_sent_len = np.mean(features['avg_sentence_length'])
        avg_diversity = np.mean(features['lexical_diversity'])
        
        # Normalize and combine metrics
        complexity = (avg_sent_len / 20.0 + avg_diversity) / 2.0
        return float(min(complexity, 1.0))
    
    def _calculate_user_engagement(self, df: pd.DataFrame, user_col: str, text_col: str) -> Dict[str, float]:
        """Calculate user engagement metrics."""
        user_groups = df.groupby(user_col)
        
        engagement_metrics = {}
        for user, group in user_groups:
            message_count = len(group)
            avg_length = group[text_col].str.len().mean()
            
            # Simple engagement score
            engagement = (message_count * 0.6) + (avg_length * 0.4)
            engagement_metrics[user] = float(engagement)
        
        return engagement_metrics
