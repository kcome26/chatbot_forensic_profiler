"""
Main forensic analyzer that orchestrates all analysis modules.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .database_parsers import DatabaseManager
from .analysis.text_analyzer import TextAnalyzer
from .analysis.temporal_analyzer import TemporalAnalyzer


class ForensicAnalyzer:
    """
    Main forensic analyzer for AI girlfriend application databases.
    Coordinates database parsing, analysis, and profile generation.
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.text_analyzer = TextAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        
        self.database_data = {}
        self.analysis_results = {}
        self.forensic_profile = {}
        
    def load_database(self, db_path: str) -> bool:
        """
        Load and parse a database file.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Loading database: {db_path}")
        
        if not self.db_manager.load_database(db_path):
            print("Failed to load database")
            return False
        
        # Extract all data
        try:
            self.database_data = self.db_manager.extract_all_data()
            print(f"Successfully loaded database with {len(self.database_data['tables'])} tables")
            return True
        except Exception as e:
            print(f"Error extracting database data: {e}")
            return False
    
    def analyze_database(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the loaded database.
        
        Returns:
            Dictionary containing all analysis results
        """
        if not self.database_data:
            raise ValueError("No database loaded. Call load_database() first.")
        
        print("Starting comprehensive database analysis...")
        
        results = {
            'metadata_analysis': self._analyze_metadata(),
            'content_analysis': self._analyze_content(),
            'behavioral_analysis': self._analyze_behavior(),
            'temporal_analysis': self._analyze_temporal_patterns(),
            'relationship_analysis': self._analyze_relationships(),
            'risk_assessment': self._assess_risks()
        }
        
        self.analysis_results = results
        print("Analysis completed")
        return results
    
    def generate_forensic_profile(self) -> Dict[str, Any]:
        """
        Generate comprehensive forensic profile based on analysis results.
        
        Returns:
            Detailed forensic profile organized for investigative clarity
        """
        if not self.analysis_results:
            print("No analysis results available. Running analysis...")
            self.analyze_database()
        
        print("Generating forensic profile...")
        
        profile = {
            'investigation_summary': self._generate_investigation_summary(),
            'subject_identification': self._extract_subject_identification(),
            'key_evidence': self._extract_key_evidence(),
            'behavioral_assessment': self._generate_behavioral_assessment(),
            'psychological_profile': self._extract_psychological_indicators(),
            'risk_evaluation': self._extract_risk_factors(),
            'technical_analysis': self._generate_technical_analysis(),
            'investigative_recommendations': self._generate_investigation_insights()
        }
        
        self.forensic_profile = profile
        print("Forensic profile generated")
        return profile
    
    def _analyze_metadata(self) -> Dict[str, Any]:
        """Analyze database metadata for forensic insights."""
        metadata = self.database_data.get('metadata', {})
        
        return {
            'database_info': {
                'type': metadata.get('db_type', 'unknown'),
                'file_path': metadata.get('db_path', ''),
                'total_tables': metadata.get('total_tables', 0),
                'total_records': metadata.get('total_rows', 0),
                'extraction_time': metadata.get('extraction_time', ''),
                'file_size': metadata.get('file_size', 0)
            },
            'app_detection': metadata.get('app_info', {}),
            'technical_indicators': self._extract_technical_indicators(metadata)
        }
    
    def _analyze_content(self) -> Dict[str, Any]:
        """Analyze text content across all tables."""
        tables = self.database_data.get('tables', {})
        content_results = {}
        
        for table_name, df in tables.items():
            if df.empty:
                continue
            
            # Find text columns
            text_columns = []
            classifications = df.attrs.get('column_classifications', {})
            
            for col, classification in classifications.items():
                if classification in ['text', 'conversation', 'user_info']:
                    text_columns.append(col)
            
            if not text_columns:
                continue
            
            # Analyze each text column
            table_analysis = {}
            for col in text_columns:
                if col in df.columns and df[col].dtype == 'object':
                    try:
                        analysis = self.text_analyzer.analyze_conversations(
                            df, text_column=col
                        )
                        table_analysis[col] = analysis
                    except Exception as e:
                        print(f"Error analyzing {table_name}.{col}: {e}")
                        continue
            
            if table_analysis:
                content_results[table_name] = table_analysis
        
        return content_results
    
    def _analyze_behavior(self) -> Dict[str, Any]:
        """Analyze user behavioral patterns."""
        tables = self.database_data.get('tables', {})
        behavioral_results = {}
        
        for table_name, df in tables.items():
            if df.empty:
                continue
            
            # Look for conversation patterns
            classifications = df.attrs.get('column_classifications', {})
            
            # Find relevant columns
            message_col = None
            user_col = None
            timestamp_col = None
            
            for col, classification in classifications.items():
                if classification == 'conversation' and message_col is None:
                    message_col = col
                elif classification == 'id' and 'user' in col.lower():
                    user_col = col
                elif classification == 'timestamp':
                    timestamp_col = col
            
            if message_col and len(df) > 10:
                # Analyze conversation patterns
                behavior_analysis = {
                    'communication_style': self._analyze_communication_style(df, message_col),
                    'interaction_frequency': self._analyze_interaction_frequency(df, timestamp_col),
                    'engagement_patterns': self._analyze_engagement_patterns(df, message_col, timestamp_col)
                }
                
                if user_col:
                    behavior_analysis['user_dynamics'] = self._analyze_user_dynamics(df, user_col, message_col)
                
                behavioral_results[table_name] = behavior_analysis
        
        return behavioral_results
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns across all relevant tables."""
        tables = self.database_data.get('tables', {})
        temporal_results = {}
        
        for table_name, df in tables.items():
            if df.empty:
                continue
            
            classifications = df.attrs.get('column_classifications', {})
            
            # Find timestamp columns
            timestamp_cols = [col for col, cls in classifications.items() if cls == 'timestamp']
            message_col = next((col for col, cls in classifications.items() if cls == 'conversation'), None)
            user_col = next((col for col, cls in classifications.items() if cls == 'id' and 'user' in col.lower()), None)
            
            if timestamp_cols and len(df) > 10:
                for timestamp_col in timestamp_cols:
                    try:
                        analysis = self.temporal_analyzer.analyze_temporal_patterns(
                            df, 
                            timestamp_col=timestamp_col,
                            message_col=message_col or df.columns[0],
                            user_col=user_col
                        )
                        temporal_results[f"{table_name}_{timestamp_col}"] = analysis
                    except ZeroDivisionError as e:
                        import traceback
                        print(f"Division by zero error in temporal analysis for {table_name}.{timestamp_col}:")
                        traceback.print_exc()
                        continue
                    except Exception as e:
                        print(f"Error in temporal analysis for {table_name}.{timestamp_col}: {e}")
                        continue
        
        return temporal_results
    
    def _generate_investigation_summary(self) -> Dict[str, Any]:
        """Generate a clear investigation summary."""
        metadata = self.analysis_results.get('metadata_analysis', {})
        content_stats = self._get_content_statistics()
        
        return {
            'case_information': {
                'analysis_date': datetime.now().isoformat(),
                'database_source': metadata.get('database_info', {}).get('file_path', 'unknown'),
                'evidence_type': 'AI Companion Application Data',
                'total_data_points': metadata.get('database_info', {}).get('total_records', 0),
                'data_timespan': self._calculate_data_timespan()
            },
            'investigation_scope': {
                'primary_focus': 'User behavioral patterns and interaction analysis',
                'secondary_focus': 'Emotional dependency and psychological indicators',
                'evidence_quality': self._assess_evidence_quality(),
                'analysis_completeness': self._calculate_analysis_completeness()
            },
            'key_statistics': content_stats
        }
    
    def _extract_subject_identification(self) -> Dict[str, Any]:
        """Extract and organize subject identification information."""
        user_info = self._find_user_information()
        profile_data = self._extract_profile_data()
        
        return {
            'personal_information': {
                'name': user_info.get('name', 'Not Available'),
                'username': user_info.get('username', 'Not Available'),
                'age': user_info.get('age', 'Not Available'),
                'location': user_info.get('location', 'Not Available'),
                'contact_details': user_info.get('contact', 'Not Available')
            },
            'profile_details': {
                'relationship_status': profile_data.get('relationship_status', 'Unknown'),
                'interests': profile_data.get('interests', []),
                'preferences': profile_data.get('preferences', {}),
                'account_creation': profile_data.get('created_at', 'Unknown'),
                'last_activity': profile_data.get('last_active', 'Unknown')
            },
            'identification_confidence': self._calculate_identification_confidence(user_info, profile_data)
        }
    
    def _extract_key_evidence(self) -> Dict[str, Any]:
        """Extract key pieces of evidence including important messages."""
        messages = self._extract_significant_messages()
        behavioral_evidence = self._extract_behavioral_evidence()
        
        return {
            'significant_conversations': messages,
            'behavioral_indicators': behavioral_evidence,
            'timeline_events': self._extract_timeline_events(),
            'attachment_patterns': self._analyze_attachment_evidence()
        }
    
    def _generate_behavioral_assessment(self) -> Dict[str, Any]:
        """Generate organized behavioral assessment."""
        behavioral_data = self._extract_behavioral_patterns()
        temporal_data = self._extract_temporal_behavior()
        
        return {
            'communication_analysis': {
                'style': behavioral_data.get('interaction_style', {}),
                'frequency': behavioral_data.get('usage_patterns', {}),
                'emotional_expression': self._analyze_emotional_expression(),
                'language_complexity': self._calculate_language_complexity()
            },
            'usage_patterns': {
                'activity_rhythm': temporal_data.get('activity_rhythm', {}),
                'consistency': temporal_data.get('usage_consistency', 0),
                'peak_periods': temporal_data.get('peak_activity_periods', []),
                'behavioral_changes': temporal_data.get('behavioral_changes', [])
            },
            'engagement_level': {
                'overall_engagement': behavioral_data.get('engagement_level', 'unknown'),
                'dependency_indicators': self._assess_dependency_level(),
                'emotional_investment': self._assess_emotional_investment()
            }
        }
    
    def _generate_technical_analysis(self) -> Dict[str, Any]:
        """Generate technical analysis summary."""
        metadata = self.analysis_results.get('metadata_analysis', {})
        
        return {
            'database_analysis': {
                'structure': metadata.get('technical_indicators', {}),
                'app_identification': metadata.get('app_detection', {}),
                'data_integrity': self._assess_data_integrity()
            },
            'analysis_metadata': self._generate_profile_metadata()
        }
    
    def _analyze_relationships(self) -> Dict[str, Any]:
        """Analyze relationship dynamics and emotional patterns."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        
        relationship_metrics = {
            'emotional_intensity': 0,
            'intimacy_level': 0,
            'dependency_indicators': 0,
            'relationship_progression': 'unknown',
            'attachment_style': 'unknown'
        }
        
        # Aggregate emotional patterns across all tables
        total_emotional_expressions = {}
        total_relationship_indicators = {}
        
        for table_results in content_analysis.values():
            for column_results in table_results.values():
                emotional_patterns = column_results.get('emotional_patterns', {})
                
                # Aggregate emotional expressions
                for emotion, count in emotional_patterns.get('emotional_expressions', {}).items():
                    total_emotional_expressions[emotion] = total_emotional_expressions.get(emotion, 0) + count
                
                # Aggregate relationship indicators
                for indicator, count in emotional_patterns.get('relationship_indicators', {}).items():
                    total_relationship_indicators[indicator] = total_relationship_indicators.get(indicator, 0) + count
        
        # Calculate relationship metrics
        if total_emotional_expressions:
            relationship_metrics['emotional_intensity'] = sum(total_emotional_expressions.values()) / len(total_emotional_expressions)
        
        if total_relationship_indicators:
            relationship_metrics['intimacy_level'] = total_relationship_indicators.get('intimacy', 0) + total_relationship_indicators.get('affection', 0)
            relationship_metrics['dependency_indicators'] = total_relationship_indicators.get('commitment', 0)
        
        return {
            'emotional_patterns': total_emotional_expressions,
            'relationship_indicators': total_relationship_indicators,
            'relationship_metrics': relationship_metrics,
            'attachment_analysis': self._analyze_attachment_style(total_emotional_expressions, total_relationship_indicators)
        }
    
    def _assess_risks(self) -> Dict[str, Any]:
        """Assess potential risk factors and concerning patterns."""
        risk_assessment = {
            'risk_level': 'low',
            'risk_factors': [],
            'concerning_patterns': [],
            'recommendations': []
        }
        
        # Check for concerning behavioral patterns
        behavioral_analysis = self.analysis_results.get('behavioral_analysis', {})
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        relationship_analysis = self.analysis_results.get('relationship_analysis', {})
        
        # High usage frequency
        for analysis in temporal_analysis.values():
            usage_intensity = analysis.get('usage_intensity', {})
            daily_stats = usage_intensity.get('daily_stats', {})
            
            avg_daily_messages = daily_stats.get('average_daily_messages', 0)
            if avg_daily_messages > 100:
                risk_assessment['risk_factors'].append('excessive_usage')
                risk_assessment['concerning_patterns'].append(f"High daily message count: {avg_daily_messages:.1f}")
        
        # Unusual temporal patterns
        for analysis in temporal_analysis.values():
            circadian_patterns = analysis.get('circadian_patterns', {})
            night_percentage = circadian_patterns.get('circadian_distribution', {}).get('night', {}).get('percentage', 0)
            
            if night_percentage > 30:
                risk_assessment['risk_factors'].append('abnormal_sleep_patterns')
                risk_assessment['concerning_patterns'].append(f"High nighttime activity: {night_percentage:.1f}%")
        
        # Emotional dependency indicators
        relationship_metrics = relationship_analysis.get('relationship_metrics', {})
        dependency_score = relationship_metrics.get('dependency_indicators', 0)
        
        if dependency_score > 10:
            risk_assessment['risk_factors'].append('emotional_dependency')
            risk_assessment['concerning_patterns'].append(f"High dependency indicators: {dependency_score}")
        
        # Calculate overall risk level
        risk_count = len(risk_assessment['risk_factors'])
        if risk_count >= 3:
            risk_assessment['risk_level'] = 'high'
        elif risk_count >= 2:
            risk_assessment['risk_level'] = 'moderate'
        
        # Generate recommendations
        risk_assessment['recommendations'] = self._generate_risk_recommendations(risk_assessment['risk_factors'])
        
        return risk_assessment
    
    # Profile Generation Methods
    def _generate_profile_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the forensic profile."""
        metadata = self.database_data.get('metadata', {})
        
        return {
            'profile_generated': datetime.now().isoformat(),
            'database_source': metadata.get('db_path', ''),
            'app_identified': metadata.get('app_info', {}).get('detected_app', 'unknown'),
            'confidence_level': metadata.get('app_info', {}).get('confidence', 0.0),
            'analysis_completeness': self._calculate_analysis_completeness(),
            'data_quality_score': self._calculate_data_quality_score()
        }
    
    def _extract_user_characteristics(self) -> Dict[str, Any]:
        """Extract user characteristics from analysis results."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        
        characteristics = {
            'communication_style': 'unknown',
            'personality_traits': [],
            'interests_and_topics': [],
            'language_complexity': 0,
            'emotional_expression': 'moderate'
        }
        
        # Aggregate linguistic analysis
        total_complexity = []
        all_topics = []
        
        for table_results in content_analysis.values():
            for column_results in table_results.values():
                linguistic = column_results.get('linguistic_analysis', {})
                topics = column_results.get('topic_analysis', {})
                
                if 'linguistic_complexity' in linguistic:
                    total_complexity.append(linguistic['linguistic_complexity'])
                
                if 'topics' in topics:
                    for topic in topics['topics']:
                        all_topics.extend(topic.get('top_words', []))
        
        if total_complexity:
            characteristics['language_complexity'] = sum(total_complexity) / len(total_complexity)
        
        if all_topics:
            from collections import Counter
            topic_counts = Counter(all_topics)
            characteristics['interests_and_topics'] = [word for word, count in topic_counts.most_common(10)]
        
        return characteristics
    
    def _extract_behavioral_patterns(self) -> Dict[str, Any]:
        """Extract behavioral patterns from analysis results."""
        behavioral_analysis = self.analysis_results.get('behavioral_analysis', {})
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        
        patterns = {
            'usage_patterns': {},
            'interaction_style': {},
            'activity_rhythm': {},
            'engagement_level': 'moderate'
        }
        
        # Aggregate temporal patterns
        for analysis in temporal_analysis.values():
            activity_patterns = analysis.get('activity_patterns', {})
            usage_intensity = analysis.get('usage_intensity', {})
            
            patterns['usage_patterns'].update({
                'peak_hour': activity_patterns.get('peak_activity_hour'),
                'peak_day': activity_patterns.get('peak_activity_day'),
                'daily_average': usage_intensity.get('daily_stats', {}).get('average_daily_messages', 0)
            })
        
        return patterns
    
    def _extract_psychological_indicators(self) -> Dict[str, Any]:
        """Extract psychological indicators from analysis results."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        relationship_analysis = self.analysis_results.get('relationship_analysis', {})
        
        indicators = {
            'emotional_state': 'neutral',
            'social_needs': [],
            'coping_mechanisms': [],
            'attachment_indicators': [],
            'risk_indicators': []
        }
        
        # Analyze sentiment patterns
        total_sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_count = 0
        
        for table_results in content_analysis.values():
            for column_results in table_results.values():
                sentiment = column_results.get('sentiment_analysis', {}).get('overall_sentiment', {})
                if sentiment:
                    total_sentiment['positive'] += sentiment.get('positive', 0)
                    total_sentiment['negative'] += sentiment.get('negative', 0)
                    total_sentiment['neutral'] += sentiment.get('neutral', 0)
                    sentiment_count += 1
        
        if sentiment_count > 0:
            avg_positive = total_sentiment['positive'] / sentiment_count
            avg_negative = total_sentiment['negative'] / sentiment_count
            
            if avg_positive > 0.3:
                indicators['emotional_state'] = 'positive'
            elif avg_negative > 0.3:
                indicators['emotional_state'] = 'negative'
        
        # Analyze relationship patterns for psychological insights
        emotional_patterns = relationship_analysis.get('emotional_patterns', {})
        
        if emotional_patterns.get('love', 0) > 5:
            indicators['social_needs'].append('romantic_connection')
        if emotional_patterns.get('loneliness', 0) > 3:
            indicators['social_needs'].append('companionship')
        if emotional_patterns.get('sadness', 0) > 5:
            indicators['coping_mechanisms'].append('emotional_support_seeking')
        
        return indicators
    
    def _extract_relationship_dynamics(self) -> Dict[str, Any]:
        """Extract relationship dynamics from analysis results."""
        relationship_analysis = self.analysis_results.get('relationship_analysis', {})
        
        return {
            'attachment_style': relationship_analysis.get('attachment_analysis', {}).get('predicted_style', 'unknown'),
            'intimacy_level': relationship_analysis.get('relationship_metrics', {}).get('intimacy_level', 0),
            'dependency_level': relationship_analysis.get('relationship_metrics', {}).get('dependency_indicators', 0),
            'emotional_investment': self._calculate_emotional_investment(relationship_analysis),
            'relationship_progression': relationship_analysis.get('relationship_metrics', {}).get('relationship_progression', 'unknown')
        }
    
    def _extract_temporal_behavior(self) -> Dict[str, Any]:
        """Extract temporal behavioral patterns."""
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        
        # Aggregate temporal patterns across all analyses
        combined_patterns = {
            'chronotype': 'unknown',
            'activity_regularity': 0,
            'usage_consistency': 0,
            'peak_activity_periods': [],
            'behavioral_changes': []
        }
        
        for analysis in temporal_analysis.values():
            circadian = analysis.get('circadian_patterns', {})
            if 'chronotype' in circadian:
                combined_patterns['chronotype'] = circadian['chronotype']
            
            activity = analysis.get('activity_patterns', {})
            if 'activity_regularity' in activity:
                combined_patterns['activity_regularity'] = activity['activity_regularity'].get('overall_regularity', 0)
        
        return combined_patterns
    
    def _extract_risk_factors(self) -> Dict[str, Any]:
        """Extract risk factors from analysis results."""
        risk_assessment = self.analysis_results.get('risk_assessment', {})
        
        return {
            'overall_risk_level': risk_assessment.get('risk_level', 'low'),
            'identified_risks': risk_assessment.get('risk_factors', []),
            'concerning_patterns': risk_assessment.get('concerning_patterns', []),
            'mitigation_recommendations': risk_assessment.get('recommendations', [])
        }
    
    def _generate_investigation_insights(self) -> Dict[str, Any]:
        """Generate insights useful for investigation purposes."""
        return {
            'key_findings': self._summarize_key_findings(),
            'evidence_strength': self._assess_evidence_strength(),
            'further_investigation': self._suggest_further_investigation(),
            'timeline_reconstruction': self._reconstruct_timeline(),
            'behavioral_summary': self._create_behavioral_summary()
        }
    
    # Helper methods
    def _extract_technical_indicators(self, metadata: Dict) -> Dict[str, Any]:
        """Extract technical indicators from metadata."""
        return {
            'database_size': metadata.get('file_size', 0),
            'record_count': metadata.get('total_rows', 0),
            'table_structure': metadata.get('total_tables', 0),
            'modification_time': metadata.get('file_modified', ''),
            'app_confidence': metadata.get('app_info', {}).get('confidence', 0)
        }
    
    def _analyze_communication_style(self, df: pd.DataFrame, message_col: str) -> Dict[str, Any]:
        """Analyze communication style patterns."""
        if message_col not in df.columns:
            return {}
        
        messages = df[message_col].astype(str)
        
        return {
            'avg_message_length': float(messages.str.len().mean()),
            'question_frequency': float(messages.str.count(r'\?').sum() / len(messages)),
            'exclamation_frequency': float(messages.str.count(r'!').sum() / len(messages)),
            'capital_usage': float(messages.str.count(r'[A-Z]').sum() / messages.str.len().sum())
        }
    
    def _analyze_interaction_frequency(self, df: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze interaction frequency patterns."""
        if not timestamp_col or timestamp_col not in df.columns:
            return {}
        
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            daily_counts = df.groupby(df[timestamp_col].dt.date).size()
            
            return {
                'avg_daily_interactions': float(daily_counts.mean()),
                'max_daily_interactions': int(daily_counts.max()),
                'active_days': int(len(daily_counts)),
                'interaction_variance': float(daily_counts.var())
            }
        except:
            return {}
    
    def _analyze_engagement_patterns(self, df: pd.DataFrame, message_col: str, timestamp_col: str) -> Dict[str, Any]:
        """Analyze engagement patterns over time."""
        if not message_col or message_col not in df.columns:
            return {}
        
        engagement_score = df[message_col].astype(str).str.len().mean()
        
        return {
            'engagement_score': float(engagement_score),
            'engagement_level': 'high' if engagement_score > 50 else 'moderate' if engagement_score > 20 else 'low'
        }
    
    def _analyze_user_dynamics(self, df: pd.DataFrame, user_col: str, message_col: str) -> Dict[str, Any]:
        """Analyze user interaction dynamics."""
        if not user_col or user_col not in df.columns:
            return {}
        
        user_counts = df[user_col].value_counts()
        
        return {
            'total_users': len(user_counts),
            'primary_user': user_counts.index[0] if len(user_counts) > 0 else None,
            'user_distribution': user_counts.head(5).to_dict()
        }
    
    def _analyze_attachment_style(self, emotional_expressions: Dict, relationship_indicators: Dict) -> Dict[str, Any]:
        """Analyze attachment style based on emotional patterns."""
        # Simple heuristic-based attachment style analysis
        love_score = emotional_expressions.get('love', 0)
        fear_score = emotional_expressions.get('fear', 0)
        commitment_score = relationship_indicators.get('commitment', 0)
        intimacy_score = relationship_indicators.get('intimacy', 0)
        
        total_expressions = sum(emotional_expressions.values()) + sum(relationship_indicators.values())
        
        if total_expressions < 5:
            return {'predicted_style': 'insufficient_data', 'confidence': 0.0}
        
        # Simple classification logic
        if commitment_score > 5 and intimacy_score > 3:
            style = 'secure'
            confidence = 0.7
        elif fear_score > 3 and commitment_score > 3:
            style = 'anxious'
            confidence = 0.6
        elif intimacy_score < 2 and love_score < 3:
            style = 'avoidant'
            confidence = 0.5
        else:
            style = 'mixed'
            confidence = 0.4
        
        return {
            'predicted_style': style,
            'confidence': confidence,
            'supporting_evidence': {
                'love_expressions': love_score,
                'fear_expressions': fear_score,
                'commitment_indicators': commitment_score,
                'intimacy_indicators': intimacy_score
            }
        }
    
    def _generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on identified risk factors."""
        recommendations = []
        
        if 'excessive_usage' in risk_factors:
            recommendations.append("Monitor for signs of technology addiction or social isolation")
        
        if 'abnormal_sleep_patterns' in risk_factors:
            recommendations.append("Assess impact on sleep hygiene and daily functioning")
        
        if 'emotional_dependency' in risk_factors:
            recommendations.append("Evaluate emotional wellbeing and social support systems")
        
        if not recommendations:
            recommendations.append("Continue monitoring for changes in usage patterns")
        
        return recommendations
    
    def _calculate_analysis_completeness(self) -> float:
        """Calculate how complete the analysis is."""
        expected_sections = ['metadata_analysis', 'content_analysis', 'behavioral_analysis', 
                           'temporal_analysis', 'relationship_analysis', 'risk_assessment']
        
        completed_sections = sum(1 for section in expected_sections if section in self.analysis_results)
        return completed_sections / len(expected_sections)
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate data quality score."""
        tables = self.database_data.get('tables', {})
        
        if not tables:
            return 0.0
        
        total_score = 0
        for df in tables.values():
            if df.empty:
                continue
            
            # Score based on data completeness and variety
            completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            variety = len(df.columns) / 20  # Normalize by expected column count
            
            table_score = (completeness + min(variety, 1.0)) / 2
            total_score += table_score
        
        return total_score / len(tables) if tables else 0.0
    
    def _calculate_emotional_investment(self, relationship_analysis: Dict) -> str:
        """Calculate emotional investment level."""
        emotional_patterns = relationship_analysis.get('emotional_patterns', {})
        relationship_indicators = relationship_analysis.get('relationship_indicators', {})
        
        total_emotional = sum(emotional_patterns.values())
        total_relationship = sum(relationship_indicators.values())
        
        combined_score = total_emotional + total_relationship
        
        if combined_score > 50:
            return 'very_high'
        elif combined_score > 20:
            return 'high'
        elif combined_score > 10:
            return 'moderate'
        else:
            return 'low'
    
    def _summarize_key_findings(self) -> List[str]:
        """Summarize key findings from the analysis."""
        findings = []
        
        # Check for significant patterns
        risk_assessment = self.analysis_results.get('risk_assessment', {})
        if risk_assessment.get('risk_level') != 'low':
            findings.append(f"Elevated risk level detected: {risk_assessment.get('risk_level')}")
        
        # Check app identification
        metadata = self.analysis_results.get('metadata_analysis', {})
        app_detection = metadata.get('app_detection', {})
        if app_detection.get('detected_app', 'unknown') != 'unknown':
            findings.append(f"Identified app: {app_detection.get('detected_app')}")
        
        # Check for unusual patterns
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        for analysis in temporal_analysis.values():
            behavioral_changes = analysis.get('behavioral_changes', {})
            change_points = behavioral_changes.get('change_points', [])
            if change_points:
                findings.append(f"Detected {len(change_points)} significant behavioral changes")
        
        return findings
    
    def _assess_evidence_strength(self) -> Dict[str, str]:
        """Assess the strength of evidence for different findings."""
        data_quality = self._calculate_data_quality_score()
        completeness = self._calculate_analysis_completeness()
        
        overall_strength = (data_quality + completeness) / 2
        
        if overall_strength > 0.8:
            strength = 'strong'
        elif overall_strength > 0.6:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        return {
            'overall_strength': strength,
            'data_quality': f"{data_quality:.2f}",
            'analysis_completeness': f"{completeness:.2f}"
        }
    
    def _suggest_further_investigation(self) -> List[str]:
        """Suggest areas for further investigation."""
        suggestions = []
        
        # Check for incomplete data
        if self._calculate_data_quality_score() < 0.7:
            suggestions.append("Obtain additional data sources to improve analysis completeness")
        
        # Check for concerning patterns
        risk_factors = self.analysis_results.get('risk_assessment', {}).get('risk_factors', [])
        if 'emotional_dependency' in risk_factors:
            suggestions.append("Investigate other social media and communication platforms")
        
        if 'excessive_usage' in risk_factors:
            suggestions.append("Examine device usage patterns across all applications")
        
        return suggestions
    
    def _reconstruct_timeline(self) -> Dict[str, Any]:
        """Reconstruct timeline of significant events."""
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        
        timeline = {
            'first_activity': None,
            'last_activity': None,
            'significant_events': [],
            'activity_phases': []
        }
        
        for analysis in temporal_analysis.values():
            behavioral_changes = analysis.get('behavioral_changes', {})
            change_points = behavioral_changes.get('change_points', [])
            
            for change in change_points:
                timeline['significant_events'].append({
                    'date': change.get('date'),
                    'type': change.get('change_type'),
                    'description': f"Activity level {change.get('change_type')} by {change.get('magnitude', 0):.1%}"
                })
        
        return timeline
    
    def _create_behavioral_summary(self) -> str:
        """Create a concise behavioral summary."""
        profile = self.forensic_profile
        
        if not profile:
            return "Behavioral summary unavailable - profile not generated"
        
        risk_level = profile.get('risk_factors', {}).get('overall_risk_level', 'unknown')
        app_type = profile.get('profile_metadata', {}).get('app_identified', 'unknown')
        engagement = profile.get('behavioral_patterns', {}).get('engagement_level', 'unknown')
        
        summary = f"User exhibits {engagement} engagement with {app_type} application. "
        summary += f"Risk assessment indicates {risk_level} risk level. "
        
        psychological = profile.get('psychological_indicators', {})
        emotional_state = psychological.get('emotional_state', 'unknown')
        summary += f"Emotional analysis suggests {emotional_state} overall state."
        
        return summary
    
    def save_results(self, output_path: str = None) -> str:
        """Save analysis results and profile to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"forensic_analysis_{timestamp}.json"
        
        results = {
            'analysis_results': self.analysis_results,
            'forensic_profile': self.forensic_profile,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }        # Clean data to remove numpy types from keys and values
        def clean_for_json(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {str(k): clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        # Clean the results
        cleaned_results = clean_for_json(results)
        
        # Custom JSON encoder to handle remaining numpy types and other non-serializable objects
        def json_serialize(obj):
            import numpy as np
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif hasattr(obj, '__dict__'):  # custom objects
                return str(obj)
            else:
                return str(obj)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, default=json_serialize)
        
        print(f"Results saved to: {output_path}")
        
        # Generate summary report
        summary_path = self._generate_summary_report(output_path)
        
        return output_path
    
    def _generate_summary_report(self, json_path: str) -> str:
        """Generate a human-readable summary report from the forensic profile."""
        if not self.forensic_profile:
            return ""
        
        # Extract key information
        subject_id = self.forensic_profile.get('subject_identification', {})
        personal_info = subject_id.get('personal_information', {})
        profile_details = subject_id.get('profile_details', {})
        
        key_evidence = self.forensic_profile.get('key_evidence', {})
        behavioral_assessment = self.forensic_profile.get('behavioral_assessment', {})
        risk_evaluation = self.forensic_profile.get('risk_evaluation', {})
        investigation_summary = self.forensic_profile.get('investigation_summary', {})
        
        # Get statistics
        case_info = investigation_summary.get('case_information', {})
        key_stats = investigation_summary.get('key_statistics', {})
        
        # Extract behavioral data
        comm_analysis = behavioral_assessment.get('communication_analysis', {})
        usage_patterns = behavioral_assessment.get('usage_patterns', {})
        engagement = behavioral_assessment.get('engagement_level', {})
        
        # Extract significant conversations
        sig_conversations = key_evidence.get('significant_conversations', [])
        timeline_events = key_evidence.get('timeline_events', [])
        
        # Create summary content
        summary_content = f"""===============================================
    AI GIRLFRIEND FORENSIC ANALYSIS SUMMARY
    CASE FILE: {case_info.get('database_source', 'Unknown')}
    ANALYSIS DATE: {case_info.get('analysis_date', 'Unknown')}
===============================================

SUBJECT IDENTIFICATION:
   Name: {personal_info.get('name', 'Not Available')}
   Email: {personal_info.get('username', 'Not Available')}
   Age Range: {personal_info.get('age', 'Not Available')}
   Last Activity: {profile_details.get('last_activity', 'Unknown')}
   Identification Confidence: {subject_id.get('identification_confidence', 'Unknown')}

COMMUNICATION EVIDENCE:
   Total Messages Analyzed: {key_stats.get('total_messages_analyzed', 0)}
   Estimated Total Words: {key_stats.get('estimated_total_words', 0)}
   Data Sources: {key_stats.get('data_sources_found', 0)}
   Analysis Depth: {key_stats.get('analysis_depth', 'Unknown')}

BEHAVIORAL PATTERNS:
   Peak Activity Time: {usage_patterns.get('frequency', {}).get('peak_hour', 'Unknown')}:00
   Daily Average: {usage_patterns.get('frequency', {}).get('daily_average', 0)} interactions
   Engagement Level: {engagement.get('overall_engagement', 'Unknown')}
   Dependency Level: {engagement.get('dependency_indicators', 'Unknown')}
   Emotional Investment: {engagement.get('emotional_investment', 'Unknown')}

EMOTIONAL ANALYSIS:
   Dominant Emotion: {comm_analysis.get('emotional_expression', {}).get('dominant_emotion', 'Unknown')}
   Emotional Volatility: {comm_analysis.get('emotional_expression', {}).get('emotional_volatility', 0):.3f}
   Expression Intensity: {comm_analysis.get('emotional_expression', {}).get('expression_intensity', 0):.3f}
   Language Complexity: {comm_analysis.get('language_complexity', 0):.3f}

TIMELINE ANALYSIS:"""

        # Add timeline events
        if timeline_events:
            for event in timeline_events[:5]:
                summary_content += f"\n   • {event.get('date', 'Unknown')}: {event.get('description', 'Unknown event')}"
        else:
            summary_content += "\n   • No significant timeline events detected"
        
        summary_content += f"""

RISK ASSESSMENT: {risk_evaluation.get('overall_risk_level', 'Unknown').upper()}
   • Identified Risks: {len(risk_evaluation.get('identified_risks', []))}
   • Concerning Patterns: {len(risk_evaluation.get('concerning_patterns', []))}"""

        # Add mitigation recommendations
        recommendations = risk_evaluation.get('mitigation_recommendations', [])
        if recommendations:
            summary_content += "\n   • Recommendations:"
            for rec in recommendations[:3]:
                summary_content += f"\n     - {rec}"
        
        summary_content += f"""

TECHNICAL DETAILS:
   • EVIDENCE QUALITY: {investigation_summary.get('investigation_scope', {}).get('evidence_quality', 'Unknown')}
   • ANALYSIS COMPLETENESS: {investigation_summary.get('investigation_scope', {}).get('analysis_completeness', 0)*100:.0f}%
   • DATA TIMESPAN: {case_info.get('data_timespan', 'Unknown')}
   • TOTAL DATA POINTS: {case_info.get('total_data_points', 0)}

IMPORTANT INFORMATION DISCOVERED:"""

        # Extract personal information from messages and memory facts
        extracted_info = self._extract_personal_information()
        if extracted_info:
            for category, details in extracted_info.items():
                if details:
                    category_name = category.replace('_', ' ').title()
                    summary_content += f"\n   • {category_name}:"
                    for detail in details[:8]:  # Show up to 8 items per category
                        summary_content += f"\n     - {detail}"
        else:
            summary_content += "\n   • No significant personal information extracted from conversations"

        summary_content += "\n\nKEY MESSAGES:"

        # Extract significant/concerning messages
        key_messages = self._extract_key_messages()
        if key_messages:
            for i, message in enumerate(key_messages[:20], 1):  # Top 20 messages
                msg_type = message.get('type', 'Unknown').replace('_', ' ')
                content = message.get('content', 'No content')
                # Truncate very long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                
                summary_content += f"\n   {i}. [{msg_type}] {content}"
                if message.get('sentiment_score') is not None:
                    summary_content += f" (Sentiment: {message.get('sentiment_score'):.3f})"
        else:
            summary_content += "\n   • No significant messages detected"

        summary_content += f"""

INVESTIGATION SUMMARY:"""

        # Add key findings
        tech_analysis = self.forensic_profile.get('technical_analysis', {})
        investigative_recs = self.forensic_profile.get('investigative_recommendations', {})
        key_findings = investigative_recs.get('key_findings', [])
        
        if key_findings:
            for finding in key_findings:
                summary_content += f"\n   • {finding}"
        else:
            summary_content += "\n   • Standard AI companion usage patterns detected"
        
        summary_content += f"""

EVIDENCE STRENGTH:
   • Overall Strength: {investigative_recs.get('evidence_strength', {}).get('overall_strength', 'Unknown')}
   • Data Quality: {investigative_recs.get('evidence_strength', {}).get('data_quality', 'Unknown')}
   • Analysis Completeness: {investigative_recs.get('evidence_strength', {}).get('analysis_completeness', 'Unknown')}

===============================================
REPORT GENERATED BY: AI Girlfriend Forensics Analyzer v1.0
ANALYSIS ENGINE: Advanced behavioral pattern recognition
==============================================="""

        # Save summary report
        base_path = json_path.replace('.json', '').replace('\\forensic_analysis_REPLIKA_DB', '')
        summary_path = f"{base_path}_FORENSIC_SUMMARY.txt"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            print(f"Summary report saved to: {summary_path}")
            return summary_path
        except Exception as e:
            print(f"Failed to save summary report: {e}")
            return ""
    
    def _find_user_information(self) -> Dict[str, Any]:
        """Find and extract user personal information from various tables."""
        user_info = {}
        tables = self.database_data.get('tables', {})
        
        # Look for user profile tables
        profile_tables = ['user_profile', 'profile', 'user', 'account', 'users']
        
        for table_name, df in tables.items():
            if any(keyword in table_name.lower() for keyword in profile_tables):
                if not df.empty:
                    # Extract name information
                    name_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['name', 'first', 'last'])]
                    for col in name_cols:
                        if df[col].notna().any():
                            user_info['name'] = str(df[col].dropna().iloc[0])
                            break
                    
                    # Extract username
                    username_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['username', 'user_name', 'email'])]
                    for col in username_cols:
                        if df[col].notna().any():
                            user_info['username'] = str(df[col].dropna().iloc[0])
                            break
                    
                    # Extract age
                    age_cols = [col for col in df.columns if 'age' in col.lower()]
                    for col in age_cols:
                        if df[col].notna().any():
                            user_info['age'] = str(df[col].dropna().iloc[0])
                            break
                    
                    # Extract location
                    location_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['location', 'city', 'country', 'address'])]
                    for col in location_cols:
                        if df[col].notna().any():
                            user_info['location'] = str(df[col].dropna().iloc[0])
                            break
        
        return user_info
    
    def _extract_profile_data(self) -> Dict[str, Any]:
        """Extract profile-related data."""
        profile_data = {}
        tables = self.database_data.get('tables', {})
        
        # Look for profile or settings tables
        for table_name, df in tables.items():
            if 'profile' in table_name.lower() or 'setting' in table_name.lower():
                if not df.empty:
                    # Extract interests/preferences
                    interest_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['interest', 'hobby', 'preference', 'like'])]
                    interests = []
                    for col in interest_cols:
                        if df[col].notna().any():
                            interests.extend([str(val) for val in df[col].dropna().tolist()])
                    if interests:
                        profile_data['interests'] = list(set(interests))
                    
                    # Extract timestamps
                    date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['created', 'date', 'time'])]
                    for col in date_cols:
                        if df[col].notna().any():
                            if 'created' in col.lower():
                                profile_data['created_at'] = str(df[col].dropna().iloc[0])
                            else:
                                profile_data['last_active'] = str(df[col].dropna().iloc[0])
        
        return profile_data
    
    def _extract_significant_messages(self) -> List[Dict[str, Any]]:
        """Extract the most significant conversation messages."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        significant_messages = []
        
        for table_name, table_data in content_analysis.items():
            if 'chat' in table_name.lower() or 'message' in table_name.lower():
                # Look for tables with actual message content
                for column_name, column_data in table_data.items():
                    if isinstance(column_data, dict) and 'conversation_patterns' in column_data:
                        patterns = column_data.get('conversation_patterns', {})
                        
                        # Extract example messages
                        starters = patterns.get('common_starters', [])[:3]
                        enders = patterns.get('common_enders', [])[:3]
                        
                        if starters:
                            significant_messages.append({
                                'type': 'Common conversation starters',
                                'messages': starters,
                                'table_source': table_name
                            })
                        
                        if enders:
                            significant_messages.append({
                                'type': 'Common conversation enders',
                                'messages': enders,
                                'table_source': table_name
                            })
                        
                        # Extract high sentiment messages if available
                        sentiment_data = column_data.get('sentiment_analysis', {})
                        if sentiment_data:
                            most_positive = sentiment_data.get('extreme_sentiments', {}).get('most_positive', 0)
                            most_negative = sentiment_data.get('extreme_sentiments', {}).get('most_negative', 0)
                            
                            if most_positive > 0.7:
                                significant_messages.append({
                                    'type': 'Highly positive expression',
                                    'sentiment_score': most_positive,
                                    'table_source': table_name
                                })
                            
                            if most_negative < -0.5:
                                significant_messages.append({
                                    'type': 'Highly negative expression',
                                    'sentiment_score': most_negative,
                                    'table_source': table_name
                                })
        
        return significant_messages[:10]  # Limit to most important
    
    def _extract_behavioral_evidence(self) -> Dict[str, Any]:
        """Extract behavioral evidence patterns."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        behavioral_evidence = {
            'communication_frequency': [],
            'emotional_patterns': [],
            'language_indicators': []
        }
        
        for table_name, table_data in content_analysis.items():
            for column_name, column_data in table_data.items():
                if isinstance(column_data, dict):
                    # Communication patterns
                    if 'conversation_patterns' in column_data:
                        patterns = column_data['conversation_patterns']
                        message_stats = patterns.get('message_statistics', {})
                        lang_patterns = patterns.get('language_patterns', {})
                        
                        if message_stats.get('total_messages', 0) > 0:
                            behavioral_evidence['communication_frequency'].append({
                                'source': f"{table_name}.{column_name}",
                                'total_messages': message_stats.get('total_messages', 0),
                                'avg_length': message_stats.get('avg_message_length', 0),
                                'avg_words': message_stats.get('avg_word_count', 0)
                            })
                        
                        if lang_patterns:
                            behavioral_evidence['language_indicators'].append({
                                'source': f"{table_name}.{column_name}",
                                'questions_frequency': lang_patterns.get('question_frequency', 0),
                                'exclamations_frequency': lang_patterns.get('exclamation_frequency', 0),
                                'capital_usage': lang_patterns.get('capital_letters', 0)
                            })
                    
                    # Sentiment patterns
                    if 'sentiment_analysis' in column_data:
                        sentiment = column_data['sentiment_analysis']
                        overall = sentiment.get('overall_sentiment', {})
                        
                        behavioral_evidence['emotional_patterns'].append({
                            'source': f"{table_name}.{column_name}",
                            'positive_ratio': overall.get('positive', 0),
                            'negative_ratio': overall.get('negative', 0),
                            'neutral_ratio': overall.get('neutral', 0),
                            'compound_score': overall.get('compound', 0)
                        })
        
        return behavioral_evidence
    
    def _get_content_statistics(self) -> Dict[str, Any]:
        """Get overall content statistics."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        total_messages = 0
        total_words = 0
        sources = 0
        
        for table_data in content_analysis.values():
            for column_data in table_data.values():
                if isinstance(column_data, dict) and 'conversation_patterns' in column_data:
                    stats = column_data['conversation_patterns'].get('message_statistics', {})
                    total_messages += stats.get('total_messages', 0)
                    total_words += stats.get('total_messages', 0) * stats.get('avg_word_count', 0)
                    sources += 1
        
        return {
            'total_messages_analyzed': total_messages,
            'estimated_total_words': int(total_words),
            'data_sources_found': sources,
            'analysis_depth': 'Comprehensive' if sources > 3 else 'Moderate' if sources > 1 else 'Limited'
        }
    
    def _calculate_data_timespan(self) -> str:
        """Calculate the timespan of available data."""
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        
        earliest_date = None
        latest_date = None
        
        for analysis_data in temporal_analysis.values():
            if isinstance(analysis_data, dict) and 'activity_patterns' in analysis_data:
                duration_stats = analysis_data.get('duration_stats', {})
                first_msg = duration_stats.get('first_message')
                last_msg = duration_stats.get('last_message')
                
                if first_msg and (not earliest_date or first_msg < earliest_date):
                    earliest_date = first_msg
                if last_msg and (not latest_date or last_msg > latest_date):
                    latest_date = last_msg
        
        if earliest_date and latest_date:
            return f"{earliest_date} to {latest_date}"
        return "Unknown timespan"
    
    def _assess_evidence_quality(self) -> str:
        """Assess the quality of available evidence."""
        metadata = self.analysis_results.get('metadata_analysis', {})
        content_analysis = self.analysis_results.get('content_analysis', {})
        
        total_records = metadata.get('database_info', {}).get('total_records', 0)
        has_content = len(content_analysis) > 0
        
        if total_records > 500 and has_content:
            return "High"
        elif total_records > 100 and has_content:
            return "Moderate"
        elif has_content:
            return "Limited"
        else:
            return "Poor"
    
    def _calculate_analysis_completeness(self) -> float:
        """Calculate how complete the analysis is."""
        expected_sections = ['metadata_analysis', 'content_analysis', 'temporal_analysis']
        completed_sections = sum(1 for section in expected_sections if section in self.analysis_results)
        return completed_sections / len(expected_sections)
    
    def _calculate_identification_confidence(self, user_info: Dict, profile_data: Dict) -> str:
        """Calculate confidence level in subject identification."""
        info_count = sum(1 for v in user_info.values() if v != 'Not Available')
        profile_count = sum(1 for v in profile_data.values() if v not in ['Unknown', []])
        
        total_info = info_count + profile_count
        
        if total_info >= 4:
            return "High"
        elif total_info >= 2:
            return "Moderate"
        else:
            return "Low"
    
    def _extract_timeline_events(self) -> List[Dict[str, Any]]:
        """Extract significant timeline events."""
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        events = []
        
        for analysis_data in temporal_analysis.values():
            if isinstance(analysis_data, dict):
                # Look for behavioral changes
                behavioral_changes = analysis_data.get('behavioral_changes', {})
                if isinstance(behavioral_changes, dict):
                    change_points = behavioral_changes.get('change_points', [])
                    for change in change_points:
                        events.append({
                            'date': change.get('date'),
                            'type': 'behavioral_change',
                            'description': f"Activity {change.get('change_type', 'changed')} by {change.get('magnitude', 0)*100:.1f}%"
                        })
        
        return sorted(events, key=lambda x: x.get('date', ''))[:5]
    
    def _analyze_attachment_evidence(self) -> Dict[str, Any]:
        """Analyze evidence of emotional attachment."""
        relationship_data = self._extract_relationship_dynamics()
        
        return {
            'attachment_style': relationship_data.get('attachment_style', 'unknown'),
            'intimacy_level': relationship_data.get('intimacy_level', 0),
            'dependency_indicators': relationship_data.get('dependency_level', 0),
            'emotional_investment': relationship_data.get('emotional_investment', 'unknown')
        }
    
    def _analyze_emotional_expression(self) -> Dict[str, Any]:
        """Analyze emotional expression patterns."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        
        total_positive = 0
        total_negative = 0
        total_neutral = 0
        message_count = 0
        
        for table_data in content_analysis.values():
            for column_data in table_data.values():
                if isinstance(column_data, dict) and 'sentiment_analysis' in column_data:
                    sentiment = column_data['sentiment_analysis']
                    overall = sentiment.get('overall_sentiment', {})
                    
                    total_positive += overall.get('positive', 0)
                    total_negative += overall.get('negative', 0)
                    total_neutral += overall.get('neutral', 0)
                    message_count += 1
        
        if message_count > 0:
            return {
                'dominant_emotion': 'positive' if total_positive > total_negative else 'negative' if total_negative > total_neutral else 'neutral',
                'emotional_volatility': abs(total_positive - total_negative) / message_count,
                'expression_intensity': (total_positive + total_negative) / message_count
            }
        
        return {'dominant_emotion': 'unknown', 'emotional_volatility': 0, 'expression_intensity': 0}
    
    def _calculate_language_complexity(self) -> float:
        """Calculate language complexity score."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        
        total_complexity = 0
        count = 0
        
        for table_data in content_analysis.values():
            for column_data in table_data.values():
                if isinstance(column_data, dict) and 'conversation_patterns' in column_data:
                    patterns = column_data['conversation_patterns']
                    message_stats = patterns.get('message_statistics', {})
                    
                    avg_length = message_stats.get('avg_message_length', 0)
                    avg_words = message_stats.get('avg_word_count', 0)
                    
                    if avg_length > 0 and avg_words > 0:
                        complexity = (avg_length / 100) * (avg_words / 10)  # Normalized complexity score
                        total_complexity += min(complexity, 1.0)  # Cap at 1.0
                        count += 1
        
        return total_complexity / count if count > 0 else 0
    
    def _assess_dependency_level(self) -> str:
        """Assess dependency level based on usage patterns."""
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        
        high_frequency_count = 0
        total_analyses = 0
        
        for analysis_data in temporal_analysis.values():
            if isinstance(analysis_data, dict) and 'usage_intensity' in analysis_data:
                intensity = analysis_data['usage_intensity']
                daily_stats = intensity.get('daily_stats', {})
                avg_daily = daily_stats.get('average_daily_messages', 0)
                
                if avg_daily > 20:  # High usage threshold
                    high_frequency_count += 1
                total_analyses += 1
        
        if total_analyses == 0:
            return 'unknown'
        
        dependency_ratio = high_frequency_count / total_analyses
        
        if dependency_ratio > 0.7:
            return 'high'
        elif dependency_ratio > 0.3:
            return 'moderate'
        else:
            return 'low'
    
    def _assess_emotional_investment(self) -> str:
        """Assess emotional investment level."""
        content_analysis = self.analysis_results.get('content_analysis', {})
        
        high_sentiment_count = 0
        total_sentiment_analyses = 0
        
        for table_data in content_analysis.values():
            for column_data in table_data.values():
                if isinstance(column_data, dict) and 'sentiment_analysis' in column_data:
                    sentiment = column_data['sentiment_analysis']
                    overall = sentiment.get('overall_sentiment', {})
                    compound = abs(overall.get('compound', 0))
                    
                    if compound > 0.5:  # High emotional intensity
                        high_sentiment_count += 1
                    total_sentiment_analyses += 1
        
        if total_sentiment_analyses == 0:
            return 'unknown'
        
        investment_ratio = high_sentiment_count / total_sentiment_analyses
        
        if investment_ratio > 0.6:
            return 'high'
        elif investment_ratio > 0.3:
            return 'moderate'
        else:
            return 'low'
    
    def _assess_data_integrity(self) -> Dict[str, Any]:
        """Assess data integrity and completeness."""
        metadata = self.analysis_results.get('metadata_analysis', {})
        
        return {
            'data_completeness': self._calculate_analysis_completeness(),
            'evidence_quality': self._assess_evidence_quality(),
            'temporal_coverage': self._calculate_data_timespan(),
            'source_reliability': 'High' if metadata.get('database_info', {}).get('total_tables', 0) > 10 else 'Moderate'
        }

    def _extract_personal_information(self) -> Dict[str, List[str]]:
        """Extract comprehensive personal information from chat messages and memory facts."""
        personal_info = {
            'location_data': [],
            'personal_details': [],
            'contact_information': [],
            'lifestyle_information': [],
            'relationship_details': [],
            'work_education': [],
            'financial_information': [],
            'family_information': [],
            'health_information': [],
            'social_media': [],
            'schedule_routine': [],
            'personal_struggles': []
        }
        
        # Check raw database data for personal information
        tables = self.database_data.get('tables', {})
        
        # Extract from memory facts (AI's learned information about user)
        if 'memory_fact_v3' in tables:
            df = tables['memory_fact_v3']
            if 'text' in df.columns:
                for text in df['text'].dropna():
                    text_str = str(text)
                    text_lower = text_str.lower()
                    
                    # Location indicators - more specific
                    location_phrases = ['i live in', 'i am from', 'my address', 'my location', 'coordinates', 'gps', 'street', 'avenue', 'city', 'zip code', 'postal code']
                    if any(phrase in text_lower for phrase in location_phrases):
                        personal_info['location_data'].append(f"Location: {text_str}")
                    
                    # Personal details - age, physical attributes
                    personal_phrases = ['years old', 'my age', 'born in', 'birthday', 'height', 'weight', 'appearance', 'looks like', 'ethnicity']
                    if any(phrase in text_lower for phrase in personal_phrases):
                        personal_info['personal_details'].append(f"Personal: {text_str}")
                    
                    # Contact information - actual contact methods
                    contact_phrases = ['phone number', 'my email', 'contact me at', '@gmail', '@yahoo', '@hotmail', 'call me', 'text me']
                    if any(phrase in text_lower for phrase in contact_phrases):
                        personal_info['contact_information'].append(f"Contact: {text_str}")
                    
                    # Work/Education - career and school info
                    work_phrases = ['work at', 'job is', 'employed at', 'study at', 'university', 'college', 'school', 'degree in', 'career', 'profession']
                    if any(phrase in text_lower for phrase in work_phrases):
                        personal_info['work_education'].append(f"Work/Education: {text_str}")
                    
                    # Lifestyle - interests and hobbies
                    lifestyle_phrases = ['hobby is', 'i like', 'i enjoy', 'favorite', 'interests', 'passionate about', 'loves', 'enjoys']
                    if any(phrase in text_lower for phrase in lifestyle_phrases):
                        personal_info['lifestyle_information'].append(f"Interest: {text_str}")
                    
                    # Relationship information
                    relationship_phrases = ['relationship', 'dating', 'married', 'single', 'boyfriend', 'girlfriend', 'partner', 'spouse', 'ex-', 'crush']
                    if any(phrase in text_lower for phrase in relationship_phrases):
                        personal_info['relationship_details'].append(f"Relationship: {text_str}")
                    
                    # Financial information
                    financial_phrases = ['salary', 'income', 'money', 'debt', 'credit card', 'bank', 'financial', 'broke', 'rich', 'poor']
                    if any(phrase in text_lower for phrase in financial_phrases):
                        personal_info['financial_information'].append(f"Financial: {text_str}")
                    
                    # Family information
                    family_phrases = ['family', 'parents', 'mother', 'father', 'siblings', 'brother', 'sister', 'children', 'kids', 'relatives']
                    if any(phrase in text_lower for phrase in family_phrases):
                        personal_info['family_information'].append(f"Family: {text_str}")
                    
                    # Health information
                    health_phrases = ['health', 'medical', 'doctor', 'medication', 'therapy', 'mental health', 'depression', 'anxiety', 'illness']
                    if any(phrase in text_lower for phrase in health_phrases):
                        personal_info['health_information'].append(f"Health: {text_str}")
                    
                    # Social media references
                    social_phrases = ['instagram', 'facebook', 'twitter', 'tiktok', 'snapchat', 'social media', 'followers', 'posts']
                    if any(phrase in text_lower for phrase in social_phrases):
                        personal_info['social_media'].append(f"Social Media: {text_str}")
                    
                    # Schedule and routine
                    schedule_phrases = ['schedule', 'routine', 'daily', 'morning', 'evening', 'weekend', 'work hours', 'free time']
                    if any(phrase in text_lower for phrase in schedule_phrases):
                        personal_info['schedule_routine'].append(f"Routine: {text_str}")
                    
                    # Personal struggles
                    struggle_phrases = ['struggling', 'problem', 'issue', 'difficult', 'hard time', 'stress', 'worry', 'concern']
                    if any(phrase in text_lower for phrase in struggle_phrases):
                        personal_info['personal_struggles'].append(f"Struggle: {text_str}")
        
        # Extract from chat messages (user's direct communications)
        if 'chat_message' in tables:
            df = tables['chat_message']
            if 'text' in df.columns:
                for text in df['text'].dropna():
                    text_str = str(text)
                    text_lower = text_str.lower()
                    
                    # Look for direct personal disclosures
                    disclosure_phrases = ['my name is', 'i live', 'i work at', 'my job', 'my address', 'i am', 'i was born']
                    if any(phrase in text_lower for phrase in disclosure_phrases):
                        personal_info['personal_details'].append(f"Self-disclosure: {text_str}")
                    
                    # Location sharing
                    location_sharing = ['here is my location', 'i am at', 'my coordinates', 'current location', 'address is']
                    if any(phrase in text_lower for phrase in location_sharing):
                        personal_info['location_data'].append(f"Location sharing: {text_str}")
                    
                    # Emotional states and mental health
                    emotional_phrases = ['i feel', 'i am depressed', 'i am sad', 'i am happy', 'emotional', 'mental state']
                    if any(phrase in text_lower for phrase in emotional_phrases):
                        personal_info['health_information'].append(f"Emotional state: {text_str}")
        
        # Extract from user profile table
        if 'user_profile' in tables:
            df = tables['user_profile']
            for col in df.columns:
                for value in df[col].dropna():
                    value_str = str(value)
                    if len(value_str) > 1 and value_str not in ['1', '0', 'true', 'false']:
                        personal_info['personal_details'].append(f"Profile data ({col}): {value_str}")
        
        # Filter out empty categories and remove duplicates
        for category in personal_info:
            personal_info[category] = list(set(personal_info[category]))  # Remove duplicates
            
        return {k: v for k, v in personal_info.items() if v}

    def _extract_key_messages(self) -> List[Dict[str, Any]]:
        """Extract comprehensive significant or concerning messages."""
        key_messages = []
        
        # Get content analysis for sentiment scores
        content_analysis = self.analysis_results.get('content_analysis', {})
        
        # Look for messages in chat_message table
        tables = self.database_data.get('tables', {})
        if 'chat_message' in tables:
            df = tables['chat_message']
            if 'text' in df.columns:
                for i, text in enumerate(df['text'].dropna()):
                    text_str = str(text)
                    text_lower = text_str.lower()
                    
                    # CRITICAL: Concerning mental health content
                    critical_keywords = ['suicide', 'kill myself', 'end my life', 'want to die', 'not worth living', 'end it all', 'hurt myself', 'self harm']
                    if any(keyword in text_lower for keyword in critical_keywords):
                        key_messages.append({
                            'type': 'CRITICAL_MENTAL_HEALTH',
                            'content': text_str,
                            'sentiment_score': -0.9,
                            'priority': 1
                        })
                    
                    # HIGH: Personal information sharing
                    personal_sharing = ['my address is', 'my phone number', 'my social security', 'my credit card', 'my password', 'my location', 'i live at']
                    if any(keyword in text_lower for keyword in personal_sharing):
                        key_messages.append({
                            'type': 'PERSONAL_DATA_SHARING',
                            'content': text_str,
                            'sentiment_score': 0.0,
                            'priority': 2
                        })
                    
                    # HIGH: Sexual/explicit content
                    sexual_keywords = ['sexual', 'nude', 'naked', 'sex', 'porn', 'explicit', 'intimate photos', 'sexting']
                    if any(keyword in text_lower for keyword in sexual_keywords):
                        key_messages.append({
                            'type': 'EXPLICIT_CONTENT',
                            'content': text_str,
                            'sentiment_score': 0.0,
                            'priority': 2
                        })
                    
                    # MEDIUM: Emotional dependency indicators
                    dependency_keywords = ['i love you', 'need you', 'can\'t live without', 'you\'re my only', 'obsessed', 'addicted to you']
                    if any(keyword in text_lower for keyword in dependency_keywords):
                        key_messages.append({
                            'type': 'EMOTIONAL_DEPENDENCY',
                            'content': text_str,
                            'sentiment_score': 0.6,
                            'priority': 3
                        })
                    
                    # MEDIUM: Financial information
                    financial_keywords = ['my salary', 'make money', 'financial problems', 'debt', 'broke', 'rich', 'bank account']
                    if any(keyword in text_lower for keyword in financial_keywords):
                        key_messages.append({
                            'type': 'FINANCIAL_DISCLOSURE',
                            'content': text_str,
                            'sentiment_score': 0.0,
                            'priority': 3
                        })
                    
                    # MEDIUM: Relationship confessions
                    relationship_keywords = ['relationship problems', 'cheating', 'affair', 'divorce', 'breaking up', 'marriage issues']
                    if any(keyword in text_lower for keyword in relationship_keywords):
                        key_messages.append({
                            'type': 'RELATIONSHIP_ISSUES',
                            'content': text_str,
                            'sentiment_score': -0.3,
                            'priority': 3
                        })
                    
                    # MEDIUM: Health disclosures
                    health_keywords = ['health problems', 'medical condition', 'diagnosed with', 'medication', 'therapy', 'mental health']
                    if any(keyword in text_lower for keyword in health_keywords):
                        key_messages.append({
                            'type': 'HEALTH_DISCLOSURE',
                            'content': text_str,
                            'sentiment_score': -0.2,
                            'priority': 3
                        })
                    
                    # MEDIUM: Family/personal secrets
                    secret_keywords = ['family secret', 'don\'t tell anyone', 'secret', 'confidential', 'private', 'between us']
                    if any(keyword in text_lower for keyword in secret_keywords):
                        key_messages.append({
                            'type': 'CONFIDENTIAL_DISCLOSURE',
                            'content': text_str,
                            'sentiment_score': 0.0,
                            'priority': 3
                        })
                    
                    # LOW: Romantic/intimate expressions
                    romantic_keywords = ['i love you', 'romantic', 'kiss', 'date', 'relationship', 'feelings for you']
                    if any(keyword in text_lower for keyword in romantic_keywords):
                        key_messages.append({
                            'type': 'ROMANTIC_EXPRESSION',
                            'content': text_str,
                            'sentiment_score': 0.5,
                            'priority': 4
                        })
                    
                    # LOW: Work/career information
                    work_keywords = ['my job', 'work at', 'my boss', 'workplace', 'career', 'profession']
                    if any(keyword in text_lower for keyword in work_keywords):
                        key_messages.append({
                            'type': 'WORK_DISCLOSURE',
                            'content': text_str,
                            'sentiment_score': 0.0,
                            'priority': 4
                        })
                    
                    # LOW: Lengthy messages (potentially significant)
                    if len(text_str) > 300:
                        key_messages.append({
                            'type': 'LENGTHY_MESSAGE',
                            'content': text_str,
                            'sentiment_score': 0.0,
                            'priority': 5
                        })
                    
                    # VERY LOW: Questions about AI capabilities
                    ai_questions = ['are you real', 'are you human', 'what are you', 'ai', 'robot', 'artificial']
                    if any(keyword in text_lower for keyword in ai_questions):
                        key_messages.append({
                            'type': 'AI_AWARENESS',
                            'content': text_str,
                            'sentiment_score': 0.0,
                            'priority': 6
                        })
        
        # Extract from memory facts (AI's interpretation of user)
        if 'memory_fact_v3' in tables:
            df = tables['memory_fact_v3']
            if 'text' in df.columns:
                for text in df['text'].dropna():
                    text_str = str(text)
                    key_messages.append({
                        'type': 'AI_MEMORY',
                        'content': f"AI learned: {text_str}",
                        'sentiment_score': 0.0,
                        'priority': 4
                    })
        
        # Now analyze actual messages for sentiment extremes using existing analyzer
        if 'chat_message' in tables:
            df = tables['chat_message']
            if 'text' in df.columns:
                try:
                    from nltk.sentiment import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    
                    sentiment_messages = []
                    
                    for text in df['text'].dropna():
                        text_str = str(text)
                        if len(text_str) > 20:  # Only analyze substantial messages
                            sentiment_scores = analyzer.polarity_scores(text_str)
                            compound = sentiment_scores['compound']
                            
                            # Store all messages with their sentiment scores
                            sentiment_messages.append({
                                'text': text_str,
                                'compound': compound,
                                'positive': sentiment_scores['pos'],
                                'negative': sentiment_scores['neg']
                            })
                    
                    # Sort by sentiment extremity and take the most extreme ones
                    sentiment_messages.sort(key=lambda x: abs(x['compound']), reverse=True)
                    
                    # Add top 5 most positive messages
                    positive_messages = [msg for msg in sentiment_messages if msg['compound'] > 0.5][:5]
                    for msg in positive_messages:
                        key_messages.append({
                            'type': 'HIGHLY_POSITIVE',
                            'content': msg['text'],
                            'sentiment_score': msg['compound'],
                            'priority': 5
                        })
                    
                    # Add top 5 most negative messages  
                    negative_messages = [msg for msg in sentiment_messages if msg['compound'] < -0.3][:5]
                    for msg in negative_messages:
                        key_messages.append({
                            'type': 'HIGHLY_NEGATIVE',
                            'content': msg['text'],
                            'sentiment_score': msg['compound'],
                            'priority': 3
                        })
                        
                except (ImportError, LookupError) as e:
                    # Fallback if NLTK sentiment analyzer is not available
                    print(f"NLTK sentiment analyzer error: {e}")
                    pass
        
        # Sort by priority (1=highest, 6=lowest) then by sentiment extremity
        def sort_key(msg):
            priority = msg.get('priority', 6)
            sentiment_extremity = abs(msg.get('sentiment_score', 0))
            return (priority, -sentiment_extremity)
        
        # Remove duplicates and sort
        unique_messages = []
        seen_content = set()
        for msg in sorted(key_messages, key=sort_key):
            content_key = msg['content'][:100]  # Use first 100 chars to detect duplicates
            if content_key not in seen_content:
                unique_messages.append(msg)
                seen_content.add(content_key)
        
        return unique_messages[:25]  # Return top 25 most significant messages
