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
            Detailed forensic profile
        """
        if not self.analysis_results:
            print("No analysis results available. Running analysis...")
            self.analyze_database()
        
        print("Generating forensic profile...")
        
        profile = {
            'profile_metadata': self._generate_profile_metadata(),
            'user_characteristics': self._extract_user_characteristics(),
            'behavioral_patterns': self._extract_behavioral_patterns(),
            'psychological_indicators': self._extract_psychological_indicators(),
            'relationship_dynamics': self._extract_relationship_dynamics(),
            'temporal_behavior': self._extract_temporal_behavior(),
            'risk_factors': self._extract_risk_factors(),
            'investigation_insights': self._generate_investigation_insights()
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
        return output_path
