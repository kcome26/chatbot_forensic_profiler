"""
Risk assessment module for identifying concerning behavioral patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RiskAssessment:
    """Risk assessment system for identifying concerning behavioral patterns."""
    
    def __init__(self):
        self.risk_thresholds = self._initialize_risk_thresholds()
        self.risk_weights = self._initialize_risk_weights()
        
    def _initialize_risk_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk assessment thresholds."""
        return {
            'usage_frequency': {
                'low': 10,      # messages per day
                'moderate': 50,
                'high': 100,
                'extreme': 200
            },
            'session_duration': {
                'low': 15,      # minutes per session
                'moderate': 60,
                'high': 180,
                'extreme': 360
            },
            'nighttime_usage': {
                'low': 5,       # percentage of activity
                'moderate': 15,
                'high': 30,
                'extreme': 50
            },
            'emotional_dependency': {
                'low': 5,       # dependency indicators count
                'moderate': 10,
                'high': 20,
                'extreme': 40
            },
            'social_isolation': {
                'low': 0.1,     # isolation score
                'moderate': 0.3,
                'high': 0.6,
                'extreme': 0.8
            }
        }
    
    def _initialize_risk_weights(self) -> Dict[str, float]:
        """Initialize weights for different risk categories."""
        return {
            'addiction_potential': 0.25,
            'emotional_dependency': 0.30,
            'social_isolation': 0.20,
            'sleep_disruption': 0.15,
            'psychological_impact': 0.10
        }
    
    def assess_comprehensive_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment across all behavioral dimensions.
        
        Args:
            analysis_results: Complete analysis results from forensic analyzer
            
        Returns:
            Comprehensive risk assessment
        """
        risk_assessment = {
            'overall_risk_score': 0.0,
            'risk_level': 'low',
            'risk_categories': {},
            'specific_risks': {},
            'risk_factors': [],
            'protective_factors': [],
            'recommendations': [],
            'intervention_urgency': 'none',
            'monitoring_priorities': []
        }
        
        # Assess individual risk categories
        risk_assessment['risk_categories'] = {
            'addiction_potential': self._assess_addiction_risk(analysis_results),
            'emotional_dependency': self._assess_emotional_dependency_risk(analysis_results),
            'social_isolation': self._assess_social_isolation_risk(analysis_results),
            'sleep_disruption': self._assess_sleep_disruption_risk(analysis_results),
            'psychological_impact': self._assess_psychological_impact_risk(analysis_results)
        }
        
        # Calculate weighted overall risk score
        overall_score = 0.0
        for category, weight in self.risk_weights.items():
            category_score = risk_assessment['risk_categories'].get(category, {}).get('risk_score', 0.0)
            overall_score += category_score * weight
        
        risk_assessment['overall_risk_score'] = overall_score
        risk_assessment['risk_level'] = self._categorize_risk_level(overall_score)
        
        # Identify specific risks and factors
        risk_assessment['specific_risks'] = self._identify_specific_risks(analysis_results, risk_assessment['risk_categories'])
        risk_assessment['risk_factors'] = self._compile_risk_factors(risk_assessment['risk_categories'])
        risk_assessment['protective_factors'] = self._identify_protective_factors(analysis_results)
        
        # Generate recommendations and priorities
        risk_assessment['recommendations'] = self._generate_recommendations(risk_assessment)
        risk_assessment['intervention_urgency'] = self._determine_intervention_urgency(risk_assessment)
        risk_assessment['monitoring_priorities'] = self._determine_monitoring_priorities(risk_assessment)
        
        return risk_assessment
    
    def _assess_addiction_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of technology/relationship addiction."""
        addiction_risk = {
            'risk_score': 0.0,
            'risk_level': 'low',
            'indicators': [],
            'severity_factors': {},
            'temporal_patterns': {}
        }
        
        temporal_analysis = analysis_results.get('temporal_analysis', {})
        
        # Analyze usage frequency
        total_daily_usage = 0
        usage_variance = 0
        session_data = []
        
        for analysis in temporal_analysis.values():
            usage_intensity = analysis.get('usage_intensity', {})
            daily_stats = usage_intensity.get('daily_stats', {})
            
            avg_daily = daily_stats.get('average_daily_messages', 0)
            total_daily_usage += avg_daily
            usage_variance += daily_stats.get('daily_variance', 0)
            
            # Session analysis
            sessions = analysis.get('session_analysis', {})
            session_overview = sessions.get('session_overview', {})
            avg_session_duration = session_overview.get('avg_session_duration_minutes', 0)
            session_data.append(avg_session_duration)
        
        # Calculate risk factors
        if total_daily_usage > self.risk_thresholds['usage_frequency']['high']:
            addiction_risk['indicators'].append('excessive_daily_usage')
            addiction_risk['severity_factors']['daily_usage'] = total_daily_usage
        
        if session_data and max(session_data) > self.risk_thresholds['session_duration']['high']:
            addiction_risk['indicators'].append('prolonged_sessions')
            addiction_risk['severity_factors']['max_session_duration'] = max(session_data)
        
        if usage_variance > 1000:  # High variance indicates irregular, potentially compulsive usage
            addiction_risk['indicators'].append('irregular_usage_patterns')
            addiction_risk['severity_factors']['usage_variance'] = usage_variance
        
        # Check for usage streaks
        for analysis in temporal_analysis.values():
            usage_intensity = analysis.get('usage_intensity', {})
            streaks = usage_intensity.get('usage_streaks', {})
            longest_streak = streaks.get('longest_streak', 0)
            
            if longest_streak > 30:  # More than 30 consecutive days
                addiction_risk['indicators'].append('extended_usage_streaks')
                addiction_risk['severity_factors']['longest_streak'] = longest_streak
        
        # Calculate risk score
        risk_score = 0.0
        if total_daily_usage > self.risk_thresholds['usage_frequency']['extreme']:
            risk_score += 0.4
        elif total_daily_usage > self.risk_thresholds['usage_frequency']['high']:
            risk_score += 0.2
        
        if session_data and max(session_data) > self.risk_thresholds['session_duration']['extreme']:
            risk_score += 0.3
        elif session_data and max(session_data) > self.risk_thresholds['session_duration']['high']:
            risk_score += 0.15
        
        if len(addiction_risk['indicators']) >= 3:
            risk_score += 0.3
        
        addiction_risk['risk_score'] = min(1.0, risk_score)
        addiction_risk['risk_level'] = self._categorize_risk_level(addiction_risk['risk_score'])
        
        return addiction_risk
    
    def _assess_emotional_dependency_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of unhealthy emotional dependency."""
        dependency_risk = {
            'risk_score': 0.0,
            'risk_level': 'low',
            'indicators': [],
            'emotional_patterns': {},
            'attachment_concerns': {}
        }
        
        relationship_analysis = analysis_results.get('relationship_analysis', {})
        content_analysis = analysis_results.get('content_analysis', {})
        
        # Analyze dependency indicators
        relationship_metrics = relationship_analysis.get('relationship_metrics', {})
        dependency_score = relationship_metrics.get('dependency_indicators', 0)
        intimacy_level = relationship_metrics.get('intimacy_level', 0)
        
        if dependency_score > self.risk_thresholds['emotional_dependency']['high']:
            dependency_risk['indicators'].append('high_dependency_expressions')
        
        if intimacy_level > 15:
            dependency_risk['indicators'].append('excessive_intimacy_seeking')
        
        # Analyze emotional patterns
        emotional_patterns = relationship_analysis.get('emotional_patterns', {})
        
        # Check for concerning emotional expressions
        concerning_emotions = {
            'loneliness': 5,
            'sadness': 10,
            'fear': 8,
            'anxiety': 6
        }
        
        for emotion, threshold in concerning_emotions.items():
            count = emotional_patterns.get(emotion, 0)
            if count > threshold:
                dependency_risk['indicators'].append(f'elevated_{emotion}_expressions')
                dependency_risk['emotional_patterns'][emotion] = count
        
        # Analyze attachment style
        attachment_analysis = relationship_analysis.get('attachment_analysis', {})
        attachment_style = attachment_analysis.get('predicted_style', 'unknown')
        
        if attachment_style == 'anxious':
            dependency_risk['indicators'].append('anxious_attachment_style')
            dependency_risk['attachment_concerns']['style'] = attachment_style
        
        # Check sentiment patterns for dependency signs
        for table_results in content_analysis.values():
            for column_results in table_results.values():
                sentiment = column_results.get('sentiment_analysis', {})
                sentiment_trends = sentiment.get('sentiment_trends', {})
                
                if sentiment_trends.get('volatility', 0) > 0.3:
                    dependency_risk['indicators'].append('emotional_volatility')
        
        # Calculate risk score
        risk_score = 0.0
        
        if dependency_score > self.risk_thresholds['emotional_dependency']['extreme']:
            risk_score += 0.4
        elif dependency_score > self.risk_thresholds['emotional_dependency']['high']:
            risk_score += 0.2
        
        if attachment_style == 'anxious':
            risk_score += 0.2
        
        if len(dependency_risk['indicators']) >= 3:
            risk_score += 0.3
        
        emotional_intensity = sum(dependency_risk['emotional_patterns'].values())
        if emotional_intensity > 20:
            risk_score += 0.1
        
        dependency_risk['risk_score'] = min(1.0, risk_score)
        dependency_risk['risk_level'] = self._categorize_risk_level(dependency_risk['risk_score'])
        
        return dependency_risk
    
    def _assess_social_isolation_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of social isolation and withdrawal."""
        isolation_risk = {
            'risk_score': 0.0,
            'risk_level': 'low',
            'indicators': [],
            'behavioral_patterns': {},
            'temporal_concerns': {}
        }
        
        temporal_analysis = analysis_results.get('temporal_analysis', {})
        
        # Analyze temporal patterns for isolation indicators
        for analysis in temporal_analysis.values():
            circadian_patterns = analysis.get('circadian_patterns', {})
            circadian_dist = circadian_patterns.get('circadian_distribution', {})
            
            # Check nighttime activity
            night_percentage = circadian_dist.get('night', {}).get('percentage', 0)
            if night_percentage > self.risk_thresholds['nighttime_usage']['high']:
                isolation_risk['indicators'].append('excessive_nighttime_activity')
                isolation_risk['temporal_concerns']['night_percentage'] = night_percentage
            
            # Check for irregular patterns
            activity_patterns = analysis.get('activity_patterns', {})
            regularity = activity_patterns.get('activity_regularity', {}).get('overall_regularity', 1.0)
            
            if regularity < 0.3:
                isolation_risk['indicators'].append('irregular_activity_patterns')
                isolation_risk['behavioral_patterns']['activity_regularity'] = regularity
            
            # Check for weekend vs weekday patterns
            weekly_patterns = analysis.get('weekly_patterns', {})
            weekend_pref = weekly_patterns.get('weekday_vs_weekend', {}).get('weekend_preference', False)
            
            if weekend_pref:
                weekend_pct = weekly_patterns.get('weekday_vs_weekend', {}).get('weekend_percentage', 0)
                if weekend_pct > 70:
                    isolation_risk['indicators'].append('weekend_isolation_pattern')
        
        # Analyze content for social isolation indicators
        content_analysis = analysis_results.get('content_analysis', {})
        
        # Look for isolation-related language
        isolation_keywords = ['lonely', 'alone', 'isolated', 'no friends', 'nobody']
        isolation_mentions = 0
        
        for table_results in content_analysis.values():
            for column_results in table_results.values():
                # This would require text analysis - simplified for now
                emotional_patterns = column_results.get('emotional_patterns', {})
                loneliness_score = emotional_patterns.get('emotional_expressions', {}).get('sadness', 0)
                isolation_mentions += loneliness_score
        
        if isolation_mentions > 10:
            isolation_risk['indicators'].append('frequent_loneliness_expressions')
            isolation_risk['behavioral_patterns']['loneliness_expressions'] = isolation_mentions
        
        # Calculate risk score
        risk_score = 0.0
        
        # Night activity scoring
        max_night_pct = 0
        for analysis in temporal_analysis.values():
            circadian_patterns = analysis.get('circadian_patterns', {})
            circadian_dist = circadian_patterns.get('circadian_distribution', {})
            night_pct = circadian_dist.get('night', {}).get('percentage', 0)
            max_night_pct = max(max_night_pct, night_pct)
        
        if max_night_pct > self.risk_thresholds['nighttime_usage']['extreme']:
            risk_score += 0.4
        elif max_night_pct > self.risk_thresholds['nighttime_usage']['high']:
            risk_score += 0.2
        
        if len(isolation_risk['indicators']) >= 2:
            risk_score += 0.3
        
        if isolation_mentions > 15:
            risk_score += 0.3
        
        isolation_risk['risk_score'] = min(1.0, risk_score)
        isolation_risk['risk_level'] = self._categorize_risk_level(isolation_risk['risk_score'])
        
        return isolation_risk
    
    def _assess_sleep_disruption_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of sleep pattern disruption."""
        sleep_risk = {
            'risk_score': 0.0,
            'risk_level': 'low',
            'indicators': [],
            'sleep_patterns': {},
            'disruption_severity': {}
        }
        
        temporal_analysis = analysis_results.get('temporal_analysis', {})
        
        for analysis in temporal_analysis.values():
            circadian_patterns = analysis.get('circadian_patterns', {})
            circadian_dist = circadian_patterns.get('circadian_distribution', {})
            
            # Check late night activity (11 PM - 3 AM)
            night_percentage = circadian_dist.get('night', {}).get('percentage', 0)
            
            if night_percentage > 20:
                sleep_risk['indicators'].append('late_night_activity')
                sleep_risk['sleep_patterns']['night_activity_percentage'] = night_percentage
            
            # Check chronotype
            chronotype = circadian_patterns.get('chronotype', 'unknown')
            if chronotype == 'night_owl':
                sleep_risk['indicators'].append('night_owl_pattern')
                sleep_risk['sleep_patterns']['chronotype'] = chronotype
            
            # Check for irregular sleep patterns
            regularity = circadian_patterns.get('circadian_regularity', 1.0)
            if regularity < 0.4:
                sleep_risk['indicators'].append('irregular_sleep_schedule')
                sleep_risk['sleep_patterns']['circadian_regularity'] = regularity
            
            # Check session patterns for late night sessions
            sessions = analysis.get('session_analysis', {})
            session_timing = sessions.get('session_timing', {})
            preferred_hours = session_timing.get('preferred_session_hours', {})
            
            late_night_sessions = sum(count for hour, count in preferred_hours.items() 
                                    if isinstance(hour, int) and (hour >= 23 or hour <= 3))
            
            if late_night_sessions > 10:
                sleep_risk['indicators'].append('frequent_late_night_sessions')
                sleep_risk['disruption_severity']['late_night_sessions'] = late_night_sessions
        
        # Calculate risk score
        risk_score = 0.0
        
        max_night_activity = max((analysis.get('circadian_patterns', {})
                                .get('circadian_distribution', {})
                                .get('night', {}).get('percentage', 0))
                               for analysis in temporal_analysis.values())
        
        if max_night_activity > 40:
            risk_score += 0.5
        elif max_night_activity > 25:
            risk_score += 0.3
        elif max_night_activity > 15:
            risk_score += 0.1
        
        if len(sleep_risk['indicators']) >= 2:
            risk_score += 0.3
        
        sleep_risk['risk_score'] = min(1.0, risk_score)
        sleep_risk['risk_level'] = self._categorize_risk_level(sleep_risk['risk_score'])
        
        return sleep_risk
    
    def _assess_psychological_impact_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of negative psychological impact."""
        psych_risk = {
            'risk_score': 0.0,
            'risk_level': 'low',
            'indicators': [],
            'mental_health_concerns': {},
            'behavioral_changes': {}
        }
        
        content_analysis = analysis_results.get('content_analysis', {})
        temporal_analysis = analysis_results.get('temporal_analysis', {})
        
        # Analyze sentiment patterns for psychological distress
        negative_sentiment_total = 0
        emotional_volatility = 0
        sentiment_count = 0
        
        for table_results in content_analysis.values():
            for column_results in table_results.values():
                sentiment = column_results.get('sentiment_analysis', {})
                overall_sentiment = sentiment.get('overall_sentiment', {})
                
                if overall_sentiment:
                    negative_sentiment_total += overall_sentiment.get('negative', 0)
                    sentiment_count += 1
                
                sentiment_trends = sentiment.get('sentiment_trends', {})
                volatility = sentiment_trends.get('volatility', 0)
                emotional_volatility = max(emotional_volatility, volatility)
        
        if sentiment_count > 0:
            avg_negative = negative_sentiment_total / sentiment_count
            if avg_negative > 0.3:
                psych_risk['indicators'].append('elevated_negative_sentiment')
                psych_risk['mental_health_concerns']['negative_sentiment'] = avg_negative
        
        if emotional_volatility > 0.4:
            psych_risk['indicators'].append('emotional_instability')
            psych_risk['mental_health_concerns']['emotional_volatility'] = emotional_volatility
        
        # Check for behavioral changes indicating distress
        for analysis in temporal_analysis.values():
            behavioral_changes = analysis.get('behavioral_changes', {})
            change_points = behavioral_changes.get('change_points', [])
            
            if len(change_points) > 2:
                psych_risk['indicators'].append('frequent_behavioral_changes')
                psych_risk['behavioral_changes']['change_point_count'] = len(change_points)
            
            # Check for declining engagement
            trends = behavioral_changes.get('behavioral_trends', {})
            if trends.get('avg_daily_messages') == 'decreasing':
                psych_risk['indicators'].append('declining_engagement')
        
        # Analyze emotional expressions for mental health indicators
        relationship_analysis = analysis_results.get('relationship_analysis', {})
        emotional_patterns = relationship_analysis.get('emotional_patterns', {})
        
        depression_indicators = emotional_patterns.get('sadness', 0) + emotional_patterns.get('loneliness', 0)
        anxiety_indicators = emotional_patterns.get('fear', 0) + emotional_patterns.get('worry', 0)
        
        if depression_indicators > 15:
            psych_risk['indicators'].append('depression_indicators')
            psych_risk['mental_health_concerns']['depression_score'] = depression_indicators
        
        if anxiety_indicators > 10:
            psych_risk['indicators'].append('anxiety_indicators')
            psych_risk['mental_health_concerns']['anxiety_score'] = anxiety_indicators
        
        # Calculate risk score
        risk_score = 0.0
        
        if avg_negative > 0.4:
            risk_score += 0.3
        
        if emotional_volatility > 0.5:
            risk_score += 0.2
        
        if depression_indicators > 20:
            risk_score += 0.3
        
        if anxiety_indicators > 15:
            risk_score += 0.2
        
        if len(psych_risk['indicators']) >= 3:
            risk_score += 0.3
        
        psych_risk['risk_score'] = min(1.0, risk_score)
        psych_risk['risk_level'] = self._categorize_risk_level(psych_risk['risk_score'])
        
        return psych_risk
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk score into risk level."""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'moderate'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _identify_specific_risks(self, analysis_results: Dict[str, Any], risk_categories: Dict[str, Any]) -> Dict[str, Any]:
        """Identify specific risks across all categories."""
        specific_risks = {
            'immediate_concerns': [],
            'developing_issues': [],
            'long_term_risks': [],
            'risk_interactions': []
        }
        
        # Categorize risks by urgency
        for category, risk_data in risk_categories.items():
            risk_level = risk_data.get('risk_level', 'low')
            indicators = risk_data.get('indicators', [])
            
            if risk_level in ['critical', 'high']:
                specific_risks['immediate_concerns'].extend([
                    f"{category}: {indicator}" for indicator in indicators
                ])
            elif risk_level == 'moderate':
                specific_risks['developing_issues'].extend([
                    f"{category}: {indicator}" for indicator in indicators
                ])
            elif risk_level == 'low':
                specific_risks['long_term_risks'].extend([
                    f"{category}: {indicator}" for indicator in indicators
                ])
        
        # Identify risk interactions (when multiple risk categories are elevated)
        high_risk_categories = [cat for cat, data in risk_categories.items() 
                              if data.get('risk_level') in ['high', 'critical']]
        
        if len(high_risk_categories) >= 2:
            specific_risks['risk_interactions'].append(
                f"Multiple high-risk areas: {', '.join(high_risk_categories)}"
            )
        
        return specific_risks
    
    def _compile_risk_factors(self, risk_categories: Dict[str, Any]) -> List[str]:
        """Compile all risk factors from all categories."""
        all_risk_factors = []
        
        for category, risk_data in risk_categories.items():
            indicators = risk_data.get('indicators', [])
            all_risk_factors.extend(indicators)
        
        return list(set(all_risk_factors))  # Remove duplicates
    
    def _identify_protective_factors(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify protective factors that mitigate risk."""
        protective_factors = []
        
        temporal_analysis = analysis_results.get('temporal_analysis', {})
        
        # Regular activity patterns
        for analysis in temporal_analysis.values():
            activity_patterns = analysis.get('activity_patterns', {})
            regularity = activity_patterns.get('activity_regularity', {}).get('overall_regularity', 0)
            
            if regularity > 0.7:
                protective_factors.append('regular_activity_patterns')
            
            # Healthy sleep patterns
            circadian_patterns = analysis.get('circadian_patterns', {})
            night_pct = circadian_patterns.get('circadian_distribution', {}).get('night', {}).get('percentage', 0)
            
            if night_pct < 10:
                protective_factors.append('healthy_sleep_schedule')
            
            # Balanced weekly patterns
            weekly_patterns = analysis.get('weekly_patterns', {})
            consistency = weekly_patterns.get('weekly_consistency', 0)
            
            if consistency > 0.6:
                protective_factors.append('consistent_weekly_routine')
        
        # Positive emotional expressions
        relationship_analysis = analysis_results.get('relationship_analysis', {})
        emotional_patterns = relationship_analysis.get('emotional_patterns', {})
        
        positive_emotions = emotional_patterns.get('happiness', 0) + emotional_patterns.get('love', 0)
        if positive_emotions > 10:
            protective_factors.append('positive_emotional_expressions')
        
        # Moderate usage levels
        for analysis in temporal_analysis.values():
            usage_intensity = analysis.get('usage_intensity', {})
            daily_stats = usage_intensity.get('daily_stats', {})
            avg_daily = daily_stats.get('average_daily_messages', 0)
            
            if 10 <= avg_daily <= 50:  # Moderate usage range
                protective_factors.append('moderate_usage_levels')
        
        return list(set(protective_factors))
    
    def _generate_recommendations(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on risk assessment."""
        recommendations = []
        overall_risk = risk_assessment.get('overall_risk_score', 0)
        risk_categories = risk_assessment.get('risk_categories', {})
        
        # General recommendations based on overall risk
        if overall_risk >= 0.8:
            recommendations.append("Immediate intervention recommended - consider professional consultation")
        elif overall_risk >= 0.6:
            recommendations.append("Enhanced monitoring and support measures recommended")
        elif overall_risk >= 0.4:
            recommendations.append("Regular monitoring and preventive measures advised")
        
        # Category-specific recommendations
        if risk_categories.get('addiction_potential', {}).get('risk_level') in ['high', 'critical']:
            recommendations.extend([
                "Implement usage limits and scheduled breaks",
                "Monitor for signs of withdrawal or distress when app is unavailable",
                "Consider digital wellness counseling"
            ])
        
        if risk_categories.get('emotional_dependency', {}).get('risk_level') in ['high', 'critical']:
            recommendations.extend([
                "Assess real-world social support systems",
                "Monitor for signs of depression or anxiety",
                "Consider therapeutic intervention for attachment issues"
            ])
        
        if risk_categories.get('social_isolation', {}).get('risk_level') in ['high', 'critical']:
            recommendations.extend([
                "Encourage face-to-face social interactions",
                "Assess for underlying social anxiety or depression",
                "Monitor for signs of progressive social withdrawal"
            ])
        
        if risk_categories.get('sleep_disruption', {}).get('risk_level') in ['high', 'critical']:
            recommendations.extend([
                "Assess sleep hygiene and patterns",
                "Consider device usage restrictions before bedtime",
                "Monitor for impact on daily functioning"
            ])
        
        if risk_categories.get('psychological_impact', {}).get('risk_level') in ['high', 'critical']:
            recommendations.extend([
                "Mental health assessment recommended",
                "Monitor for signs of depression, anxiety, or other mental health concerns",
                "Consider professional psychological evaluation"
            ])
        
        return recommendations
    
    def _determine_intervention_urgency(self, risk_assessment: Dict[str, Any]) -> str:
        """Determine the urgency level for intervention."""
        overall_risk = risk_assessment.get('overall_risk_score', 0)
        immediate_concerns = risk_assessment.get('specific_risks', {}).get('immediate_concerns', [])
        
        if overall_risk >= 0.8 or len(immediate_concerns) >= 3:
            return 'immediate'
        elif overall_risk >= 0.6 or len(immediate_concerns) >= 2:
            return 'urgent'
        elif overall_risk >= 0.4 or len(immediate_concerns) >= 1:
            return 'moderate'
        else:
            return 'routine'
    
    def _determine_monitoring_priorities(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Determine priorities for ongoing monitoring."""
        priorities = []
        risk_categories = risk_assessment.get('risk_categories', {})
        
        # Sort categories by risk score
        sorted_categories = sorted(risk_categories.items(), 
                                 key=lambda x: x[1].get('risk_score', 0), 
                                 reverse=True)
        
        # Add top 3 risk categories as priorities
        for category, risk_data in sorted_categories[:3]:
            if risk_data.get('risk_score', 0) > 0.3:
                priorities.append(category.replace('_', ' ').title())
        
        # Add specific high-priority indicators
        specific_risks = risk_assessment.get('specific_risks', {})
        immediate_concerns = specific_risks.get('immediate_concerns', [])
        
        if immediate_concerns:
            priorities.append("Monitor immediate concerns closely")
        
        if not priorities:
            priorities.append("General behavioral pattern monitoring")
        
        return priorities[:5]  # Limit to 5 priorities
