"""
Temporal analysis module for examining time-based patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalyzer:
    """Analyzer for temporal patterns and behavioral changes over time."""
    
    def __init__(self):
        self.timezone_offset = 0  # Will be detected from data
        
    def analyze_temporal_patterns(self, df: pd.DataFrame, 
                                timestamp_col: str = 'timestamp',
                                user_col: str = 'user_id',
                                message_col: str = 'message') -> Dict[str, Any]:
        """
        Comprehensive temporal analysis of user behavior.
        
        Args:
            df: DataFrame with temporal data
            timestamp_col: Column containing timestamps
            user_col: Column containing user identifiers
            message_col: Column containing message content
            
        Returns:
            Dictionary containing temporal analysis results
        """
        # Prepare temporal data
        df = self._prepare_temporal_data(df, timestamp_col)
        
        if df.empty or len(df) == 0:
            return {'error': 'No valid temporal data found'}
        
        results = {
            'activity_patterns': self._analyze_activity_patterns(df, timestamp_col),
            'usage_intensity': self._analyze_usage_intensity(df, timestamp_col),
            'behavioral_changes': self._analyze_behavioral_changes(df, timestamp_col, message_col),
            'session_analysis': self._analyze_sessions(df, timestamp_col),
            'circadian_patterns': self._analyze_circadian_patterns(df, timestamp_col),
            'weekly_patterns': self._analyze_weekly_patterns(df, timestamp_col),
            'seasonal_patterns': self._analyze_seasonal_patterns(df, timestamp_col),
            'engagement_evolution': self._analyze_engagement_evolution(df, timestamp_col, message_col)
        }
        
        return results
    
    def _prepare_temporal_data(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Prepare and clean temporal data."""
        df = df.copy()
        
        try:
            # Handle different timestamp formats
            if df[timestamp_col].dtype in ['int64', 'float64']:
                # Assume Unix timestamp in milliseconds if values are large
                if df[timestamp_col].iloc[0] > 9999999999:  # Likely milliseconds
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
                else:  # Likely seconds
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
            else:
                # Convert string timestamps to datetime
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Remove rows with invalid timestamps
            df = df.dropna(subset=[timestamp_col])
            
            # Sort by timestamp
            df = df.sort_values(timestamp_col)
            
            # Detect timezone patterns
            self._detect_timezone_patterns(df, timestamp_col)
            
            return df
            
        except Exception as e:
            print(f"Error preparing temporal data for {timestamp_col}: {e}")
            return pd.DataFrame()
    
    def _detect_timezone_patterns(self, df: pd.DataFrame, timestamp_col: str):
        """Detect likely timezone from activity patterns."""
        # Extract hour of activity
        hours = df[timestamp_col].dt.hour
        
        # Find peak activity hours
        hour_counts = hours.value_counts()
        peak_hours = hour_counts.nlargest(3).index.tolist()
        
        # Estimate timezone based on peak activity (assuming typical waking hours)
        avg_peak = np.mean(peak_hours)
        
        # Typical peak activity is around 8 PM (20:00)
        if 18 <= avg_peak <= 22:
            self.timezone_offset = 0  # Likely local time
        else:
            # Estimate offset needed to shift peak to evening
            target_hour = 20
            self.timezone_offset = target_hour - avg_peak
    
    def _analyze_activity_patterns(self, df: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze daily and hourly activity patterns."""
        # Hourly patterns
        df['hour'] = df[timestamp_col].dt.hour
        hourly_activity = df['hour'].value_counts().sort_index()
        
        # Daily patterns (day of week)
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        daily_activity = df['day_of_week'].value_counts().sort_index()
        
        # Monthly patterns
        df['month'] = df[timestamp_col].dt.month
        monthly_activity = df['month'].value_counts().sort_index()
        
        # Activity intensity by time period
        morning_activity = hourly_activity[6:12].sum()  # 6 AM - 12 PM
        afternoon_activity = hourly_activity[12:18].sum()  # 12 PM - 6 PM
        evening_activity = hourly_activity[18:24].sum()  # 6 PM - 12 AM
        night_activity = hourly_activity[0:6].sum()  # 12 AM - 6 AM
        
        total_activity = len(df)
        
        return {
            'hourly_distribution': hourly_activity.to_dict(),
            'daily_distribution': daily_activity.to_dict(),
            'monthly_distribution': monthly_activity.to_dict(),
            'peak_activity_hour': int(hourly_activity.idxmax()),
            'peak_activity_day': int(daily_activity.idxmax()),
            'time_period_breakdown': {
                'morning': {'count': int(morning_activity), 'percentage': float(morning_activity/total_activity*100)},
                'afternoon': {'count': int(afternoon_activity), 'percentage': float(afternoon_activity/total_activity*100)},
                'evening': {'count': int(evening_activity), 'percentage': float(evening_activity/total_activity*100)},
                'night': {'count': int(night_activity), 'percentage': float(night_activity/total_activity*100)}
            },
            'activity_regularity': self._calculate_activity_regularity(hourly_activity, daily_activity)
        }
    
    def _analyze_usage_intensity(self, df: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze usage intensity and frequency patterns."""
        # Daily message counts
        daily_counts = df.groupby(df[timestamp_col].dt.date).size()
        
        # Weekly message counts
        df['week'] = df[timestamp_col].dt.isocalendar().week
        weekly_counts = df.groupby('week').size()
        
        # Calculate usage streaks
        usage_streaks = self._calculate_usage_streaks(daily_counts)
        
        # Intensity categories
        low_activity_days = (daily_counts <= daily_counts.quantile(0.25)).sum()
        medium_activity_days = ((daily_counts > daily_counts.quantile(0.25)) & 
                               (daily_counts <= daily_counts.quantile(0.75))).sum()
        high_activity_days = (daily_counts > daily_counts.quantile(0.75)).sum()
        
        return {
            'daily_stats': {
                'average_daily_messages': float(daily_counts.mean()),
                'max_daily_messages': int(daily_counts.max()),
                'min_daily_messages': int(daily_counts.min()),
                'daily_variance': float(daily_counts.var()),
                'active_days': int(len(daily_counts))
            },
            'weekly_stats': {
                'average_weekly_messages': float(weekly_counts.mean()),
                'max_weekly_messages': int(weekly_counts.max()),
                'weekly_variance': float(weekly_counts.var())
            },
            'activity_distribution': {
                'low_activity_days': int(low_activity_days),
                'medium_activity_days': int(medium_activity_days),
                'high_activity_days': int(high_activity_days)
            },
            'usage_streaks': usage_streaks,
            'usage_consistency': self._calculate_usage_consistency(daily_counts)
        }
    
    def _analyze_behavioral_changes(self, df: pd.DataFrame, timestamp_col: str, message_col: str) -> Dict[str, Any]:
        """Analyze how behavior changes over time."""
        if len(df) < 30:  # Need sufficient data for trend analysis
            return {'insufficient_data': True}
        
        # Split data into time periods
        df['period'] = pd.cut(df.index, bins=5, labels=['period_1', 'period_2', 'period_3', 'period_4', 'period_5'])
        
        period_stats = {}
        for period in df['period'].unique():
            if pd.isna(period):
                continue
                
            period_data = df[df['period'] == period]
            
            # Calculate stats for this period
            daily_messages = period_data.groupby(period_data[timestamp_col].dt.date).size()
            message_lengths = period_data[message_col].astype(str).str.len()
            
            period_stats[period] = {
                'avg_daily_messages': float(daily_messages.mean()) if len(daily_messages) > 0 else 0,
                'avg_message_length': float(message_lengths.mean()),
                'total_messages': len(period_data),
                'unique_days': len(daily_messages),
                'activity_variance': float(daily_messages.var()) if len(daily_messages) > 1 else 0
            }
        
        # Calculate trends
        trends = self._calculate_behavioral_trends(period_stats)
        
        return {
            'period_analysis': period_stats,
            'behavioral_trends': trends,
            'change_points': self._detect_change_points(df, timestamp_col, message_col)
        }
    
    def _analyze_sessions(self, df: pd.DataFrame, timestamp_col: str, session_gap_minutes: int = 30) -> Dict[str, Any]:
        """Analyze conversation sessions and their characteristics."""
        # Define sessions based on time gaps
        df = df.sort_values(timestamp_col)
        time_diffs = df[timestamp_col].diff()
        
        # Mark session boundaries
        session_boundaries = time_diffs > timedelta(minutes=session_gap_minutes)
        df['session_id'] = session_boundaries.cumsum()
        
        # Analyze sessions
        session_stats = df.groupby('session_id').agg({
            timestamp_col: ['min', 'max', 'count'],
        })
        
        session_stats.columns = ['session_start', 'session_end', 'message_count']
        session_stats['session_duration'] = (session_stats['session_end'] - 
                                            session_stats['session_start']).dt.total_seconds() / 60  # minutes
        
        # Session characteristics
        avg_session_duration = session_stats['session_duration'].mean()
        avg_messages_per_session = session_stats['message_count'].mean()
        total_sessions = len(session_stats)
        
        # Session patterns
        session_hours = session_stats['session_start'].dt.hour
        session_days = session_stats['session_start'].dt.dayofweek
        
        return {
            'session_overview': {
                'total_sessions': int(total_sessions),
                'avg_session_duration_minutes': float(avg_session_duration),
                'avg_messages_per_session': float(avg_messages_per_session),
                'longest_session_minutes': float(session_stats['session_duration'].max()),
                'shortest_session_minutes': float(session_stats['session_duration'].min())
            },
            'session_timing': {
                'preferred_session_hours': session_hours.value_counts().head(5).to_dict(),
                'preferred_session_days': session_days.value_counts().to_dict()
            },
            'session_intensity': self._categorize_session_intensity(session_stats)
        }
    
    def _analyze_circadian_patterns(self, df: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze circadian rhythm patterns."""
        df['hour'] = df[timestamp_col].dt.hour
        hourly_counts = df['hour'].value_counts().sort_index()
        
        # Categorize hours into circadian phases
        morning_hours = list(range(6, 12))
        afternoon_hours = list(range(12, 18))
        evening_hours = list(range(18, 24))
        night_hours = list(range(0, 6))
        
        morning_activity = hourly_counts[hourly_counts.index.isin(morning_hours)].sum()
        afternoon_activity = hourly_counts[hourly_counts.index.isin(afternoon_hours)].sum()
        evening_activity = hourly_counts[hourly_counts.index.isin(evening_hours)].sum()
        night_activity = hourly_counts[hourly_counts.index.isin(night_hours)].sum()
        
        total_activity = len(df)
        
        # Detect chronotype (morning person vs night owl)
        if total_activity == 0:
            morning_ratio = 0
            night_ratio = 0
        else:
            morning_ratio = morning_activity / total_activity
            night_ratio = night_activity / total_activity
        
        if morning_ratio > 0.3:
            chronotype = 'morning_person'
        elif night_ratio > 0.2:
            chronotype = 'night_owl'
        else:
            chronotype = 'intermediate'
        
        return {
            'circadian_distribution': {
                'morning': {'count': int(morning_activity), 'percentage': float(morning_activity/total_activity*100)},
                'afternoon': {'count': int(afternoon_activity), 'percentage': float(afternoon_activity/total_activity*100)},
                'evening': {'count': int(evening_activity), 'percentage': float(evening_activity/total_activity*100)},
                'night': {'count': int(night_activity), 'percentage': float(night_activity/total_activity*100)}
            },
            'chronotype': chronotype,
            'peak_activity_period': self._identify_peak_period(hourly_counts),
            'circadian_regularity': self._calculate_circadian_regularity(hourly_counts)
        }
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze weekly usage patterns."""
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_name'] = df[timestamp_col].dt.day_name()
        
        daily_counts = df['day_of_week'].value_counts().sort_index()
        
        # Categorize weekdays vs weekends
        weekday_activity = daily_counts[daily_counts.index.isin([0, 1, 2, 3, 4])].sum()  # Mon-Fri
        weekend_activity = daily_counts[daily_counts.index.isin([5, 6])].sum()  # Sat-Sun
        
        total_activity = len(df)
        
        return {
            'daily_distribution': {
                'monday': int(daily_counts.get(0, 0)),
                'tuesday': int(daily_counts.get(1, 0)),
                'wednesday': int(daily_counts.get(2, 0)),
                'thursday': int(daily_counts.get(3, 0)),
                'friday': int(daily_counts.get(4, 0)),
                'saturday': int(daily_counts.get(5, 0)),
                'sunday': int(daily_counts.get(6, 0))
            },
            'weekday_vs_weekend': {
                'weekday_percentage': float(weekday_activity / total_activity * 100) if total_activity > 0 else 0,
                'weekend_percentage': float(weekend_activity / total_activity * 100) if total_activity > 0 else 0,
                'weekend_preference': weekend_activity > weekday_activity
            },
            'most_active_day': df['day_name'].value_counts().index[0],
            'weekly_consistency': self._calculate_weekly_consistency(daily_counts)
        }
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        """Analyze seasonal usage patterns."""
        df['month'] = df[timestamp_col].dt.month
        df['season'] = df['month'].apply(self._get_season)
        
        monthly_counts = df['month'].value_counts().sort_index()
        seasonal_counts = df['season'].value_counts()
        
        return {
            'monthly_distribution': monthly_counts.to_dict(),
            'seasonal_distribution': seasonal_counts.to_dict(),
            'peak_season': seasonal_counts.idxmax(),
            'seasonal_variance': float(seasonal_counts.var()),
            'usage_seasonality': self._calculate_seasonality_score(monthly_counts)
        }
    
    def _analyze_engagement_evolution(self, df: pd.DataFrame, timestamp_col: str, message_col: str) -> Dict[str, Any]:
        """Analyze how engagement evolves over time."""
        # Calculate rolling engagement metrics
        df = df.sort_values(timestamp_col)
        df['message_length'] = df[message_col].astype(str).str.len()
        
        # Weekly rolling averages
        df['week'] = df[timestamp_col].dt.isocalendar().week
        weekly_engagement = df.groupby('week').agg({
            'message_length': 'mean',
            timestamp_col: 'count'
        }).rename(columns={timestamp_col: 'message_count'})
        
        # Calculate engagement trends
        if len(weekly_engagement) > 1:
            message_count_trend = np.polyfit(range(len(weekly_engagement)), 
                                           weekly_engagement['message_count'], 1)[0]
            message_length_trend = np.polyfit(range(len(weekly_engagement)), 
                                            weekly_engagement['message_length'], 1)[0]
        else:
            message_count_trend = 0
            message_length_trend = 0
        
        return {
            'weekly_engagement': weekly_engagement.to_dict('index'),
            'engagement_trends': {
                'message_frequency_trend': float(message_count_trend),
                'message_length_trend': float(message_length_trend),
                'overall_engagement_direction': 'increasing' if message_count_trend > 0 else 'decreasing'
            },
            'engagement_phases': self._identify_engagement_phases(weekly_engagement)
        }
    
    # Helper methods
    def _calculate_activity_regularity(self, hourly_activity: pd.Series, daily_activity: pd.Series) -> Dict[str, float]:
        """Calculate regularity scores for activity patterns."""
        # Calculate coefficient of variation for regularity
        hourly_cv = hourly_activity.std() / hourly_activity.mean() if hourly_activity.mean() > 0 else 0
        daily_cv = daily_activity.std() / daily_activity.mean() if daily_activity.mean() > 0 else 0
        
        # Lower CV means more regular
        hourly_regularity = max(0, 1 - hourly_cv / 2)  # Normalize
        daily_regularity = max(0, 1 - daily_cv / 2)
        
        return {
            'hourly_regularity': float(hourly_regularity),
            'daily_regularity': float(daily_regularity),
            'overall_regularity': float((hourly_regularity + daily_regularity) / 2)
        }
    
    def _calculate_usage_streaks(self, daily_counts: pd.Series) -> Dict[str, int]:
        """Calculate usage streak statistics."""
        # Convert to binary (active/inactive days)
        active_days = daily_counts > 0
        
        # Find streaks
        streaks = []
        current_streak = 0
        
        for is_active in active_days:
            if is_active:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        # Add final streak if active
        if current_streak > 0:
            streaks.append(current_streak)
        
        return {
            'longest_streak': max(streaks) if streaks else 0,
            'average_streak': int(np.mean(streaks)) if streaks else 0,
            'total_streaks': len(streaks)
        }
    
    def _calculate_usage_consistency(self, daily_counts: pd.Series) -> float:
        """Calculate consistency score for usage patterns."""
        if len(daily_counts) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        cv = daily_counts.std() / daily_counts.mean() if daily_counts.mean() > 0 else float('inf')
        
        # Convert to consistency score (0-1, higher is more consistent)
        consistency = max(0, 1 - cv / 2)
        return float(consistency)
    
    def _calculate_behavioral_trends(self, period_stats: Dict) -> Dict[str, str]:
        """Calculate trends in behavioral metrics."""
        periods = sorted(period_stats.keys())
        if len(periods) < 2:
            return {}
        
        trends = {}
        metrics = ['avg_daily_messages', 'avg_message_length', 'activity_variance']
        
        for metric in metrics:
            values = [period_stats[period][metric] for period in periods]
            
            # Calculate trend (protect against division by zero)
            if len(values) < 2 or values[0] == 0:
                trends[metric] = 'unknown'
                continue
                
            if values[-1] > values[0]:
                if (values[-1] - values[0]) / values[0] > 0.2:
                    trends[metric] = 'increasing'
                else:
                    trends[metric] = 'stable'
            else:
                if (values[0] - values[-1]) / values[0] > 0.2:
                    trends[metric] = 'decreasing'
                else:
                    trends[metric] = 'stable'
        
        return trends
    
    def _detect_change_points(self, df: pd.DataFrame, timestamp_col: str, message_col: str) -> List[Dict]:
        """Detect significant changes in behavior patterns."""
        # Simple change point detection based on activity level
        df['date'] = df[timestamp_col].dt.date
        daily_activity = df.groupby('date').size()
        
        if len(daily_activity) < 10:
            return []
        
        # Calculate rolling mean and detect significant changes
        rolling_mean = daily_activity.rolling(window=7, min_periods=3).mean()
        changes = []
        
        for i in range(7, len(rolling_mean)):
            current_mean = rolling_mean.iloc[i]
            previous_mean = rolling_mean.iloc[i-7]
            
            if abs(current_mean - previous_mean) > previous_mean * 0.5 and previous_mean > 0:  # 50% change threshold
                changes.append({
                    'date': daily_activity.index[i].isoformat(),
                    'change_type': 'increase' if current_mean > previous_mean else 'decrease',
                    'magnitude': float(abs(current_mean - previous_mean) / previous_mean)
                })
        
        return changes
    
    def _categorize_session_intensity(self, session_stats: pd.DataFrame) -> Dict[str, int]:
        """Categorize sessions by intensity."""
        # Define intensity categories based on message count and duration
        short_sessions = ((session_stats['session_duration'] < 15) | 
                         (session_stats['message_count'] < 5)).sum()
        medium_sessions = ((session_stats['session_duration'].between(15, 60)) & 
                          (session_stats['message_count'].between(5, 20))).sum()
        long_sessions = ((session_stats['session_duration'] > 60) | 
                        (session_stats['message_count'] > 20)).sum()
        
        return {
            'short_sessions': int(short_sessions),
            'medium_sessions': int(medium_sessions),
            'long_sessions': int(long_sessions)
        }
    
    def _identify_peak_period(self, hourly_counts: pd.Series) -> str:
        """Identify the time period with peak activity."""
        peak_hour = hourly_counts.idxmax()
        
        if 6 <= peak_hour < 12:
            return 'morning'
        elif 12 <= peak_hour < 18:
            return 'afternoon'
        elif 18 <= peak_hour < 24:
            return 'evening'
        else:
            return 'night'
    
    def _calculate_circadian_regularity(self, hourly_counts: pd.Series) -> float:
        """Calculate circadian rhythm regularity."""
        # Calculate how concentrated activity is in specific hours
        total_activity = hourly_counts.sum()
        if total_activity == 0:
            return 0.0
        
        # Calculate entropy (lower entropy = more regular)
        probabilities = hourly_counts / total_activity
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Convert to regularity score (0-1)
        max_entropy = np.log2(24)  # Maximum possible entropy for 24 hours
        regularity = 1 - (entropy / max_entropy)
        
        return float(regularity)
    
    def _calculate_weekly_consistency(self, daily_counts: pd.Series) -> float:
        """Calculate weekly pattern consistency."""
        if len(daily_counts) == 0:
            return 0.0
        
        # Calculate coefficient of variation
        cv = daily_counts.std() / daily_counts.mean() if daily_counts.mean() > 0 else float('inf')
        
        # Convert to consistency score
        consistency = max(0, 1 - cv / 2)
        return float(consistency)
    
    def _get_season(self, month: int) -> str:
        """Get season from month number."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _calculate_seasonality_score(self, monthly_counts: pd.Series) -> float:
        """Calculate how seasonal the usage pattern is."""
        if len(monthly_counts) < 12:
            return 0.0
        
        # Calculate coefficient of variation
        cv = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
        
        # Higher CV indicates more seasonality
        seasonality = min(1.0, cv)
        return float(seasonality)
    
    def _identify_engagement_phases(self, weekly_engagement: pd.DataFrame) -> List[Dict]:
        """Identify distinct phases of engagement."""
        if len(weekly_engagement) < 4:
            return []
        
        phases = []
        message_counts = weekly_engagement['message_count'].values
        
        # Simple phase detection based on activity level changes
        high_threshold = np.percentile(message_counts, 75)
        low_threshold = np.percentile(message_counts, 25)
        
        current_phase = None
        phase_start = 0
        
        for i, count in enumerate(message_counts):
            if count > high_threshold:
                phase_type = 'high_engagement'
            elif count < low_threshold:
                phase_type = 'low_engagement'
            else:
                phase_type = 'moderate_engagement'
            
            if phase_type != current_phase:
                if current_phase is not None:
                    phases.append({
                        'phase_type': current_phase,
                        'start_week': phase_start,
                        'end_week': i - 1,
                        'duration_weeks': i - phase_start
                    })
                current_phase = phase_type
                phase_start = i
        
        # Add final phase
        if current_phase is not None:
            phases.append({
                'phase_type': current_phase,
                'start_week': phase_start,
                'end_week': len(message_counts) - 1,
                'duration_weeks': len(message_counts) - phase_start
            })
        
        return phases
