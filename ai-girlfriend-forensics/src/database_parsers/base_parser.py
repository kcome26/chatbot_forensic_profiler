"""
Base database parser interface for the forensic analyzer.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime


class DatabaseParser(ABC):
    """Abstract base class for database parsers."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = {}
        self.metadata = {}
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def discover_schema(self) -> Dict[str, Any]:
        """Discover the database schema and structure."""
        pass
    
    @abstractmethod
    def extract_tables(self) -> Dict[str, pd.DataFrame]:
        """Extract all tables/collections as DataFrames."""
        pass
    
    @abstractmethod
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract database metadata (creation time, app info, etc.)."""
        pass
    
    def classify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Classify columns by their likely content type.
        Returns dict mapping column names to types.
        """
        classifications = {}
        
        for col in df.columns:
            sample_data = df[col].dropna().head(100)
            if len(sample_data) == 0:
                classifications[col] = 'empty'
                continue
                
            # Check for timestamps - be more strict
            if self._is_timestamp_column(col, sample_data):
                classifications[col] = 'timestamp'
            # Check for IDs
            elif self._is_id_column(col, sample_data):
                classifications[col] = 'id'
            # Check for text content
            elif self._is_text_column(sample_data):
                classifications[col] = 'text'
            # Check for user info
            elif self._is_user_info_column(col, sample_data):
                classifications[col] = 'user_info'
            # Check for conversation data
            elif self._is_conversation_column(col, sample_data):
                classifications[col] = 'conversation'
            else:
                classifications[col] = 'other'
                
        return classifications
    
    def _is_timestamp_column(self, col_name: str, series: pd.Series) -> bool:
        """Check if column contains timestamp data."""
        # First check if column name suggests it's a timestamp
        timestamp_keywords = ['timestamp', 'created_at', 'updated_at', 'date', 'time']
        col_lower = col_name.lower()
        
        # Exclude columns that are clearly not timestamps
        exclude_keywords = ['is_', 'hide_', 'uses_', 'from_', 'reroll_', 'romantic_', 'min', 'max', 'smooth', 'duration_in_second']
        if any(col_lower.startswith(keyword) for keyword in exclude_keywords):
            return False
            
        # Must have timestamp-related name or be explicitly named as such
        has_timestamp_name = any(keyword in col_lower for keyword in timestamp_keywords)
        
        try:
            # Try to parse as datetime
            pd.to_datetime(series.iloc[0])
            return has_timestamp_name or col_lower in ['timestamp', 'timestamp_ms']
        except:
            # Check for unix timestamps
            if series.dtype in ['int64', 'float64'] and has_timestamp_name:
                # Check if values are in reasonable timestamp range
                first_val = series.iloc[0]
                if isinstance(first_val, (int, float)) and 1000000000 <= first_val <= 9999999999999:  # Unix timestamp range
                    # Additional checks to avoid false positives
                    unique_vals = series.nunique()
                    if unique_vals <= 2:  # Likely a boolean column
                        return False
                    
                    # Check if all values are within reasonable timestamp range
                    valid_timestamps = series[(series >= 1000000000) & (series <= 9999999999999)]
                    if len(valid_timestamps) / len(series) < 0.8:  # Less than 80% valid timestamps
                        return False
                        
                    return True
        return False
    
    def _is_id_column(self, col_name: str, series: pd.Series) -> bool:
        """Check if column contains ID data."""
        id_keywords = ['id', 'uuid', 'key', 'index']
        col_lower = col_name.lower()
        return any(keyword in col_lower for keyword in id_keywords)
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if column contains text content."""
        if series.dtype == 'object':
            # Check average string length
            avg_length = series.str.len().mean()
            return avg_length > 10  # Arbitrary threshold for meaningful text
        return False
    
    def _is_user_info_column(self, col_name: str, series: pd.Series) -> bool:
        """Check if column contains user information."""
        user_keywords = ['name', 'age', 'gender', 'location', 'preference', 'profile']
        col_lower = col_name.lower()
        return any(keyword in col_lower for keyword in user_keywords)
    
    def _is_conversation_column(self, col_name: str, series: pd.Series) -> bool:
        """Check if column contains conversation data."""
        conv_keywords = ['message', 'chat', 'conversation', 'reply', 'response']
        col_lower = col_name.lower()
        return any(keyword in col_lower for keyword in conv_keywords)
