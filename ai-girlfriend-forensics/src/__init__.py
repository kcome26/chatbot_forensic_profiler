"""
Main package initialization.
"""
from .forensic_analyzer import ForensicAnalyzer
from .database_parsers import DatabaseManager, DatabaseParserFactory
from .analysis import TextAnalyzer, TemporalAnalyzer
from .profiling import UserProfiler, RiskAssessment
from .reporting import ForensicReportGenerator

__version__ = "1.0.0"

__all__ = [
    'ForensicAnalyzer',
    'DatabaseManager',
    'DatabaseParserFactory',
    'TextAnalyzer',
    'TemporalAnalyzer',
    'UserProfiler',
    'RiskAssessment',
    'ForensicReportGenerator'
]
