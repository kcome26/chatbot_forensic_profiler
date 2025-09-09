#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing database parsers...")
    from database_parsers import DatabaseManager
    print("✓ DatabaseManager imported successfully")
    
    print("Testing analysis modules...")
    from analysis.text_analyzer import TextAnalyzer
    print("✓ TextAnalyzer imported successfully")
    
    from analysis.temporal_analyzer import TemporalAnalyzer
    print("✓ TemporalAnalyzer imported successfully")
    
    print("Testing profiling modules...")
    from profiling.user_profiler import UserProfiler
    print("✓ UserProfiler imported successfully")
    
    from profiling.risk_assessment import RiskAssessment
    print("✓ RiskAssessment imported successfully")
    
    print("Testing reporting modules...")
    from reporting.report_generator import ForensicReportGenerator
    print("✓ ForensicReportGenerator imported successfully")
    
    print("Testing main forensic analyzer...")
    from forensic_analyzer import ForensicAnalyzer
    print("✓ ForensicAnalyzer imported successfully")
    
    # Test basic instantiation
    analyzer = ForensicAnalyzer()
    print("✓ ForensicAnalyzer instantiated successfully")
    
    print("\n🎉 All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
