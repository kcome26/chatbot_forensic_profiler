#!/usr/bin/env python3
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Starting basic test...")

try:
    from src.forensic_analyzer import ForensicAnalyzer
    print("✓ Import successful")
    
    analyzer = ForensicAnalyzer()
    print("✓ Analyzer created")
    
    print("✓ Basic test completed successfully!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
