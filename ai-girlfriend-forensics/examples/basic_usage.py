"""
Basic usage example for the AI Girlfriend Forensics Analyzer.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.forensic_analyzer import ForensicAnalyzer
from src.reporting.report_generator import ForensicReportGenerator


def main():
    """Basic usage example."""
    print("AI Girlfriend Forensics Analyzer - Basic Example")
    print("=" * 50)
    
    # Initialize the forensic analyzer
    analyzer = ForensicAnalyzer()
    
    # Example database path (you would replace with actual path)
    # db_path = "path/to/your/database.db"
    # 
    # For demo purposes, we'll show what the process would look like:
    print("Step 1: Load database")
    print("analyzer.load_database('path/to/database.db')")
    
    print("\nStep 2: Analyze database")
    print("analysis_results = analyzer.analyze_database()")
    
    print("\nStep 3: Generate forensic profile")
    print("forensic_profile = analyzer.generate_forensic_profile()")
    
    print("\nStep 4: Generate report")
    print("report_generator = ForensicReportGenerator()")
    print("report_path = report_generator.generate_report(analysis_results, forensic_profile)")
    
    print("\nStep 5: Save results")
    print("analyzer.save_results('forensic_analysis_results.json')")
    
    print("\n" + "=" * 50)
    print("Example complete!")
    print("\nTo use with real data:")
    print("1. Replace 'path/to/database.db' with actual database path")
    print("2. Ensure database is from supported app (SQLite or JSON)")
    print("3. Run the analysis")
    print("4. Review generated report and JSON results")


def analyze_sample_database(db_path: str):
    """
    Complete analysis workflow for a real database.
    
    Args:
        db_path: Path to the database file
    """
    try:
        # Initialize analyzer
        analyzer = ForensicAnalyzer()
        
        # Load database
        print(f"Loading database: {db_path}")
        if not analyzer.load_database(db_path):
            print("Failed to load database")
            return
        
        # Perform analysis
        print("Performing comprehensive analysis...")
        analysis_results = analyzer.analyze_database()
        
        # Generate forensic profile
        print("Generating forensic profile...")
        forensic_profile = analyzer.generate_forensic_profile()
        
        # Generate report
        print("Generating HTML report...")
        report_generator = ForensicReportGenerator()
        report_path = report_generator.generate_report(analysis_results, forensic_profile)
        
        # Save JSON results
        print("Saving detailed results...")
        json_path = analyzer.save_results()
        
        print("\nAnalysis Complete!")
        print(f"HTML Report: {report_path}")
        print(f"JSON Results: {json_path}")
        
        # Display summary
        risk_level = forensic_profile.get('risk_factors', {}).get('overall_risk_level', 'unknown')
        app_detected = forensic_profile.get('profile_metadata', {}).get('app_identified', 'unknown')
        
        print(f"\nQuick Summary:")
        print(f"- Detected App: {app_detected}")
        print(f"- Risk Level: {risk_level}")
        print(f"- Analysis Quality: {forensic_profile.get('profile_metadata', {}).get('data_quality_score', 0):.2f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If database path provided as command line argument
        db_path = sys.argv[1]
        analyze_sample_database(db_path)
    else:
        # Show basic example
        main()
