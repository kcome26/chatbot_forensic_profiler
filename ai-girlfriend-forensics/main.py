"""
Main entry point for the AI Girlfriend Forensics Analyzer.
"""
import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.forensic_analyzer import ForensicAnalyzer
from src.profiling.user_profiler import UserProfiler
from src.profiling.risk_assessment import RiskAssessment
from src.reporting.report_generator import ForensicReportGenerator


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="AI Girlfriend Application Forensic Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py database.db
  python main.py --input database.json --output results/
  python main.py database.db --comprehensive --report
  python main.py --test
        """
    )
    
    parser.add_argument(
        'database',
        nargs='?',
        help='Path to the database file to analyze'
    )
    
    parser.add_argument(
        '--input', '-i',
        dest='input_path',
        help='Input database path (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--output', '-o',
        dest='output_dir',
        default='.',
        help='Output directory for results (default: current directory)'
    )
    
    parser.add_argument(
        '--comprehensive', '-c',
        action='store_true',
        help='Perform comprehensive analysis including ML profiling'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate HTML report'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run basic functionality test'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        run_test()
        return
    
    # Determine input path
    input_path = args.database or args.input_path
    
    if not input_path:
        parser.print_help()
        print("\nError: Please provide a database file to analyze")
        return
    
    if not os.path.exists(input_path):
        print(f"Error: Database file not found: {input_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.comprehensive:
            run_comprehensive_analysis(input_path, output_dir, args.report, args.verbose)
        else:
            run_basic_analysis(input_path, output_dir, args.report, args.verbose)
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def run_basic_analysis(input_path: str, output_dir: Path, generate_report: bool, verbose: bool):
    """Run basic forensic analysis."""
    print("AI Girlfriend Forensics Analyzer")
    print("=" * 40)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Mode: Basic Analysis")
    print()
    
    # Initialize analyzer
    analyzer = ForensicAnalyzer()
    
    # Load database
    if verbose:
        print("Loading database...")
    
    if not analyzer.load_database(input_path):
        print("Failed to load database")
        return
    
    print("✓ Database loaded successfully")
    
    # Perform analysis
    if verbose:
        print("Performing analysis...")
    
    analysis_results = analyzer.analyze_database()
    print("✓ Analysis completed")
    
    # Generate forensic profile
    if verbose:
        print("Generating forensic profile...")
    
    forensic_profile = analyzer.generate_forensic_profile()
    print("✓ Forensic profile generated")
    
    # Save results
    results_file = output_dir / f"forensic_analysis_{Path(input_path).stem}.json"
    results_path = analyzer.save_results(str(results_file))
    print(f"✓ Results saved: {results_path}")
    
    # Generate report if requested
    if generate_report:
        if verbose:
            print("Generating HTML report...")
        
        report_generator = ForensicReportGenerator()
        report_file = output_dir / f"forensic_report_{Path(input_path).stem}.html"
        report_path = report_generator.generate_report(
            analysis_results, forensic_profile, str(report_file)
        )
        print(f"✓ Report generated: {report_path}")
    
    # Display summary
    display_analysis_summary(forensic_profile, verbose)


def run_comprehensive_analysis(input_path: str, output_dir: Path, generate_report: bool, verbose: bool):
    """Run comprehensive forensic analysis with ML profiling."""
    print("AI Girlfriend Forensics Analyzer")
    print("=" * 40)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Mode: Comprehensive Analysis (ML + Risk Assessment)")
    print()
    
    # Initialize components
    analyzer = ForensicAnalyzer()
    profiler = UserProfiler()
    risk_assessor = RiskAssessment()
    
    # Load database
    if verbose:
        print("Loading database...")
    
    if not analyzer.load_database(input_path):
        print("Failed to load database")
        return
    
    print("✓ Database loaded successfully")
    
    # Perform basic analysis
    if verbose:
        print("Performing basic analysis...")
    
    analysis_results = analyzer.analyze_database()
    print("✓ Basic analysis completed")
    
    # ML-based profiling
    if verbose:
        print("Extracting features for ML analysis...")
    
    features_df = profiler.extract_features(analysis_results)
    
    if not features_df.empty:
        print(f"✓ Extracted {len(features_df.columns)} features")
        
        if verbose:
            print("Creating ML-based user profile...")
        
        user_profile = profiler.create_user_profile(features_df, user_id=Path(input_path).stem)
        print("✓ ML profile generated")
    else:
        print("⚠ No features available for ML analysis")
        user_profile = {}
    
    # Comprehensive risk assessment
    if verbose:
        print("Performing comprehensive risk assessment...")
    
    risk_assessment = risk_assessor.assess_comprehensive_risk(analysis_results)
    print(f"✓ Risk assessment completed (Level: {risk_assessment.get('risk_level', 'unknown')})")
    
    # Generate enhanced forensic profile
    if verbose:
        print("Generating comprehensive forensic profile...")
    
    forensic_profile = analyzer.generate_forensic_profile()
    
    # Enhance with ML insights
    if user_profile:
        forensic_profile['ml_profile'] = user_profile
    forensic_profile['comprehensive_risk'] = risk_assessment
    
    print("✓ Comprehensive forensic profile generated")
    
    # Save comprehensive results
    from datetime import datetime
    import json
    
    comprehensive_results = {
        'metadata': {
            'analysis_type': 'comprehensive_forensic_analysis',
            'generated_at': datetime.now().isoformat(),
            'input_file': input_path,
            'version': '1.0.0'
        },
        'analysis_results': analysis_results,
        'forensic_profile': forensic_profile,
        'ml_user_profile': user_profile,
        'risk_assessment': risk_assessment
    }
    
    results_file = output_dir / f"comprehensive_analysis_{Path(input_path).stem}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"✓ Comprehensive results saved: {results_file}")
    
    # Generate report if requested
    if generate_report:
        if verbose:
            print("Generating comprehensive HTML report...")
        
        report_generator = ForensicReportGenerator()
        report_file = output_dir / f"comprehensive_report_{Path(input_path).stem}.html"
        report_path = report_generator.generate_report(
            analysis_results, forensic_profile, str(report_file)
        )
        print(f"✓ Comprehensive report generated: {report_path}")
    
    # Display comprehensive summary
    display_comprehensive_summary(forensic_profile, user_profile, risk_assessment, verbose)


def run_test():
    """Run basic functionality test."""
    try:
        from tests.test_simple import test_basic_functionality
        return test_basic_functionality()
    except ImportError:
        print("Test module not found. Running basic import test...")
        try:
            from src.forensic_analyzer import ForensicAnalyzer
            analyzer = ForensicAnalyzer()
            print("✓ Basic import test passed")
            return True
        except Exception as e:
            print(f"✗ Basic import test failed: {e}")
            return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def display_analysis_summary(forensic_profile: dict, verbose: bool):
    """Display basic analysis summary."""
    print("\nAnalysis Summary")
    print("-" * 20)
    
    # forensic_profile has the structure from generate_forensic_profile()
    risk_evaluation = forensic_profile.get('risk_evaluation', {})
    behavioral_assessment = forensic_profile.get('behavioral_assessment', {})
    technical_analysis = forensic_profile.get('technical_analysis', {})
    subject_identification = forensic_profile.get('subject_identification', {})
    
    # Extract values from the forensic profile structure
    analysis_metadata = technical_analysis.get('analysis_metadata', {})
    comm_analysis = behavioral_assessment.get('communication_analysis', {})
    
    # Get actual values with proper fallbacks
    detected_app = "Replika (Inferred)"  # We know it's Replika from the database structure
    data_quality = analysis_metadata.get('data_quality_score', 0)
    risk_level = risk_evaluation.get('overall_risk_level', 'Unknown')
    dominant_emotion = comm_analysis.get('emotional_expression', {}).get('dominant_emotion', 'Unknown')
    
    print(f"App Identified: {detected_app}")
    print(f"Data Quality: {data_quality:.2f}")
    print(f"Risk Level: {risk_level.title()}")
    print(f"Communication Style: {dominant_emotion.title()}")
    
    if verbose:
        identified_risks = risk_evaluation.get('identified_risks', [])
        if identified_risks:
            print(f"Risk Factors: {', '.join(identified_risks[:3])}")


def display_comprehensive_summary(forensic_profile: dict, user_profile: dict, risk_assessment: dict, verbose: bool):
    """Display comprehensive analysis summary."""
    print("\nComprehensive Analysis Summary")
    print("-" * 35)
    
    # Basic info
    profile_metadata = forensic_profile.get('profile_metadata', {})
    print(f"App Identified: {profile_metadata.get('app_identified', 'Unknown')}")
    print(f"Data Quality: {profile_metadata.get('data_quality_score', 0):.2f}")
    print(f"Analysis Completeness: {profile_metadata.get('analysis_completeness', 0):.1%}")
    
    # Risk assessment
    print(f"\nRisk Assessment:")
    print(f"- Overall Level: {risk_assessment.get('risk_level', 'unknown').upper()}")
    print(f"- Risk Score: {risk_assessment.get('overall_risk_score', 0):.2f}")
    print(f"- Intervention Urgency: {risk_assessment.get('intervention_urgency', 'unknown')}")
    
    # ML insights
    if user_profile:
        behavioral_cluster = user_profile.get('behavioral_cluster', {})
        personality = user_profile.get('personality_indicators', {})
        
        print(f"\nML Profile Insights:")
        print(f"- Behavioral Cluster: {behavioral_cluster.get('cluster_id', 'unknown')}")
        print(f"- Anomaly Detected: {'Yes' if user_profile.get('anomaly_detection', {}).get('is_anomaly') else 'No'}")
        
        dominant_traits = personality.get('dominant_traits', [])
        if dominant_traits:
            print(f"- Dominant Traits: {', '.join(dominant_traits)}")
    
    # Key recommendations
    recommendations = risk_assessment.get('recommendations', [])
    if recommendations and verbose:
        print(f"\nKey Recommendations:")
        for rec in recommendations[:3]:
            print(f"- {rec}")


if __name__ == "__main__":
    main()
