"""
Advanced usage example with custom analysis and profiling.
"""
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.forensic_analyzer import ForensicAnalyzer
from src.profiling.user_profiler import UserProfiler
from src.profiling.risk_assessment import RiskAssessment
from src.reporting.report_generator import ForensicReportGenerator


def advanced_analysis_example(db_path: str):
    """
    Advanced analysis example with detailed profiling.
    
    Args:
        db_path: Path to the database file
    """
    print("AI Girlfriend Forensics Analyzer - Advanced Example")
    print("=" * 60)
    
    try:
        # Initialize components
        analyzer = ForensicAnalyzer()
        profiler = UserProfiler()
        risk_assessor = RiskAssessment()
        
        # Load and analyze database
        print("1. Loading and analyzing database...")
        if not analyzer.load_database(db_path):
            print("Failed to load database")
            return
        
        analysis_results = analyzer.analyze_database()
        print("   ✓ Basic analysis complete")
        
        # Advanced ML-based profiling
        print("2. Performing advanced ML profiling...")
        
        # Extract features for ML processing
        features_df = profiler.extract_features(analysis_results)
        print(f"   ✓ Extracted {len(features_df.columns)} features")
        
        # Create detailed user profile
        user_profile = profiler.create_user_profile(features_df, user_id="subject_001")
        print("   ✓ Generated ML-based user profile")
        
        # Perform comprehensive risk assessment
        print("3. Conducting comprehensive risk assessment...")
        risk_assessment = risk_assessor.assess_comprehensive_risk(analysis_results)
        print(f"   ✓ Risk level: {risk_assessment.get('risk_level', 'unknown')}")
        
        # Generate forensic profile
        print("4. Generating comprehensive forensic profile...")
        forensic_profile = analyzer.generate_forensic_profile()
        
        # Enhance profile with ML insights
        forensic_profile['ml_profile'] = user_profile
        forensic_profile['comprehensive_risk'] = risk_assessment
        
        # Generate detailed report
        print("5. Generating comprehensive report...")
        report_generator = ForensicReportGenerator()
        report_path = report_generator.generate_report(analysis_results, forensic_profile)
        
        # Save all results
        results_path = save_comprehensive_results(
            analysis_results, forensic_profile, user_profile, risk_assessment
        )
        
        print("\n" + "=" * 60)
        print("ADVANCED ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Display comprehensive summary
        display_comprehensive_summary(forensic_profile, user_profile, risk_assessment)
        
        print(f"\nGenerated Files:")
        print(f"- HTML Report: {report_path}")
        print(f"- Comprehensive Results: {results_path}")
        
    except Exception as e:
        print(f"Error during advanced analysis: {e}")
        import traceback
        traceback.print_exc()


def save_comprehensive_results(analysis_results, forensic_profile, user_profile, risk_assessment):
    """Save comprehensive results to JSON file."""
    from datetime import datetime
    
    comprehensive_results = {
        'metadata': {
            'analysis_type': 'comprehensive_forensic_analysis',
            'generated_at': datetime.now().isoformat(),
            'version': '1.0.0'
        },
        'analysis_results': analysis_results,
        'forensic_profile': forensic_profile,
        'ml_user_profile': user_profile,
        'risk_assessment': risk_assessment
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_analysis_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    return filename


def display_comprehensive_summary(forensic_profile, user_profile, risk_assessment):
    """Display comprehensive analysis summary."""
    print("\nCOMPREHENSIVE ANALYSIS SUMMARY")
    print("-" * 40)
    
    # Basic profile info
    profile_metadata = forensic_profile.get('profile_metadata', {})
    print(f"App Identified: {profile_metadata.get('app_identified', 'Unknown')}")
    print(f"Data Quality: {profile_metadata.get('data_quality_score', 0):.2f}")
    print(f"Analysis Completeness: {profile_metadata.get('analysis_completeness', 0):.1%}")
    
    # Risk assessment
    print(f"\nRisk Assessment:")
    print(f"- Overall Risk Level: {risk_assessment.get('risk_level', 'unknown').upper()}")
    print(f"- Risk Score: {risk_assessment.get('overall_risk_score', 0):.2f}")
    print(f"- Intervention Urgency: {risk_assessment.get('intervention_urgency', 'unknown')}")
    
    # Risk categories
    risk_categories = risk_assessment.get('risk_categories', {})
    print(f"\nRisk Categories:")
    for category, data in risk_categories.items():
        risk_level = data.get('risk_level', 'unknown')
        risk_score = data.get('risk_score', 0)
        print(f"- {category.replace('_', ' ').title()}: {risk_level} ({risk_score:.2f})")
    
    # ML Profile insights
    ml_profile = user_profile
    behavioral_cluster = ml_profile.get('behavioral_cluster', {})
    personality = ml_profile.get('personality_indicators', {})
    
    print(f"\nML Profile Insights:")
    print(f"- Behavioral Cluster: {behavioral_cluster.get('cluster_id', 'unknown')}")
    print(f"- Anomaly Detection: {'Yes' if ml_profile.get('anomaly_detection', {}).get('is_anomaly') else 'No'}")
    
    # Personality traits
    dominant_traits = personality.get('dominant_traits', [])
    if dominant_traits:
        print(f"- Dominant Traits: {', '.join(dominant_traits)}")
    
    # Risk factors
    specific_risks = risk_assessment.get('specific_risks', {})
    immediate_concerns = specific_risks.get('immediate_concerns', [])
    if immediate_concerns:
        print(f"\nImmediate Concerns:")
        for concern in immediate_concerns[:3]:  # Show top 3
            print(f"- {concern}")
    
    # Recommendations
    recommendations = risk_assessment.get('recommendations', [])
    if recommendations:
        print(f"\nKey Recommendations:")
        for rec in recommendations[:3]:  # Show top 3
            print(f"- {rec}")


def compare_profiles_example():
    """Example of comparing multiple user profiles."""
    print("\nProfile Comparison Feature")
    print("-" * 30)
    print("This feature allows comparison of behavioral patterns between different users or time periods.")
    print("Usage: profiler.compare_profiles(profile1, profile2)")
    print("\nComparison metrics include:")
    print("- Behavioral similarity")
    print("- Risk level comparison") 
    print("- Personality trait alignment")
    print("- Communication style differences")


def batch_analysis_example():
    """Example of batch processing multiple databases."""
    print("\nBatch Analysis Feature")
    print("-" * 25)
    print("For processing multiple databases:")
    print("""
# Example batch processing code
databases = ['db1.sqlite', 'db2.json', 'db3.db']
results = []

for db_path in databases:
    analyzer = ForensicAnalyzer()
    if analyzer.load_database(db_path):
        analysis = analyzer.analyze_database()
        profile = analyzer.generate_forensic_profile()
        results.append((db_path, analysis, profile))

# Compare results across databases
# Generate comparative reports
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        advanced_analysis_example(db_path)
    else:
        print("Advanced AI Girlfriend Forensics Analyzer Examples")
        print("=" * 50)
        print("\nUsage: python advanced_usage.py <database_path>")
        print("\nFeatures demonstrated:")
        print("- ML-based user profiling")
        print("- Comprehensive risk assessment")
        print("- Advanced behavioral clustering")
        print("- Anomaly detection")
        print("- Personality analysis")
        print("- Detailed reporting")
        
        compare_profiles_example()
        batch_analysis_example()
