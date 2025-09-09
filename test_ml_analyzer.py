#!/usr/bin/env python3
"""
Test script for ML analyzer on existing extraction results.
"""

import json
import logging
import sys
from src.analysis.ml_analyzer import MLAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ml_analyzer():
    """Test the ML analyzer with existing extraction results."""
    
    # Load existing extraction results
    extraction_file = "output/full_extraction_results.json"
    
    try:
        with open(extraction_file, 'r') as f:
            extraction_data = json.load(f)
        
        logger.info(f"Loaded extraction results from {extraction_file}")
        logger.info(f"Applications found: {list(extraction_data.get('extracted_data', {}).keys())}")
        
        # Initialize ML analyzer
        ml_analyzer = MLAnalyzer()
        
        # Run ML analysis
        logger.info("Starting ML analysis...")
        ml_results = ml_analyzer.analyze_extraction_results(extraction_data)
        
        # Save results
        output_file = "output/ml_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(ml_results, f, indent=2)
        
        logger.info(f"ML analysis completed. Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("ML ANALYSIS RESULTS SUMMARY")
        print("="*60)
        
        metadata = ml_results.get('metadata', {})
        print(f"Analysis timestamp: {metadata.get('analysis_timestamp', 'Unknown')}")
        print(f"ML capabilities available: {metadata.get('ml_capabilities', {})}")
        
        # Show analyzed applications
        analyzed_apps = []
        for analysis_type in ['content_analysis', 'behavioral_analysis', 'sentiment_analysis']:
            if analysis_type in ml_results:
                analyzed_apps.extend(ml_results[analysis_type].keys())
        
        analyzed_apps = list(set(analyzed_apps))
        print(f"Applications analyzed: {analyzed_apps}")
        
        # Show content analysis summary for each app
        if 'content_analysis' in ml_results:
            print(f"\nContent Analysis Summary:")
            for app, analysis in ml_results['content_analysis'].items():
                if 'message_statistics' in analysis:
                    stats = analysis['message_statistics']
                    print(f"  {app}:")
                    print(f"    - Total messages: {stats.get('total_messages', 0)}")
                    print(f"    - Average message length: {stats.get('avg_message_length', 0):.1f}")
                    print(f"    - Average word count: {stats.get('avg_word_count', 0):.1f}")
        
        # Show sentiment analysis summary
        if 'sentiment_analysis' in ml_results:
            print(f"\nSentiment Analysis Summary:")
            for app, analysis in ml_results['sentiment_analysis'].items():
                if 'overall_sentiment' in analysis:
                    sentiment = analysis['overall_sentiment']
                    print(f"  {app}:")
                    print(f"    - Positive: {sentiment.get('positive', 0):.3f}")
                    print(f"    - Negative: {sentiment.get('negative', 0):.3f}")
                    print(f"    - Compound: {sentiment.get('compound', 0):.3f}")
        
        print("="*60)
        
    except FileNotFoundError:
        logger.error(f"Extraction file not found: {extraction_file}")
        return 1
    except Exception as e:
        logger.error(f"Error during ML analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(test_ml_analyzer())
