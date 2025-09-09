"""
Main entry point for the Forensic Chat Analyzer tool.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.database_identifier import DatabaseIdentifier

def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('forensic_chat_analyzer.log')
        ]
    )

def main():
    """Main entry point for the Forensic Chat Analyzer."""
    parser = argparse.ArgumentParser(
        description="Forensic Chat Analyzer - Digital forensics tool for AI companion chatbot databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py identify /path/to/filesystem/image
  python main.py identify C:\\forensic_image --output results.json --verbose
  
Legal Notice:
  This tool is for legitimate forensic investigations only.
  Ensure proper authorization before use.
        """
    )
    
    # Global arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config", "-c", help="Configuration directory path", default="config")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Database identification command
    identify_parser = subparsers.add_parser("identify", help="Identify databases in filesystem image")
    identify_parser.add_argument("path", help="Path to filesystem image (ZIP file or directory) to scan")
    identify_parser.add_argument("--output", "-o", help="Output JSON file", default="output/databases_identified.json")
    
    # Data extraction command
    extract_parser = subparsers.add_parser("extract", help="Extract data from identified databases")
    extract_parser.add_argument("input", help="Path to scan results JSON file or specific database file")
    extract_parser.add_argument("--output", "-o", help="Output JSON file", default="output/extracted_data.json")
    extract_parser.add_argument("--database", "-d", help="Extract from specific database file (skip scan results)")
    extract_parser.add_argument("--type", "-t", help="Database type (SQLite, JSON, LevelDB)", default="SQLite")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == "identify":
            return identify_databases(args, logger)
        elif args.command == "extract":
            return extract_data(args, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1

def identify_databases(args, logger):
    """Execute database identification command."""
    logger.info("Starting Forensic Chat Analyzer - Database Identification")
    logger.info(f"Scanning input: {args.path}")
    
    # Validate input path
    if not os.path.exists(args.path):
        logger.error(f"Input path does not exist: {args.path}")
        return 1
    
    # Check if input is a ZIP file or directory
    import zipfile
    if zipfile.is_zipfile(args.path):
        logger.info(f"Input detected as ZIP file: {args.path}")
    elif os.path.isdir(args.path):
        logger.info(f"Input detected as directory: {args.path}")
    else:
        logger.error(f"Input must be a directory or ZIP file: {args.path}")
        return 1

def extract_data(args, logger):
    """Execute data extraction command."""
    logger.info("Starting Forensic Chat Analyzer - Data Extraction")
    
    # Import data extraction modules
    from src.core.data_extractor import DataExtractor
    from src.extractors.sqlite_extractor import SQLiteExtractor
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if extracting from specific database or scan results
    if args.database:
        logger.info(f"Extracting from specific database: {args.database}")
        
        if not os.path.exists(args.database):
            logger.error(f"Database file not found: {args.database}")
            return 1
        
        # Extract from specific database
        if args.type.lower() == 'sqlite':
            with SQLiteExtractor() as extractor:
                results = extractor.extract_messages(args.database)
        else:
            # Use general data extractor
            extractor = DataExtractor()
            results = extractor.extract_messages(args.database, args.type)
        
        if not results:
            logger.error("Failed to extract data from database")
            return 1
            
    else:
        logger.info(f"Extracting from scan results: {args.input}")
        
        if not os.path.exists(args.input):
            logger.error(f"Scan results file not found: {args.input}")
            return 1
        
        # Extract from scan results
        extractor = DataExtractor()
        results = extractor.extract_from_scan_results(args.input)
        
        if not results:
            logger.error("Failed to extract data from scan results")
            return 1
    
    # Save results
    try:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Extraction results saved to: {args.output}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("FORENSIC CHAT ANALYZER - EXTRACTION RESULTS")
    print("="*60)
    
    if args.database:
        print(f"Database: {args.database}")
        print(f"Type: {args.type}")
        if 'messages' in results:
            print(f"Messages Extracted: {len(results['messages'])}")
        if 'analysis' in results:
            analysis = results['analysis']
            if 'total_messages' in analysis:
                print(f"Total Messages: {analysis['total_messages']}")
            if 'date_range' in analysis and analysis['date_range']:
                print(f"Date Range: {analysis['date_range']['earliest']} to {analysis['date_range']['latest']}")
    else:
        metadata = results.get('metadata', {})
        print(f"Scan Results: {args.input}")
        print(f"Databases Processed: {metadata.get('databases_processed', 0)}")
        print(f"Messages Extracted: {metadata.get('messages_extracted', 0)}")
        print(f"Errors: {len(metadata.get('errors', []))}")
    
    print(f"Results File: {args.output}")
    print("="*60)
    
    return 0
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Configure identifier
    config_file = os.path.join(args.config, "app_patterns.json")
    identifier = DatabaseIdentifier(config_file if os.path.exists(config_file) else None)
    
    # Run identification
    results = identifier.identify_databases(args.path, args.output)
    
    if not results:
        logger.error("Database identification failed")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("FORENSIC CHAT ANALYZER - IDENTIFICATION RESULTS")
    print("="*60)
    print(f"Input: {args.path}")
    print(f"Input Type: {results['metadata']['input_type']}")
    print(f"Scan Duration: {results['metadata']['scan_duration_seconds']:.2f} seconds")
    print(f"Files Scanned: {results['metadata']['total_files_scanned']:,}")
    print(f"Databases Found: {results['metadata']['total_databases_found']}")
    print(f"Results File: {args.output}")
    
    print("\nDATABASES BY APPLICATION:")
    print("-" * 40)
    
    total_size = 0
    for app, dbs in results['databases'].items():
        if dbs:
            app_size = sum(db.get('size', 0) for db in dbs if 'size' in db)
            total_size += app_size
            print(f"\n{app}: {len(dbs)} database(s)")
            
            for db in dbs:
                if 'error' not in db:
                    print(f"  • {db['filename']} ({db.get('size_human', 'Unknown size')})")
                    print(f"    Type: {db.get('type', 'Unknown')}")
                    print(f"    Modified: {db.get('modified', 'Unknown')}")
                else:
                    print(f"  • ERROR: {db.get('path', 'Unknown path')} - {db.get('error', 'Unknown error')}")
    
    if total_size > 0:
        print(f"\nTotal Data Size: {_human_readable_size(total_size)}")
    
    print("\n" + "="*60)
    print("⚠️  LEGAL REMINDER: Ensure proper authorization before proceeding with analysis")
    print("="*60)
    
    return 0

def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

if __name__ == "__main__":
    sys.exit(main())
