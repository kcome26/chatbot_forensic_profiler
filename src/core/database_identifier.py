"""
Forensic Chat Analyzer - Database Identifier Module

This module handles the identification and discovery of AI companion chatbot 
databases within filesystem images.
"""

import os
import pathlib
import fnmatch
import logging
import sqlite3
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DatabaseIdentifier')

class DatabaseIdentifier:
    """
    Identifies and catalogs AI companion chatbot databases in filesystem images.
    
    This class searches for database files based on known application patterns
    and validates them to ensure they are legitimate database files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DatabaseIdentifier.
        
        Args:
            config_path: Path to configuration file with app patterns
        """
        if config_path and os.path.exists(config_path):
            self.target_apps = self._load_config(config_path)
        else:
            # Default patterns for known AI companion chatbot databases
            self.target_apps = {
                "Replika": [
                    "**/com.replika.app/databases/*.db",
                    "**/Replika/databases/*.sqlite*",
                    "**/replika.*/databases/*.db"
                ],
                "Character_AI": [
                    "**/character.ai/Local Storage/leveldb/*.ldb",
                    "**/character.ai/IndexedDB/*.sqlite",
                    "**/com.characterai.app/databases/*.db"
                ],
                "Microsoft_Copilot": [
                    "**/Microsoft/Copilot/Local Storage/leveldb/*.ldb",
                    "**/Microsoft/BingChat/*.sqlite",
                    "**/copilot*/databases/*.db"
                ],
                "ChatGPT": [
                    "**/com.openai.chat/databases/*.db",
                    "**/chat.openai.com/Local Storage/leveldb/*.ldb",
                    "**/openai*/databases/*.sqlite*"
                ],
                "Claude": [
                    "**/anthropic.claude/databases/*.db",
                    "**/claude.ai/Local Storage/leveldb/*.ldb",
                    "**/com.anthropic.claude/databases/*.db"
                ],
                "Snapchat_AI": [
                    "**/com.snapchat.android/databases/*.db",
                    "**/Snapchat/databases/*.sqlite*"
                ],
                "Discord": [
                    "**/com.discord/databases/*.db",
                    "**/Discord/Local Storage/leveldb/*.ldb"
                ],
                "Telegram": [
                    "**/org.telegram.messenger/databases/*.db",
                    "**/Telegram*/databases/*.sqlite*"
                ]
            }
        
        self.scan_results = {}
        self.total_files_scanned = 0
        self.total_databases_found = 0
        
    def _load_config(self, config_path: str) -> Dict[str, List[str]]:
        """Load app patterns from configuration file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def identify_databases(self, root_path: str, output_file: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Scan the filesystem for AI companion chatbot databases.
        
        Args:
            root_path: Path to the filesystem image or directory to scan
            output_file: Optional path to save results as JSON
            
        Returns:
            Dictionary of app names with lists of found database files and their metadata
        """
        logger.info(f"Starting database identification scan on: {root_path}")
        start_time = datetime.now()
        
        if not os.path.exists(root_path):
            logger.error(f"Path does not exist: {root_path}")
            return {}
        
        results = {}
        self.total_files_scanned = 0
        self.total_databases_found = 0
        
        # For each app and its patterns
        for app_name, patterns in self.target_apps.items():
            logger.info(f"Searching for {app_name} databases...")
            results[app_name] = []
            
            for pattern in patterns:
                logger.debug(f"Searching with pattern: {pattern}")
                
                try:
                    # Use pathlib for recursive globbing
                    root = pathlib.Path(root_path)
                    for file_path in root.glob(pattern):
                        self.total_files_scanned += 1
                        
                        if self._is_likely_database(file_path):
                            file_info = self._extract_file_metadata(file_path)
                            results[app_name].append(file_info)
                            self.total_databases_found += 1
                            logger.info(f"Found {app_name} database: {file_path}")
                            
                except Exception as e:
                    logger.error(f"Error scanning pattern {pattern}: {e}")
                    continue
        
        # Calculate scan statistics
        end_time = datetime.now()
        scan_duration = (end_time - start_time).total_seconds()
        
        # Add metadata to results
        scan_metadata = {
            "scan_start": start_time.isoformat(),
            "scan_end": end_time.isoformat(),
            "scan_duration_seconds": scan_duration,
            "total_files_scanned": self.total_files_scanned,
            "total_databases_found": self.total_databases_found,
            "root_path": root_path
        }
        
        final_results = {
            "metadata": scan_metadata,
            "databases": results
        }
        
        # Save results if output file specified
        if output_file:
            self._save_results(final_results, output_file)
        
        # Log summary
        logger.info(f"Database identification complete.")
        logger.info(f"Scanned {self.total_files_scanned} files in {scan_duration:.2f} seconds")
        logger.info(f"Found {self.total_databases_found} potential database files")
        
        return final_results
    
    def _extract_file_metadata(self, file_path: pathlib.Path) -> Dict:
        """Extract comprehensive metadata from a database file."""
        try:
            stat = file_path.stat()
            
            file_info = {
                "path": str(file_path),
                "filename": file_path.name,
                "size": stat.st_size,
                "size_human": self._human_readable_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "type": self._detect_db_type(file_path),
                "hash_md5": self._calculate_file_hash(file_path, "md5"),
                "hash_sha256": self._calculate_file_hash(file_path, "sha256")
            }
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return {
                "path": str(file_path),
                "error": str(e)
            }
    
    def _is_likely_database(self, file_path: pathlib.Path) -> bool:
        """
        Check if a file is likely to be a database based on extension, size, and content.
        """
        # Common database extensions
        db_extensions = ['.db', '.sqlite', '.sqlite3', '.ldb', '.mdb', '.json', '.realm']
        
        # Check if file exists and is accessible
        if not file_path.exists() or not file_path.is_file():
            return False
        
        # Check extension
        if not any(str(file_path).lower().endswith(ext) for ext in db_extensions):
            return False
            
        try:
            # Check if file is too small to be a useful database (1KB minimum)
            if file_path.stat().st_size < 1024:
                return False
        except (OSError, PermissionError):
            logger.warning(f"Cannot access file stats for: {file_path}")
            return False
            
        # Additional content validation for SQLite files
        if str(file_path).lower().endswith(('.db', '.sqlite', '.sqlite3')):
            return self._validate_sqlite_file(file_path)
            
        return True
    
    def _validate_sqlite_file(self, file_path: pathlib.Path) -> bool:
        """Validate that a file is actually a SQLite database."""
        try:
            # Check SQLite header
            with open(file_path, 'rb') as f:
                header = f.read(16)
                if header.startswith(b'SQLite format 3\x00'):
                    return True
        except (OSError, PermissionError):
            logger.warning(f"Cannot read file header for: {file_path}")
            
        return False
    
    def _detect_db_type(self, file_path: pathlib.Path) -> str:
        """
        Attempt to detect the type of database.
        """
        file_str = str(file_path).lower()
        
        # SQLite detection
        if file_str.endswith(('.db', '.sqlite', '.sqlite3')):
            if self._validate_sqlite_file(file_path):
                return "SQLite"
        
        # LevelDB detection
        if file_str.endswith('.ldb'):
            return "LevelDB"
            
        # JSON detection
        if file_str.endswith('.json'):
            return "JSON"
            
        # Realm detection
        if file_str.endswith('.realm'):
            return "Realm"
            
        # Default to unknown with extension
        extension = file_path.suffix
        return f"Unknown ({extension})"
    
    def _human_readable_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _calculate_file_hash(self, file_path: pathlib.Path, algorithm: str = "sha256") -> str:
        """Calculate hash of file for integrity verification."""
        import hashlib
        
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating {algorithm} hash for {file_path}: {e}")
            return "error"
    
    def _save_results(self, results: Dict, output_file: str):
        """Save scan results to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")

def main():
    """Command line interface for database identification."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Identify AI companion chatbot databases in filesystem images"
    )
    parser.add_argument("path", help="Path to scan for databases")
    parser.add_argument("--output", "-o", 
                       help="Output JSON file for results", 
                       default="output/databases_identified.json")
    parser.add_argument("--config", "-c",
                       help="Configuration file with app patterns",
                       default="config/app_patterns.json")
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    identifier = DatabaseIdentifier(args.config)
    results = identifier.identify_databases(args.path, args.output)
    
    # Print summary
    print("\n=== Database Identification Summary ===")
    print(f"Scan Path: {args.path}")
    print(f"Total Files Scanned: {results['metadata']['total_files_scanned']}")
    print(f"Total Databases Found: {results['metadata']['total_databases_found']}")
    print(f"Scan Duration: {results['metadata']['scan_duration_seconds']:.2f} seconds")
    print(f"Results saved to: {args.output}")
    
    print("\n=== Databases by Application ===")
    for app, dbs in results['databases'].items():
        if dbs:
            print(f"- {app}: {len(dbs)} database(s) found")
            for db in dbs:
                print(f"  • {db['filename']} ({db['size_human']}) - {db['type']}")

if __name__ == "__main__":
    main()
