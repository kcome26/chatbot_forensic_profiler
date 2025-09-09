"""
Database parser factory and manager.
"""
import os
from typing import Dict, Any, Optional
from .base_parser import DatabaseParser
from .sqlite_parser import SQLiteParser
from .json_parser import JSONParser


class DatabaseParserFactory:
    """Factory for creating appropriate database parsers."""
    
    @staticmethod
    def create_parser(db_path: str) -> Optional[DatabaseParser]:
        """
        Create appropriate parser based on file extension and content.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            DatabaseParser instance or None if no suitable parser found
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        file_ext = os.path.splitext(db_path)[1].lower()
        
        # Try to determine format by extension first
        if file_ext in ['.db', '.sqlite', '.sqlite3']:
            return SQLiteParser(db_path)
        elif file_ext in ['.json']:
            return JSONParser(db_path)
        
        # If extension is not conclusive, try to detect by content
        return DatabaseParserFactory._detect_by_content(db_path)
    
    @staticmethod
    def _detect_by_content(db_path: str) -> Optional[DatabaseParser]:
        """Detect database format by examining file content."""
        try:
            with open(db_path, 'rb') as f:
                header = f.read(16)
            
            # SQLite magic number
            if header.startswith(b'SQLite format 3\x00'):
                return SQLiteParser(db_path)
            
            # Try JSON
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    f.read(100)  # Try to read some content as text
                return JSONParser(db_path)
            except UnicodeDecodeError:
                pass
            
        except Exception as e:
            print(f"Error detecting database format: {e}")
        
        return None
    
    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        """Get list of supported database formats."""
        return {
            'SQLite': 'SQLite database files (.db, .sqlite, .sqlite3)',
            'JSON': 'JSON data files (.json)',
        }


class DatabaseManager:
    """Manager for handling multiple database parsers and operations."""
    
    def __init__(self):
        self.parsers = {}
        self.current_parser = None
        
    def load_database(self, db_path: str, parser_name: Optional[str] = None) -> bool:
        """
        Load a database using appropriate parser.
        
        Args:
            db_path: Path to database file
            parser_name: Optional name to assign to this parser
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            parser = DatabaseParserFactory.create_parser(db_path)
            if parser is None:
                print(f"No suitable parser found for: {db_path}")
                return False
            
            if not parser.connect():
                print(f"Failed to connect to database: {db_path}")
                return False
            
            # Discover schema
            schema = parser.discover_schema()
            print(f"Discovered {len(schema)} tables/collections")
            
            # Store parser
            if parser_name is None:
                parser_name = os.path.basename(db_path)
            
            self.parsers[parser_name] = parser
            self.current_parser = parser
            
            return True
            
        except Exception as e:
            print(f"Error loading database {db_path}: {e}")
            return False
    
    def get_parser(self, parser_name: str) -> Optional[DatabaseParser]:
        """Get parser by name."""
        return self.parsers.get(parser_name)
    
    def list_parsers(self) -> Dict[str, str]:
        """List all loaded parsers."""
        return {name: parser.db_path for name, parser in self.parsers.items()}
    
    def extract_all_data(self, parser_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract all data from specified parser or current parser.
        
        Returns:
            Dictionary containing tables, metadata, and schema
        """
        parser = self.current_parser
        if parser_name:
            parser = self.parsers.get(parser_name)
        
        if parser is None:
            raise ValueError("No parser specified or available")
        
        return {
            'tables': parser.extract_tables(),
            'metadata': parser.extract_metadata(),
            'schema': parser.schema
        }
    
    def close_all(self):
        """Close all database connections."""
        for parser in self.parsers.values():
            if hasattr(parser, 'close'):
                parser.close()
        self.parsers.clear()
        self.current_parser = None
