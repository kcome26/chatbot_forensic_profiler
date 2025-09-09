"""
SQLite database parser for AI girlfriend applications.
"""
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from .base_parser import DatabaseParser


class SQLiteParser(DatabaseParser):
    """Parser for SQLite databases commonly used by mobile apps."""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.connection = None
        
    def connect(self) -> bool:
        """Establish connection to SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            print(f"Failed to connect to SQLite database: {e}")
            return False
    
    def discover_schema(self) -> Dict[str, Any]:
        """Discover SQLite database schema."""
        if not self.connection:
            raise RuntimeError("Database not connected")
            
        schema = {}
        cursor = self.connection.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table_name, in tables:
            # Get column information for each table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            table_schema = {
                'columns': {},
                'row_count': 0,
                'indexes': []
            }
            
            # Process column information
            for col_info in columns:
                col_name = col_info[1]
                col_type = col_info[2]
                is_nullable = not col_info[3]
                is_primary_key = col_info[5]
                
                table_schema['columns'][col_name] = {
                    'type': col_type,
                    'nullable': is_nullable,
                    'primary_key': is_primary_key
                }
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            table_schema['row_count'] = cursor.fetchone()[0]
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            table_schema['indexes'] = [idx[1] for idx in indexes]
            
            schema[table_name] = table_schema
            
        self.schema = schema
        return schema
    
    def extract_tables(self) -> Dict[str, pd.DataFrame]:
        """Extract all tables as pandas DataFrames."""
        if not self.connection:
            raise RuntimeError("Database not connected")
            
        tables = {}
        
        for table_name in self.schema.keys():
            try:
                # Extract table data
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.connection)
                
                # Classify columns
                classifications = self.classify_columns(df)
                
                # Store classification metadata
                df.attrs['column_classifications'] = classifications
                df.attrs['table_name'] = table_name
                
                tables[table_name] = df
                
            except Exception as e:
                print(f"Failed to extract table {table_name}: {e}")
                continue
                
        return tables
    
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract database metadata."""
        if not self.connection:
            raise RuntimeError("Database not connected")
            
        metadata = {
            'db_path': self.db_path,
            'db_type': 'SQLite',
            'extraction_time': datetime.now().isoformat(),
            'tables': list(self.schema.keys()),
            'total_tables': len(self.schema),
            'total_rows': sum(table['row_count'] for table in self.schema.values()),
            'app_info': {}
        }
        
        # Try to detect app-specific information
        metadata['app_info'] = self._detect_app_info()
        
        # Get database file information
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA database_list")
        db_info = cursor.fetchall()
        
        if db_info:
            metadata['database_info'] = {
                'name': db_info[0][1],
                'file': db_info[0][2]
            }
        
        self.metadata = metadata
        return metadata
    
    def _detect_app_info(self) -> Dict[str, Any]:
        """Attempt to detect which AI girlfriend app this database belongs to."""
        app_info = {
            'detected_app': 'unknown',
            'confidence': 0.0,
            'indicators': []
        }
        
        table_names = list(self.schema.keys())
        table_names_lower = [name.lower() for name in table_names]
        
        # App-specific table patterns
        app_patterns = {
            'replika': ['conversations', 'user_profile', 'avatars', 'memories'],
            'character_ai': ['characters', 'chats', 'messages', 'personas'],
            'anima': ['conversations', 'personality', 'mood', 'relationships'],
            'kuki': ['chats', 'user_data', 'responses', 'learning'],
            'mitsuku': ['conversations', 'user_info', 'topics', 'responses']
        }
        
        best_match = None
        best_score = 0
        
        for app_name, patterns in app_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if any(pattern in table_name for table_name in table_names_lower):
                    score += 1
                    matched_patterns.append(pattern)
            
            confidence = score / len(patterns)
            
            if confidence > best_score:
                best_score = confidence
                best_match = app_name
                app_info['indicators'] = matched_patterns
        
        if best_match and best_score > 0.3:  # Minimum confidence threshold
            app_info['detected_app'] = best_match
            app_info['confidence'] = best_score
        
        return app_info
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
