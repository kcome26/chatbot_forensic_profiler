"""
SQLite database extractor for chat messages.

This module provides comprehensive SQLite database analysis and message extraction
specifically designed for AI companion chatbot applications.
"""

import sqlite3
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class SQLiteExtractor:
    """Extract data from SQLite databases with forensic integrity."""
    
    def __init__(self):
        self.connection = None
        self.db_path = None
        self.schema_info = {}
        self.extraction_metadata = {}
        
    def load_database(self, db_path: str) -> bool:
        """
        Load SQLite database with forensic validation.
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            return False
        
        try:
            # Validate SQLite file header
            if not self._validate_sqlite_header(db_path):
                logger.error(f"Invalid SQLite file: {db_path}")
                return False
            
            # Connect to database in read-only mode
            self.connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            self.db_path = db_path
            
            # Initialize extraction metadata
            self.extraction_metadata = {
                'database_path': db_path,
                'file_size': os.path.getsize(db_path),
                'file_hash_md5': self._calculate_file_hash(db_path, 'md5'),
                'file_hash_sha256': self._calculate_file_hash(db_path, 'sha256'),
                'extraction_timestamp': datetime.now().isoformat(),
                'sqlite_version': self._get_sqlite_version()
            }
            
            # Discover database schema
            self.schema_info = self._discover_schema()
            
            logger.info(f"Successfully loaded SQLite database: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load database {db_path}: {e}")
            return False
    
    def extract_all_data(self) -> Dict[str, Any]:
        """
        Extract all data from the database with comprehensive analysis.
        
        Returns:
            Dictionary containing all extracted data and metadata
        """
        if not self.connection:
            raise RuntimeError("Database not loaded. Call load_database() first.")
        
        extraction_results = {
            'metadata': self.extraction_metadata.copy(),
            'schema': self.schema_info,
            'tables': {},
            'messages': [],
            'analysis': {
                'message_tables_found': 0,
                'total_messages': 0,
                'date_range': None,
                'user_activity': {},
                'conversation_partners': []
            }
        }
        
        # Extract data from each table
        for table_name, table_info in self.schema_info.items():
            try:
                table_data = self._extract_table_data(table_name, table_info)
                extraction_results['tables'][table_name] = table_data
                
                # Check if this table contains messages
                if self._is_message_table(table_name, table_info):
                    messages = self._extract_messages_from_table(table_name, table_info)
                    extraction_results['messages'].extend(messages)
                    extraction_results['analysis']['message_tables_found'] += 1
                    
            except Exception as e:
                logger.error(f"Error extracting from table {table_name}: {e}")
                extraction_results['tables'][table_name] = {'error': str(e)}
        
        # Perform message analysis
        if extraction_results['messages']:
            extraction_results['analysis'] = self._analyze_messages(extraction_results['messages'])
        
        extraction_results['analysis']['total_messages'] = len(extraction_results['messages'])
        extraction_results['metadata']['extraction_completed'] = datetime.now().isoformat()
        
        return extraction_results
    
    def extract_messages(self, db_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract messages from SQLite database (convenience method).
        
        Args:
            db_path: Path to SQLite database
            
        Returns:
            Dictionary containing extracted messages and metadata
        """
        if not self.load_database(db_path):
            return None
        
        return self.extract_all_data()
    
    def _validate_sqlite_header(self, db_path: str) -> bool:
        """Validate SQLite file header."""
        try:
            with open(db_path, 'rb') as f:
                header = f.read(16)
                return header.startswith(b'SQLite format 3\x00')
        except Exception:
            return False
    
    def _calculate_file_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        """Calculate file hash for forensic integrity."""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating {algorithm} hash: {e}")
            return "error"
    
    def _get_sqlite_version(self) -> str:
        """Get SQLite version information."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT sqlite_version()")
            return cursor.fetchone()[0]
        except Exception:
            return "unknown"
    
    def _discover_schema(self) -> Dict[str, Dict]:
        """Discover complete database schema."""
        schema = {}
        cursor = self.connection.cursor()
        
        try:
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = cursor.fetchall()
            
            for table_row in tables:
                table_name = table_row[0]
                
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get indexes
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = cursor.fetchall()
                
                schema[table_name] = {
                    'columns': {col[1]: {
                        'type': col[2],
                        'not_null': bool(col[3]),
                        'default': col[4],
                        'primary_key': bool(col[5])
                    } for col in columns},
                    'row_count': row_count,
                    'indexes': [idx[1] for idx in indexes],
                    'column_names': [col[1] for col in columns]
                }
                
        except Exception as e:
            logger.error(f"Error discovering schema: {e}")
        
        return schema
    
    def _extract_table_data(self, table_name: str, table_info: Dict) -> Dict[str, Any]:
        """Extract all data from a specific table."""
        cursor = self.connection.cursor()
        
        try:
            # Get all data
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                row_dict = {}
                for key in row.keys():
                    row_dict[key] = row[key]
                data.append(row_dict)
            
            return {
                'data': data,
                'row_count': len(data),
                'columns': table_info['column_names'],
                'extraction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting table {table_name}: {e}")
            return {'error': str(e)}
    
    def _is_message_table(self, table_name: str, table_info: Dict) -> bool:
        """Determine if a table likely contains chat messages."""
        table_name_lower = table_name.lower()
        columns_lower = [col.lower() for col in table_info['column_names']]
        
        # Check for message-related table names
        message_table_keywords = [
            'message', 'chat', 'conversation', 'msg', 'text', 'content',
            'reply', 'response', 'dialogue', 'talk', 'communication'
        ]
        
        if any(keyword in table_name_lower for keyword in message_table_keywords):
            return True
        
        # Check for message-related columns
        message_column_keywords = [
            'message', 'text', 'content', 'body', 'msg', 'reply',
            'response', 'dialogue', 'communication', 'chat_text'
        ]
        
        has_message_column = any(keyword in col for col in columns_lower for keyword in message_column_keywords)
        
        # Check for timestamp columns
        time_column_keywords = [
            'timestamp', 'time', 'created', 'sent', 'date', 'datetime',
            'created_at', 'sent_at', 'message_time'
        ]
        
        has_time_column = any(keyword in col for col in columns_lower for keyword in time_column_keywords)
        
        # Consider it a message table if it has both message content and timestamp
        # or if it has message-related keywords and reasonable number of columns
        return (has_message_column and has_time_column) or \
               (has_message_column and len(columns_lower) >= 3 and table_info['row_count'] > 0)
    
    def _extract_messages_from_table(self, table_name: str, table_info: Dict) -> List[Dict]:
        """Extract and parse messages from a message table."""
        cursor = self.connection.cursor()
        messages = []
        
        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            for row in rows:
                message = {}
                
                # Convert row to dictionary
                for key in row.keys():
                    message[key] = row[key]
                
                # Add forensic metadata
                message['_forensic_metadata'] = {
                    'source_table': table_name,
                    'source_database': self.db_path,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'record_hash': hashlib.sha256(str(dict(row)).encode()).hexdigest()
                }
                
                # Try to standardize common fields
                message = self._standardize_message_fields(message)
                
                messages.append(message)
                
        except Exception as e:
            logger.error(f"Error extracting messages from {table_name}: {e}")
        
        return messages
    
    def _standardize_message_fields(self, message: Dict) -> Dict:
        """Standardize message field names for consistent analysis."""
        # Common field mappings
        field_mappings = {
            'text': ['message', 'content', 'body', 'msg', 'text_content', 'message_text'],
            'timestamp': ['time', 'created', 'sent', 'date', 'datetime', 'created_at', 'sent_at', 'message_time'],
            'user_id': ['user', 'sender', 'author', 'from_user', 'user_name'],
            'conversation_id': ['chat_id', 'thread_id', 'conversation', 'room_id']
        }
        
        # Create standardized fields
        for standard_field, possible_fields in field_mappings.items():
            for field in possible_fields:
                if field in message and standard_field not in message:
                    message[f'_std_{standard_field}'] = message[field]
                    break
        
        return message
    
    def _analyze_messages(self, messages: List[Dict]) -> Dict[str, Any]:
        """Perform basic analysis on extracted messages."""
        if not messages:
            return {}
        
        analysis = {
            'total_messages': len(messages),
            'tables_with_messages': len(set(msg.get('_forensic_metadata', {}).get('source_table', 'unknown') for msg in messages)),
            'date_range': None,
            'user_activity': {},
            'message_length_stats': {},
            'temporal_patterns': {}
        }
        
        # Analyze timestamps if available
        timestamps = []
        for msg in messages:
            # Look for timestamp fields
            for field in ['timestamp', '_std_timestamp', 'time', 'created', 'sent', 'date']:
                if field in msg and msg[field]:
                    try:
                        # Handle different timestamp formats
                        ts = msg[field]
                        if isinstance(ts, (int, float)):
                            # Unix timestamp
                            if ts > 1e12:  # milliseconds
                                ts = ts / 1000
                            timestamps.append(datetime.fromtimestamp(ts))
                        elif isinstance(ts, str):
                            # Try to parse string timestamp
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S']:
                                try:
                                    timestamps.append(datetime.strptime(ts, fmt))
                                    break
                                except ValueError:
                                    continue
                        break
                    except Exception:
                        continue
        
        if timestamps:
            analysis['date_range'] = {
                'earliest': min(timestamps).isoformat(),
                'latest': max(timestamps).isoformat(),
                'span_days': (max(timestamps) - min(timestamps)).days
            }
        
        # Analyze message lengths
        message_texts = []
        for msg in messages:
            for field in ['text', '_std_text', 'message', 'content', 'body']:
                if field in msg and msg[field]:
                    text = str(msg[field])
                    message_texts.append(text)
                    break
        
        if message_texts:
            lengths = [len(text) for text in message_texts]
            analysis['message_length_stats'] = {
                'min_length': min(lengths),
                'max_length': max(lengths),
                'avg_length': sum(lengths) / len(lengths),
                'total_characters': sum(lengths)
            }
        
        return analysis
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
