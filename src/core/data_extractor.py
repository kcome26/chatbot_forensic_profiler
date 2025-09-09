"""
Data extraction module for parsing database contents.

This module handles extracting message data from identified databases using
appropriate parsers for different database types.
"""

import os
import sqlite3
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import zipfile

logger = logging.getLogger(__name__)

class DataExtractor:
    """Extract and parse data from identified databases."""
    
    def __init__(self):
        self.supported_types = ['SQLite', 'JSON', 'LevelDB']
        
    def extract_from_scan_results(self, scan_results_file: str) -> Dict[str, Any]:
        """
        Extract data from all databases found in scan results.
        
        Args:
            scan_results_file: Path to JSON file containing scan results
            
        Returns:
            Dictionary containing extracted data from all databases
        """
        logger.info(f"Extracting data from scan results: {scan_results_file}")
        
        # Load scan results
        with open(scan_results_file, 'r') as f:
            scan_data = json.load(f)
        
        extraction_results = {
            'metadata': {
                'extraction_start': datetime.now().isoformat(),
                'source_scan': scan_results_file,
                'databases_processed': 0,
                'messages_extracted': 0,
                'errors': []
            },
            'extracted_data': {}
        }
        
        # Process each application's databases
        for app_name, databases in scan_data['databases'].items():
            if not databases:
                continue
                
            logger.info(f"Processing {app_name} databases...")
            extraction_results['extracted_data'][app_name] = []
            
            for db_info in databases:
                if 'error' in db_info:
                    continue
                    
                try:
                    # Extract data based on database type
                    db_data = self.extract_messages(db_info['path'], db_info['type'])
                    
                    if db_data and db_data.get('messages'):
                        extraction_results['extracted_data'][app_name].append({
                            'database_file': db_info['filename'],
                            'database_path': db_info['path'],
                            'database_type': db_info['type'],
                            'extraction_timestamp': datetime.now().isoformat(),
                            'message_count': len(db_data['messages']),
                            'data': db_data
                        })
                        extraction_results['metadata']['messages_extracted'] += len(db_data['messages'])
                    
                    extraction_results['metadata']['databases_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to extract from {db_info['filename']}: {str(e)}"
                    logger.error(error_msg)
                    extraction_results['metadata']['errors'].append(error_msg)
        
        extraction_results['metadata']['extraction_end'] = datetime.now().isoformat()
        return extraction_results
    
    def extract_messages(self, database_path: str, db_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract messages from a database file.
        
        Args:
            database_path: Path to the database file
            db_type: Type of database (SQLite, JSON, LevelDB)
            
        Returns:
            Dictionary containing extracted messages and metadata
        """
        logger.debug(f"Extracting from {database_path} (type: {db_type})")
        
        if not os.path.exists(database_path):
            logger.error(f"Database file not found: {database_path}")
            return None
        
        try:
            if db_type == 'SQLite':
                return self._extract_sqlite(database_path)
            elif db_type == 'JSON':
                return self._extract_json(database_path)
            elif db_type.startswith('LevelDB'):
                return self._extract_leveldb(database_path)
            else:
                logger.warning(f"Unsupported database type: {db_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting from {database_path}: {e}")
            return None
    
    def _extract_sqlite(self, db_path: str) -> Optional[Dict[str, Any]]:
        """Extract data from SQLite database."""
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            messages = []
            metadata = {
                'tables': {},
                'total_tables': len(tables),
                'extraction_method': 'SQLite'
            }
            
            for table_name, in tables:
                try:
                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    
                    metadata['tables'][table_name] = {
                        'columns': columns,
                        'row_count': row_count
                    }
                    
                    # Look for message-like data
                    if row_count > 0 and self._is_message_table(table_name, columns):
                        table_messages = self._extract_messages_from_table(cursor, table_name, columns)
                        messages.extend(table_messages)
                        
                except Exception as e:
                    logger.warning(f"Error processing table {table_name}: {e}")
                    continue
            
            conn.close()
            
            return {
                'messages': messages,
                'metadata': metadata,
                'source_file': db_path
            }
            
        except Exception as e:
            logger.error(f"SQLite extraction failed: {e}")
            return None
    
    def _extract_json(self, json_path: str) -> Optional[Dict[str, Any]]:
        """Extract data from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = []
            
            # Try to find message-like structures in JSON
            if isinstance(data, list):
                messages = [item for item in data if self._is_message_like(item)]
            elif isinstance(data, dict):
                # Look for message arrays in the JSON structure
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        if self._is_message_like(value[0]):
                            messages.extend(value)
            
            return {
                'messages': messages,
                'metadata': {
                    'extraction_method': 'JSON',
                    'original_structure': type(data).__name__
                },
                'source_file': json_path
            }
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return None
    
    def _extract_leveldb(self, db_path: str) -> Optional[Dict[str, Any]]:
        """Extract data from LevelDB (placeholder - requires plyvel)."""
        logger.warning("LevelDB extraction not yet implemented")
        return {
            'messages': [],
            'metadata': {
                'extraction_method': 'LevelDB (not implemented)',
                'note': 'LevelDB extraction requires plyvel library and additional implementation'
            },
            'source_file': db_path
        }
    
    def _is_message_table(self, table_name: str, columns: List[str]) -> bool:
        """Determine if a table likely contains messages."""
        table_name_lower = table_name.lower()
        columns_lower = [col.lower() for col in columns]
        
        # Check for message-related table names
        message_keywords = ['message', 'chat', 'conversation', 'msg', 'text', 'content']
        if any(keyword in table_name_lower for keyword in message_keywords):
            return True
        
        # Check for message-related columns
        message_columns = ['message', 'text', 'content', 'body', 'msg']
        if any(col in columns_lower for col in message_columns):
            return True
        
        # Check for timestamp columns (messages usually have timestamps)
        time_columns = ['timestamp', 'time', 'created', 'sent', 'date']
        has_time = any(col in columns_lower for col in time_columns)
        
        return has_time and len(columns) >= 2  # At least timestamp and content
    
    def _extract_messages_from_table(self, cursor, table_name: str, columns: List[str]) -> List[Dict]:
        """Extract message records from a SQLite table."""
        messages = []
        
        try:
            # Select all data from the table
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            for row in rows:
                message_dict = {}
                for i, column in enumerate(columns):
                    if i < len(row):
                        message_dict[column] = row[i]
                
                # Add metadata
                message_dict['_table'] = table_name
                message_dict['_extraction_timestamp'] = datetime.now().isoformat()
                
                messages.append(message_dict)
                
        except Exception as e:
            logger.error(f"Error extracting from table {table_name}: {e}")
        
        return messages
    
    def _is_message_like(self, item: Any) -> bool:
        """Determine if a JSON item looks like a message."""
        if not isinstance(item, dict):
            return False
        
        # Look for message-like fields
        message_fields = ['message', 'text', 'content', 'body', 'msg']
        time_fields = ['timestamp', 'time', 'created', 'sent', 'date']
        
        has_message = any(field in item for field in message_fields)
        has_time = any(field in item for field in time_fields)
        
        return has_message or (has_time and len(item) >= 2)
