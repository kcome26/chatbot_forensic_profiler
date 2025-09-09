"""
JSON database parser for AI girlfriend applications.
"""
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from .base_parser import DatabaseParser


class JSONParser(DatabaseParser):
    """Parser for JSON-based databases and exports."""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.data = None
        
    def connect(self) -> bool:
        """Load JSON data from file."""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"Failed to load JSON file: {e}")
            return False
    
    def discover_schema(self) -> Dict[str, Any]:
        """Discover JSON data structure."""
        if self.data is None:
            raise RuntimeError("JSON data not loaded")
            
        schema = {}
        
        if isinstance(self.data, dict):
            # Top-level dictionary structure
            for key, value in self.data.items():
                schema[key] = self._analyze_json_structure(value, key)
        elif isinstance(self.data, list):
            # Top-level array structure
            schema['root_array'] = self._analyze_json_structure(self.data, 'root_array')
        else:
            # Single value
            schema['root_value'] = {
                'type': type(self.data).__name__,
                'sample': str(self.data)[:100]
            }
            
        self.schema = schema
        return schema
    
    def _analyze_json_structure(self, data: Any, key_name: str) -> Dict[str, Any]:
        """Analyze structure of JSON data recursively."""
        structure = {
            'type': type(data).__name__,
            'fields': {},
            'array_length': 0,
            'sample_data': None
        }
        
        if isinstance(data, dict):
            # Dictionary - analyze each field
            for field_key, field_value in data.items():
                if isinstance(field_value, (dict, list)):
                    structure['fields'][field_key] = self._analyze_json_structure(field_value, field_key)
                else:
                    structure['fields'][field_key] = {
                        'type': type(field_value).__name__,
                        'sample': str(field_value)[:50] if field_value is not None else None
                    }
                    
        elif isinstance(data, list):
            structure['array_length'] = len(data)
            if data:
                # Analyze first few items to understand structure
                first_item = data[0]
                structure['item_structure'] = self._analyze_json_structure(first_item, f"{key_name}_item")
                
                # Take sample for preview
                structure['sample_data'] = data[:3]
                
        return structure
    
    def extract_tables(self) -> Dict[str, pd.DataFrame]:
        """Convert JSON data to pandas DataFrames."""
        if self.data is None:
            raise RuntimeError("JSON data not loaded")
            
        tables = {}
        
        if isinstance(self.data, dict):
            for key, value in self.data.items():
                df = self._json_to_dataframe(value, key)
                if df is not None and not df.empty:
                    tables[key] = df
                    
        elif isinstance(self.data, list):
            df = self._json_to_dataframe(self.data, 'root_data')
            if df is not None and not df.empty:
                tables['root_data'] = df
                
        # Classify columns for each table
        for table_name, df in tables.items():
            classifications = self.classify_columns(df)
            df.attrs['column_classifications'] = classifications
            df.attrs['table_name'] = table_name
            
        return tables
    
    def _json_to_dataframe(self, data: Any, table_name: str) -> Optional[pd.DataFrame]:
        """Convert JSON data to DataFrame."""
        try:
            if isinstance(data, list):
                if not data:
                    return None
                    
                # Check if list contains dictionaries (records)
                if isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    # List of primitives
                    return pd.DataFrame({f'{table_name}_value': data})
                    
            elif isinstance(data, dict):
                # Check if values are lists (columnar data)
                if all(isinstance(v, list) for v in data.values()):
                    return pd.DataFrame(data)
                else:
                    # Single record
                    return pd.DataFrame([data])
                    
            else:
                # Single value
                return pd.DataFrame({f'{table_name}_value': [data]})
                
        except Exception as e:
            print(f"Failed to convert {table_name} to DataFrame: {e}")
            return None
    
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract JSON file metadata."""
        metadata = {
            'db_path': self.db_path,
            'db_type': 'JSON',
            'extraction_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(self.db_path),
            'file_modified': datetime.fromtimestamp(os.path.getmtime(self.db_path)).isoformat(),
            'schema_summary': self._summarize_schema(),
            'app_info': {}
        }
        
        # Try to detect app information from JSON structure
        metadata['app_info'] = self._detect_app_info()
        
        self.metadata = metadata
        return metadata
    
    def _summarize_schema(self) -> Dict[str, Any]:
        """Create a summary of the JSON schema."""
        if not self.schema:
            return {}
            
        summary = {
            'top_level_keys': list(self.schema.keys()),
            'structure_type': 'object' if isinstance(self.data, dict) else 'array',
            'estimated_records': 0
        }
        
        # Estimate number of records
        for key, structure in self.schema.items():
            if structure.get('type') == 'list':
                summary['estimated_records'] += structure.get('array_length', 0)
            elif structure.get('type') == 'dict':
                summary['estimated_records'] += 1
                
        return summary
    
    def _detect_app_info(self) -> Dict[str, Any]:
        """Attempt to detect app from JSON structure."""
        app_info = {
            'detected_app': 'unknown',
            'confidence': 0.0,
            'indicators': []
        }
        
        if not self.data:
            return app_info
            
        # Convert data to string for pattern matching
        data_str = json.dumps(self.data, default=str).lower()
        
        # App-specific patterns in JSON data
        app_patterns = {
            'replika': ['replika', 'avatar', 'memory', 'diary', 'mood'],
            'character_ai': ['character.ai', 'persona', 'character', 'bot'],
            'anima': ['anima', 'relationship', 'personality', 'emotion'],
            'chai': ['chai', 'chatbot', 'conversation'],
            'romantic_ai': ['romantic', 'dating', 'romance', 'relationship']
        }
        
        best_match = None
        best_score = 0
        
        for app_name, patterns in app_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if pattern in data_str:
                    score += 1
                    matched_patterns.append(pattern)
            
            confidence = score / len(patterns)
            
            if confidence > best_score:
                best_score = confidence
                best_match = app_name
                app_info['indicators'] = matched_patterns
        
        if best_match and best_score > 0.2:  # Lower threshold for JSON
            app_info['detected_app'] = best_match
            app_info['confidence'] = best_score
        
        return app_info
