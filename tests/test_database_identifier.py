"""
Test the database identifier functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.database_identifier import DatabaseIdentifier

def test_database_identifier_initialization():
    """Test that DatabaseIdentifier initializes correctly."""
    identifier = DatabaseIdentifier()
    assert identifier is not None
    assert len(identifier.target_apps) > 0

def test_file_validation():
    """Test file validation logic."""
    identifier = DatabaseIdentifier()
    
    # Create a temporary SQLite file for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
        # Write SQLite header
        temp_file.write(b'SQLite format 3\x00')
        temp_file.write(b'\x00' * 1000)  # Make it large enough
        temp_path = Path(temp_file.name)
    
    try:
        assert identifier._is_likely_database(temp_path) == True
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__])
