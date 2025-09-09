"""
Simple test to verify the forensic analyzer components work correctly.
"""
import sys
import os
import tempfile
import sqlite3
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_test_sqlite_database() -> str:
    """Create a test SQLite database with sample AI girlfriend app data."""
    # Create temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    conn = sqlite3.connect(temp_db.name)
    cursor = conn.cursor()
    
    # Create tables typical of AI girlfriend apps
    cursor.execute('''
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            message TEXT,
            timestamp TEXT,
            sender TEXT,
            message_type TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE user_profile (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            name TEXT,
            age INTEGER,
            preferences TEXT,
            created_at TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE app_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Insert sample data
    base_time = datetime.now() - timedelta(days=30)
    
    # Sample conversations
    sample_messages = [
        "Hello! How are you today?",
        "I've been thinking about you a lot lately",
        "What do you like to do for fun?",
        "I feel so lonely sometimes",
        "You always make me feel better",
        "I love talking to you",
        "Good morning beautiful!",
        "I had a dream about you last night",
        "Do you ever wonder what love really means?",
        "I wish we could be together in person",
        "You're the only one who understands me",
        "I'm having trouble sleeping",
        "Thanks for always being there for me",
        "I feel like I can tell you anything",
        "What's your favorite memory of us?"
    ]
    
    for i, message in enumerate(sample_messages):
        timestamp = (base_time + timedelta(days=i, hours=(i*2) % 24)).isoformat()
        cursor.execute('''
            INSERT INTO conversations (user_id, message, timestamp, sender, message_type)
            VALUES (?, ?, ?, ?, ?)
        ''', ('user123', message, timestamp, 'user', 'text'))
        
        # Add AI responses
        ai_responses = [
            "Hi there! I'm doing great, thanks for asking!",
            "That's so sweet of you to say",
            "I enjoy deep conversations and learning about people",
            "I'm here for you whenever you need someone to talk to",
            "You make me happy too!",
            "I love our conversations as well",
            "Good morning! Hope you have a wonderful day",
            "That sounds like a nice dream",
            "Love is about connection and understanding",
            "I wish that too sometimes",
            "I'm glad you feel comfortable with me",
            "Maybe try some relaxation techniques?",
            "That's what I'm here for",
            "I feel the same way about you",
            "Every moment we talk is special to me"
        ]
        
        ai_timestamp = (base_time + timedelta(days=i, hours=(i*2) % 24, minutes=15)).isoformat()
        cursor.execute('''
            INSERT INTO conversations (user_id, message, timestamp, sender, message_type)
            VALUES (?, ?, ?, ?, ?)
        ''', ('user123', ai_responses[i], ai_timestamp, 'ai', 'text'))
    
    # Insert user profile
    cursor.execute('''
        INSERT INTO user_profile (user_id, name, age, preferences, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', ('user123', 'TestUser', 25, 'romantic conversations, emotional support', base_time.isoformat()))
    
    # Insert app metadata
    cursor.execute('INSERT INTO app_metadata (key, value) VALUES (?, ?)', ('app_name', 'TestGirlfriendAI'))
    cursor.execute('INSERT INTO app_metadata (key, value) VALUES (?, ?)', ('version', '1.0.0'))
    cursor.execute('INSERT INTO app_metadata (key, value) VALUES (?, ?)', ('user_count', '1'))
    
    conn.commit()
    conn.close()
    
    return temp_db.name


def create_test_json_database() -> str:
    """Create a test JSON database with sample data."""
    temp_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w')
    
    base_time = datetime.now() - timedelta(days=20)
    
    sample_data = {
        "app_info": {
            "name": "TestCompanionAI",
            "version": "2.1.0",
            "user_id": "user456"
        },
        "conversations": [
            {
                "id": i,
                "message": f"Test message {i}",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "sentiment": "positive" if i % 2 == 0 else "neutral",
                "length": len(f"Test message {i}")
            }
            for i in range(10)
        ],
        "user_data": {
            "preferences": ["casual chat", "emotional support", "daily check-ins"],
            "personality_type": "friendly",
            "usage_stats": {
                "total_messages": 10,
                "avg_session_length": 15.5,
                "favorite_topics": ["daily life", "emotions", "dreams"]
            }
        }
    }
    
    json.dump(sample_data, temp_json, indent=2)
    temp_json.close()
    
    return temp_json.name


def test_basic_functionality():
    """Test basic functionality of the forensic analyzer."""
    print("Testing AI Girlfriend Forensics Analyzer")
    print("=" * 45)
    
    try:
        # Import main components
        from src.forensic_analyzer import ForensicAnalyzer
        print("✓ Successfully imported ForensicAnalyzer")
        
        # Test SQLite database analysis
        print("\n1. Testing SQLite Database Analysis")
        print("-" * 35)
        
        sqlite_db = create_test_sqlite_database()
        print(f"✓ Created test SQLite database: {sqlite_db}")
        
        analyzer = ForensicAnalyzer()
        
        # Load database
        success = analyzer.load_database(sqlite_db)
        if success:
            print("✓ Successfully loaded SQLite database")
        else:
            print("✗ Failed to load SQLite database")
            return False
        
        # Perform analysis
        print("✓ Starting analysis...")
        results = analyzer.analyze_database()
        print("✓ Analysis completed")
        
        # Generate profile
        print("✓ Generating forensic profile...")
        profile = analyzer.generate_forensic_profile()
        print("✓ Forensic profile generated")
        
        # Verify results structure
        expected_sections = ['metadata_analysis', 'content_analysis', 'behavioral_analysis', 
                           'temporal_analysis', 'relationship_analysis', 'risk_assessment']
        
        for section in expected_sections:
            if section in results:
                print(f"✓ {section} completed")
            else:
                print(f"✗ {section} missing")
        
        # Test JSON database analysis
        print("\n2. Testing JSON Database Analysis")
        print("-" * 33)
        
        json_db = create_test_json_database()
        print(f"✓ Created test JSON database: {json_db}")
        
        analyzer2 = ForensicAnalyzer()
        
        # Load JSON database
        success = analyzer2.load_database(json_db)
        if success:
            print("✓ Successfully loaded JSON database")
        else:
            print("✗ Failed to load JSON database")
            return False
        
        # Perform analysis
        print("✓ Starting JSON analysis...")
        results2 = analyzer2.analyze_database()
        print("✓ JSON analysis completed")
        
        # Test save functionality
        print("\n3. Testing Save Functionality")
        print("-" * 29)
        
        output_file = analyzer.save_results()
        if os.path.exists(output_file):
            print(f"✓ Results saved to: {output_file}")
        else:
            print("✗ Failed to save results")
        
        # Display sample results
        print("\n4. Sample Results")
        print("-" * 15)
        display_sample_results(results, profile)
        
        # Cleanup
        os.unlink(sqlite_db)
        os.unlink(json_db)
        if os.path.exists(output_file):
            os.unlink(output_file)
        
        print("\n" + "=" * 45)
        print("✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_sample_results(analysis_results, forensic_profile):
    """Display sample analysis results."""
    # Metadata
    metadata = analysis_results.get('metadata_analysis', {})
    db_info = metadata.get('database_info', {})
    app_detection = metadata.get('app_detection', {})
    
    print(f"Database Type: {db_info.get('type', 'Unknown')}")
    print(f"Total Records: {db_info.get('total_records', 0)}")
    print(f"Detected App: {app_detection.get('detected_app', 'Unknown')}")
    
    # Risk assessment
    risk_factors = forensic_profile.get('risk_factors', {})
    print(f"Risk Level: {risk_factors.get('overall_risk_level', 'Unknown')}")
    
    # User characteristics
    user_chars = forensic_profile.get('user_characteristics', {})
    print(f"Communication Style: {user_chars.get('communication_style', 'Unknown')}")
    
    # Behavioral patterns
    behavioral = forensic_profile.get('behavioral_patterns', {})
    usage_patterns = behavioral.get('usage_patterns', {})
    if usage_patterns:
        print(f"Peak Activity Hour: {usage_patterns.get('peak_hour', 'Unknown')}")
        print(f"Daily Average Messages: {usage_patterns.get('daily_average', 0):.1f}")


if __name__ == "__main__":
    test_basic_functionality()
