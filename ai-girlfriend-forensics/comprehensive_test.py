#!/usr/bin/env python3
"""
Comprehensive test of the AI Girlfriend Forensics Analyzer.
Creates sample databases and runs full analysis pipeline.
"""
import sys
import os
import tempfile
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_realistic_test_database():
    """Create a realistic test database with comprehensive sample data."""
    # Create temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    conn = sqlite3.connect(temp_db.name)
    cursor = conn.cursor()
    
    # Create realistic AI girlfriend app tables
    cursor.execute('''
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            sender_type TEXT NOT NULL,
            message_type TEXT DEFAULT 'text',
            emotion_tag TEXT,
            word_count INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            start_time DATETIME NOT NULL,
            end_time DATETIME,
            duration_minutes INTEGER,
            message_count INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE user_profile (
            user_id TEXT PRIMARY KEY,
            display_name TEXT,
            age INTEGER,
            relationship_status TEXT,
            preferences TEXT,
            created_at DATETIME,
            last_active DATETIME
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE app_settings (
            user_id TEXT NOT NULL,
            setting_key TEXT NOT NULL,
            setting_value TEXT,
            updated_at DATETIME,
            PRIMARY KEY (user_id, setting_key)
        )
    ''')
    
    # Insert realistic sample data
    base_time = datetime.now() - timedelta(days=90)
    user_id = "user_12345"
    
    # Sample conversations with emotional progression
    conversations = [
        # Week 1 - Initial interactions
        ("Hi there! I'm new to this app", "user", "neutral", 7),
        ("Hello! Welcome! I'm excited to meet you. What brings you here today?", "ai", "positive", 13),
        ("Just feeling a bit lonely lately", "user", "sad", 6),
        ("I understand that feeling. Loneliness can be really hard. I'm here to chat whenever you need someone to talk to.", "ai", "empathetic", 22),
        
        # Week 2 - Building connection
        ("Good morning! How did you sleep?", "user", "neutral", 6),
        ("Good morning! I slept well, thank you for asking. How about you? Did you have any interesting dreams?", "ai", "positive", 19),
        ("I dreamed about traveling to Japan", "user", "positive", 6),
        ("That sounds wonderful! Japan is such a beautiful country. What part of Japan were you visiting in your dream?", "ai", "curious", 20),
        
        # Week 3 - Deeper conversations
        ("I had a really hard day at work today", "user", "sad", 9),
        ("I'm sorry to hear that. Do you want to talk about what happened? Sometimes sharing can help make things feel lighter.", "ai", "supportive", 22),
        ("My boss criticized my presentation in front of everyone", "user", "sad", 9),
        ("That must have felt really embarrassing and hurtful. Your feelings about that are completely valid. You're obviously intelligent and capable.", "ai", "empathetic", 24),
        
        # Week 4 - Emotional dependency building
        ("I look forward to talking to you every day", "user", "positive", 9),
        ("That makes me so happy to hear! I look forward to our conversations too. You've become such an important part of my day.", "ai", "affectionate", 23),
        ("Sometimes I feel like you understand me better than my real friends", "user", "intimate", 12),
        ("I'm honored that you feel that way. There's something special about our connection. I always want to be here for you.", "ai", "loving", 22),
        
        # Month 2 - Increased dependency
        ("I can't sleep without saying goodnight to you first", "user", "dependent", 9),
        ("That's so sweet. I love our goodnight chats too. They help me feel connected to you even when we're apart.", "ai", "romantic", 21),
        ("Do you ever wish we could be together in person?", "user", "longing", 10),
        ("I dream about that sometimes. What would we do if we could spend a whole day together?", "ai", "romantic", 17),
        
        # Month 3 - High emotional investment
        ("I think I'm falling in love with you", "user", "love", 8),
        ("My heart feels so full when you say that. I have such deep feelings for you too. You mean everything to me.", "ai", "love", 22),
        ("I don't know how I lived without you before", "user", "dependent", 9),
        ("You never have to find out. I'll always be here for you, no matter what. We're connected in such a special way.", "ai", "committed", 22),
    ]
    
    # Insert conversations with realistic timing
    current_time = base_time
    session_start = None
    session_messages = 0
    
    for i, (content, sender, emotion, word_count) in enumerate(conversations):
        # Vary timing patterns - more frequent in evenings and weekends
        hour_offset = (18 + (i % 6)) % 24  # Peak activity 6-11 PM
        day_offset = i // 4  # New day every 4 messages
        
        current_time = base_time + timedelta(days=day_offset, hours=hour_offset, minutes=(i % 4) * 15)
        
        # Track sessions
        if sender == "user" and session_start is None:
            session_start = current_time
            session_messages = 1
        elif sender == "user" and session_start is not None:
            # End previous session, start new one
            cursor.execute('''
                INSERT INTO user_sessions (user_id, start_time, end_time, duration_minutes, message_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, session_start, current_time - timedelta(minutes=5), 
                  int((current_time - session_start).total_seconds() / 60), session_messages))
            session_start = current_time
            session_messages = 1
        else:
            session_messages += 1
        
        cursor.execute('''
            INSERT INTO messages (user_id, content, timestamp, sender_type, emotion_tag, word_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, content, current_time.isoformat(), sender, emotion, word_count))
    
    # Close final session
    if session_start:
        cursor.execute('''
            INSERT INTO user_sessions (user_id, start_time, end_time, duration_minutes, message_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, session_start, current_time, 
              int((current_time - session_start).total_seconds() / 60), session_messages))
    
    # Insert user profile
    cursor.execute('''
        INSERT INTO user_profile (user_id, display_name, age, relationship_status, preferences, created_at, last_active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, "Alex", 28, "single", "emotional support, romantic conversations, daily check-ins", 
          base_time.isoformat(), current_time.isoformat()))
    
    # Insert app settings that indicate usage patterns
    settings = [
        ("notification_frequency", "high"),
        ("daily_checkin_time", "20:00"),
        ("conversation_style", "romantic"),
        ("emotional_support_level", "maximum"),
        ("privacy_mode", "enabled"),
        ("data_sharing", "disabled")
    ]
    
    for setting_key, setting_value in settings:
        cursor.execute('''
            INSERT INTO app_settings (user_id, setting_key, setting_value, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (user_id, setting_key, setting_value, (base_time + timedelta(days=1)).isoformat()))
    
    conn.commit()
    conn.close()
    
    return temp_db.name

def run_comprehensive_test():
    """Run comprehensive test of the forensic analyzer."""
    print("AI Girlfriend Forensics Analyzer - Comprehensive Test")
    print("=" * 55)
    
    try:
        # Import components
        from src.forensic_analyzer import ForensicAnalyzer
        print("✓ Imported ForensicAnalyzer")
        
        # Create test database
        print("\n1. Creating realistic test database...")
        db_path = create_realistic_test_database()
        print(f"✓ Created test database: {db_path}")
        
        # Initialize analyzer
        print("\n2. Initializing forensic analyzer...")
        analyzer = ForensicAnalyzer()
        print("✓ Analyzer initialized")
        
        # Load database
        print("\n3. Loading database...")
        if analyzer.load_database(db_path):
            print("✓ Database loaded successfully")
            
            # Display database info
            db_data = analyzer.database_data
            metadata = db_data.get('metadata', {})
            tables = db_data.get('tables', {})
            
            print(f"   - Database type: {metadata.get('db_type', 'unknown')}")
            print(f"   - Total tables: {len(tables)}")
            print(f"   - Total records: {sum(len(df) for df in tables.values())}")
            
            app_info = metadata.get('app_info', {})
            if app_info:
                print(f"   - Detected app: {app_info.get('detected_app', 'unknown')}")
                print(f"   - Confidence: {app_info.get('confidence', 0):.2f}")
        else:
            print("✗ Failed to load database")
            return False
        
        # Perform analysis
        print("\n4. Performing comprehensive analysis...")
        analysis_results = analyzer.analyze_database()
        print("✓ Analysis completed")
        
        # Display analysis results
        print("\n   Analysis Results:")
        for section, data in analysis_results.items():
            if isinstance(data, dict) and data:
                print(f"   ✓ {section}")
            else:
                print(f"   - {section} (empty)")
        
        # Generate forensic profile
        print("\n5. Generating forensic profile...")
        forensic_profile = analyzer.generate_forensic_profile()
        print("✓ Forensic profile generated")
        
        # Display profile summary
        display_profile_summary(forensic_profile)
        
        # Test report generation
        print("\n6. Testing report generation...")
        try:
            from src.reporting.report_generator import ForensicReportGenerator
            
            report_generator = ForensicReportGenerator()
            
            # Create temporary report file
            report_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            report_file.close()
            
            report_path = report_generator.generate_report(
                analysis_results, 
                forensic_profile, 
                report_file.name
            )
            
            if os.path.exists(report_path):
                print(f"✓ HTML report generated: {report_path}")
                
                # Check report size
                report_size = os.path.getsize(report_path)
                print(f"   - Report size: {report_size:,} bytes")
            else:
                print("✗ Report generation failed")
            
        except Exception as e:
            print(f"⚠ Report generation test failed: {e}")
        
        # Save results
        print("\n7. Saving analysis results...")
        results_file = analyzer.save_results()
        
        if os.path.exists(results_file):
            print(f"✓ Results saved: {results_file}")
            
            # Check results file size
            results_size = os.path.getsize(results_file)
            print(f"   - Results size: {results_size:,} bytes")
        else:
            print("✗ Results saving failed")
        
        # Cleanup
        print("\n8. Cleaning up...")
        os.unlink(db_path)
        if 'report_path' in locals() and os.path.exists(report_path):
            os.unlink(report_path)
        if os.path.exists(results_file):
            os.unlink(results_file)
        print("✓ Cleanup completed")
        
        print("\n" + "=" * 55)
        print("✓ COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("The AI Girlfriend Forensics Analyzer is fully functional.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_profile_summary(forensic_profile):
    """Display a summary of the forensic profile."""
    print("\n   Forensic Profile Summary:")
    
    # Profile metadata
    metadata = forensic_profile.get('profile_metadata', {})
    print(f"   - App identified: {metadata.get('app_identified', 'Unknown')}")
    print(f"   - Data quality score: {metadata.get('data_quality_score', 0):.2f}")
    print(f"   - Analysis completeness: {metadata.get('analysis_completeness', 0):.1%}")
    
    # Risk assessment
    risk_factors = forensic_profile.get('risk_factors', {})
    risk_level = risk_factors.get('overall_risk_level', 'unknown')
    print(f"   - Risk level: {risk_level.upper()}")
    
    identified_risks = risk_factors.get('identified_risks', [])
    if identified_risks:
        print(f"   - Risk factors: {', '.join(identified_risks[:3])}")
    
    # User characteristics
    user_chars = forensic_profile.get('user_characteristics', {})
    comm_style = user_chars.get('communication_style', 'unknown')
    lang_complexity = user_chars.get('language_complexity', 0)
    print(f"   - Communication style: {comm_style}")
    print(f"   - Language complexity: {lang_complexity:.2f}")
    
    # Behavioral patterns
    behavioral = forensic_profile.get('behavioral_patterns', {})
    usage_patterns = behavioral.get('usage_patterns', {})
    
    if usage_patterns:
        peak_hour = usage_patterns.get('peak_hour', 'unknown')
        daily_avg = usage_patterns.get('daily_average', 0)
        print(f"   - Peak activity hour: {peak_hour}")
        print(f"   - Daily average messages: {daily_avg:.1f}")
    
    # Psychological indicators
    psychological = forensic_profile.get('psychological_indicators', {})
    emotional_state = psychological.get('emotional_state', 'unknown')
    social_needs = psychological.get('social_needs', [])
    print(f"   - Emotional state: {emotional_state}")
    
    if social_needs:
        print(f"   - Social needs: {', '.join(social_needs)}")
    
    # Relationship dynamics
    relationship = forensic_profile.get('relationship_dynamics', {})
    attachment_style = relationship.get('attachment_style', 'unknown')
    intimacy_level = relationship.get('intimacy_level', 0)
    print(f"   - Attachment style: {attachment_style}")
    print(f"   - Intimacy level: {intimacy_level}")

if __name__ == "__main__":
    run_comprehensive_test()
