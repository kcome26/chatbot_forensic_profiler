# Forensic Chat Analyzer

A digital forensics tool for analyzing AI companion chatbot databases to create behavioral profiles.

## Project Structure

```
forensic_chat_analyzer/
├── src/
│   ├── core/
│   │   ├── database_identifier.py    # Database discovery and identification
│   │   ├── data_extractor.py        # Database parsing and data extraction
│   │   └── filesystem_scanner.py    # Filesystem image processing
│   ├── extractors/
│   │   ├── sqlite_extractor.py      # SQLite database parser
│   │   ├── leveldb_extractor.py     # LevelDB parser
│   │   └── json_extractor.py        # JSON file parser
│   ├── analysis/
│   │   ├── ml_analyzer.py           # Machine learning analysis
│   │   ├── behavioral_profiler.py   # User behavior analysis
│   │   └── report_generator.py      # Report generation
│   └── utils/
│       ├── logging_config.py        # Logging configuration
│       ├── hash_validator.py        # File integrity validation
│       └── forensic_utils.py        # General forensic utilities
├── config/
│   ├── app_patterns.json           # Database location patterns
│   └── analysis_config.json        # ML model configuration
├── tests/
├── output/
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Create filesystem image using forensic tools (FTK Imager, dd, etc.)
2. Run database identification:
   ```bash
   python -m src.main identify /path/to/image
   ```
3. Extract data:
   ```bash
   python -m src.main extract /path/to/image
   ```
4. Analyze and generate profile:
   ```bash
   python -m src.main analyze /path/to/extracted/data
   ```

## Features

- Multi-platform database detection
- Support for SQLite, LevelDB, and JSON formats
- Machine learning-based message importance scoring
- Behavioral pattern analysis
- Comprehensive forensic reporting
- Chain of custody documentation

## Supported Applications

- Replika
- Character.AI
- kindred.AI
- persona.AI
- fantasy.AI
- linky.AI
  
