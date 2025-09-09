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
   # Basic usage with directory
   python main.py identify /path/to/filesystem/image
   
   # Basic usage with ZIP file
   python main.py identify /path/to/forensic_image.zip
   
   # With verbose output and custom results file
   python main.py --verbose identify /path/to/filesystem/image --output results/scan_results.json
   
   # Windows example with full Python path (ZIP file)
   "C:\path\to\python.exe" main.py --verbose identify "C:\path\to\forensic_image.zip" --output "output/test_results.json"
   
   # Windows example with directory
   "C:\path\to\python.exe" main.py --verbose identify "C:\path\to\filesystem\image" --output "output/test_results.json"
   ```

3. Extract data (coming soon):
   ```bash
   python main.py extract /path/to/image
   ```

4. Analyze and generate profile (coming soon):
   ```bash
   python main.py analyze /path/to/extracted/data
   ```

## Features

- **Flexible Input Support**: Accepts both directories and ZIP files
- **Multi-platform Database Detection**: Identifies databases across various AI companion apps
- **Multiple Database Formats**: Support for SQLite, LevelDB, and JSON formats
- **Automatic ZIP Extraction**: Seamlessly handles compressed forensic images
- **Forensic Integrity**: File hashing (MD5/SHA256) for evidence validation
- **Comprehensive Metadata**: Timestamps, file sizes, and database types
- **Machine Learning Ready**: Framework for ML-based message importance scoring
- **Behavioral Pattern Analysis**: Foundation for user profiling
- **Comprehensive Reporting**: Detailed JSON and console output
- **Chain of Custody**: Complete documentation for legal compliance

## Supported Applications

- Replika
- Character.AI
- kindred.AI
- persona.AI
- fantasy.AI
- linky.AI
  
