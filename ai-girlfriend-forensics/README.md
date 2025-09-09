# AI Girlfriend Forensics Analyzer

A machine learning-based forensic analysis tool for extracting user behavioral profiles from AI companion application databases.

## Features

- **Multi-Database Support**: Automatically detects and parses various database formats (SQLite, JSON, MongoDB exports)
- **Schema Discovery**: Intelligent identification of database structure and content types
- **Behavioral Analysis**: ML-powered user profiling and pattern recognition
- **Timeline Reconstruction**: Temporal analysis of user interactions
- **Comprehensive Reporting**: Detailed forensic reports with visualizations

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLP models:
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
   ```

## Usage

```python
from forensic_analyzer import ForensicAnalyzer

# Initialize analyzer
analyzer = ForensicAnalyzer()

# Load database
analyzer.load_database("path/to/database.db")

# Generate forensic profile
profile = analyzer.generate_profile()

# Create report
analyzer.generate_report("output_report.html")
```

## Project Structure

- `src/`: Core implementation
  - `database_parsers/`: Database format parsers
  - `analysis/`: ML analysis modules
  - `profiling/`: User profiling algorithms
  - `reporting/`: Report generation
- `tests/`: Unit tests
- `examples/`: Usage examples
- `data/`: Sample data for testing

## Legal Notice

This tool is designed for legitimate digital forensics and cybersecurity research purposes only. Users must ensure compliance with all applicable laws and regulations.
