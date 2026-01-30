# AutoQA

An automated testing framework for running and managing test suites.

## Project Structure

```
AutoQA/
├── src/              # Source code
│   ├── __init__.py
│   ├── runner.py     # Test runner
│   └── reporter.py   # Test reporting
├── tests/            # Test files
│   └── __init__.py
├── config/           # Configuration files
│   └── settings.py
├── reports/          # Generated test reports
├── requirements.txt  # Dependencies
└── main.py           # Entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Configuration

Edit `config/settings.py` to customize test settings.
