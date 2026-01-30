"""Configuration settings for AutoQA."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
TEST_DIR = BASE_DIR / "tests"
REPORTS_DIR = BASE_DIR / "reports"

# Test settings
VERBOSE = True
GENERATE_HTML_REPORT = True
COVERAGE_ENABLED = False

# Timeouts (in seconds)
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300

# Parallel execution
PARALLEL_WORKERS = os.cpu_count() or 1
