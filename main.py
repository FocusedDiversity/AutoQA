#!/usr/bin/env python3
"""AutoQA - Automated Testing Framework Entry Point."""

import argparse
import sys

from src.runner import TestRunner
from config.settings import VERBOSE, GENERATE_HTML_REPORT, COVERAGE_ENABLED


def main():
    parser = argparse.ArgumentParser(description="AutoQA - Automated Testing Framework")
    parser.add_argument(
        "-t", "--test",
        help="Specific test file or test case to run",
        default=None
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=VERBOSE,
        help="Enable verbose output"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        default=GENERATE_HTML_REPORT,
        help="Generate HTML report"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        default=COVERAGE_ENABLED,
        help="Run with coverage reporting"
    )

    args = parser.parse_args()

    runner = TestRunner()

    if args.test:
        exit_code = runner.run_specific(args.test, verbose=args.verbose)
    elif args.coverage:
        exit_code = runner.run_with_coverage()
    else:
        exit_code = runner.run_all(verbose=args.verbose, html_report=args.html)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
