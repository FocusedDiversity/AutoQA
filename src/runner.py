"""Test runner module for executing test suites."""

import subprocess
import sys
from pathlib import Path


class TestRunner:
    """Manages test execution."""

    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)

    def run_all(self, verbose: bool = True, html_report: bool = False) -> int:
        """Run all tests in the test directory."""
        args = ["pytest", str(self.test_dir)]

        if verbose:
            args.append("-v")

        if html_report:
            args.extend(["--html=reports/report.html", "--self-contained-html"])

        result = subprocess.run(args)
        return result.returncode

    def run_specific(self, test_path: str, verbose: bool = True) -> int:
        """Run a specific test file or test case."""
        args = ["pytest", test_path]

        if verbose:
            args.append("-v")

        result = subprocess.run(args)
        return result.returncode

    def run_with_coverage(self, source_dir: str = "src") -> int:
        """Run tests with coverage reporting."""
        args = [
            "pytest",
            str(self.test_dir),
            f"--cov={source_dir}",
            "--cov-report=html:reports/coverage",
            "--cov-report=term",
        ]

        result = subprocess.run(args)
        return result.returncode
