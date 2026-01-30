"""Test reporting module for generating and formatting test results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TestReporter:
    """Generates test reports in various formats."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_summary(self, results: dict[str, Any]) -> str:
        """Generate a text summary of test results."""
        total = results.get("total", 0)
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        skipped = results.get("skipped", 0)

        summary = f"""
Test Execution Summary
======================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Tests: {total}
Passed:      {passed}
Failed:      {failed}
Skipped:     {skipped}

Pass Rate:   {(passed / total * 100) if total > 0 else 0:.1f}%
"""
        return summary

    def save_json_report(self, results: dict[str, Any], filename: str = "results.json") -> Path:
        """Save test results as JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return output_path
