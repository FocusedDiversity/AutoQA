"""Allure report generation for test results."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from ..testing.executor import AssertionResult, StepResult, TestResult

logger = logging.getLogger(__name__)


class AllureReporter:
    """Generates Allure-compatible test reports.

    Creates JSON result files that can be processed by
    allure-commandline to generate HTML reports.
    """

    def __init__(self, results_dir: str = "allure-results"):
        """Initialize the reporter.

        Args:
            results_dir: Directory for Allure result files.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._attachments_dir = self.results_dir / "attachments"
        self._attachments_dir.mkdir(exist_ok=True)

    def report_test(self, result: TestResult) -> str:
        """Generate Allure report for a test result.

        Args:
            result: Test execution result.

        Returns:
            Path to the generated result file.
        """
        test_uuid = str(uuid4())

        # Build steps
        allure_steps = []
        for step_result in result.step_results:
            allure_step = self._convert_step(step_result)
            allure_steps.append(allure_step)

            # Add screenshot attachment if present
            if step_result.screenshot_path:
                attachment_id = self._add_attachment(
                    step_result.screenshot_path,
                    "image/png",
                    "Failure Screenshot"
                )
                allure_step["attachments"] = [{
                    "name": "Screenshot",
                    "source": attachment_id,
                    "type": "image/png"
                }]

        # Build assertions as steps
        for assertion_result in result.assertion_results:
            allure_steps.append(self._convert_assertion(assertion_result))

        # Calculate timing
        start_ms = int(result.start_time.timestamp() * 1000) if result.start_time else 0
        stop_ms = int(result.end_time.timestamp() * 1000) if result.end_time else start_ms

        # Build result container
        allure_result = {
            "uuid": test_uuid,
            "historyId": self._generate_history_id(result.test.name),
            "name": result.test.name,
            "description": result.test.description or "",
            "status": "passed" if result.passed else "failed",
            "statusDetails": {
                "message": result.error or "",
                "trace": ""
            } if result.error else {},
            "stage": "finished",
            "steps": allure_steps,
            "labels": self._build_labels(result),
            "links": [],
            "start": start_ms,
            "stop": stop_ms,
            "parameters": self._build_parameters(result)
        }

        # Write result file
        result_file = self.results_dir / f"{test_uuid}-result.json"
        with open(result_file, "w") as f:
            json.dump(allure_result, f, indent=2)

        logger.debug(f"Wrote Allure result: {result_file}")
        return str(result_file)

    def _convert_step(self, step_result: StepResult) -> dict:
        """Convert a step result to Allure format.

        Args:
            step_result: Step execution result.

        Returns:
            Allure step dictionary.
        """
        action = step_result.step.action
        step_name = f"{action.action_type.value}: {action.target.get('text', action.target.get('selector', ''))[:50]}"

        return {
            "name": step_name,
            "status": "passed" if step_result.success else "failed",
            "statusDetails": {
                "message": step_result.error or ""
            } if step_result.error else {},
            "stage": "finished",
            "parameters": [
                {"name": "user", "value": step_result.step.user},
                {"name": "selector", "value": action.target.get("selector", "")},
                {"name": "value", "value": str(action.value) if action.value else ""}
            ],
            "start": 0,
            "stop": step_result.duration_ms
        }

    def _convert_assertion(self, assertion_result: AssertionResult) -> dict:
        """Convert an assertion result to Allure format.

        Args:
            assertion_result: Assertion result.

        Returns:
            Allure step dictionary.
        """
        assertion = assertion_result.assertion
        assertion_name = f"Assert {assertion.assertion_type.value}: {assertion.check or 'check'}"

        return {
            "name": assertion_name,
            "status": "passed" if assertion_result.passed else "failed",
            "statusDetails": {
                "message": assertion_result.error or "",
                "expected": str(assertion.expected) if assertion.expected else "",
                "actual": str(assertion_result.actual_value) if assertion_result.actual_value else ""
            } if assertion_result.error or not assertion_result.passed else {},
            "stage": "finished",
            "parameters": [
                {"name": "selector", "value": assertion.selector or ""},
                {"name": "expected", "value": str(assertion.expected) if assertion.expected else ""},
                {"name": "actual", "value": str(assertion_result.actual_value) if assertion_result.actual_value else ""}
            ]
        }

    def _build_labels(self, result: TestResult) -> list[dict]:
        """Build Allure labels for a test.

        Args:
            result: Test result.

        Returns:
            List of label dictionaries.
        """
        labels = [
            {"name": "suite", "value": "AutoQA Generated Tests"},
            {"name": "framework", "value": "AutoQA"},
            {"name": "language", "value": "python"}
        ]

        # Add tags
        for tag in result.test.tags:
            labels.append({"name": "tag", "value": tag})

        # Add users
        for user in result.test.users:
            labels.append({"name": "user", "value": user})

        # Add source story if present
        if result.test.source_story:
            labels.append({"name": "story", "value": result.test.source_story})

        return labels

    def _build_parameters(self, result: TestResult) -> list[dict]:
        """Build Allure parameters for a test.

        Args:
            result: Test result.

        Returns:
            List of parameter dictionaries.
        """
        return [
            {"name": "users", "value": ", ".join(result.test.users)},
            {"name": "steps_count", "value": str(len(result.test.steps))},
            {"name": "assertions_count", "value": str(len(result.test.assertions))},
            {"name": "duration_ms", "value": str(result.duration_ms)}
        ]

    def _add_attachment(self, file_path: str, mime_type: str, name: str) -> str:
        """Add an attachment to the report.

        Args:
            file_path: Path to the file.
            mime_type: MIME type of the file.
            name: Attachment name.

        Returns:
            Attachment source ID.
        """
        import shutil

        source_id = f"{uuid4()}-attachment"
        ext = Path(file_path).suffix
        dest_path = self._attachments_dir / f"{source_id}{ext}"

        try:
            shutil.copy(file_path, dest_path)
        except Exception as e:
            logger.warning(f"Failed to copy attachment: {e}")
            return ""

        return f"{source_id}{ext}"

    def _generate_history_id(self, test_name: str) -> str:
        """Generate a stable history ID for a test.

        Args:
            test_name: Test name.

        Returns:
            History ID string.
        """
        import hashlib
        return hashlib.md5(test_name.encode()).hexdigest()

    def report_suite(self, results: list[TestResult], suite_name: str = "AutoQA Suite") -> dict:
        """Generate report for a test suite.

        Args:
            results: List of test results.
            suite_name: Name of the suite.

        Returns:
            Summary statistics.
        """
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        total_duration = sum(r.duration_ms for r in results)

        # Write individual results
        for result in results:
            self.report_test(result)

        # Write environment info
        self._write_environment()

        # Write categories
        self._write_categories()

        summary = {
            "suite": suite_name,
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(results) if results else 0,
            "duration_ms": total_duration,
            "results_dir": str(self.results_dir)
        }

        logger.info(f"Suite report: {passed}/{len(results)} passed ({summary['pass_rate']:.1%})")
        return summary

    def _write_environment(self) -> None:
        """Write environment.properties file."""
        env_file = self.results_dir / "environment.properties"
        with open(env_file, "w") as f:
            f.write(f"Framework=AutoQA\n")
            f.write(f"Python={os.sys.version.split()[0]}\n")
            f.write(f"Generated={datetime.now().isoformat()}\n")

    def _write_categories(self) -> None:
        """Write categories.json for failure categorization."""
        categories = [
            {
                "name": "Element Not Found",
                "matchedStatuses": ["failed"],
                "messageRegex": ".*Timeout.*waiting.*selector.*"
            },
            {
                "name": "Assertion Failed",
                "matchedStatuses": ["failed"],
                "messageRegex": ".*assertion.*failed.*"
            },
            {
                "name": "Network Error",
                "matchedStatuses": ["failed"],
                "messageRegex": ".*net::.*"
            }
        ]

        categories_file = self.results_dir / "categories.json"
        with open(categories_file, "w") as f:
            json.dump(categories, f, indent=2)
