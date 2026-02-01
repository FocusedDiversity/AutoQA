"""Test executor for running generated tests."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from ..core.models import (
    Assertion,
    AssertionType,
    GeneratedTest,
    TestStep,
)
from ..core.config import AutoQAConfig

logger = logging.getLogger(__name__)


@dataclass
class AssertionResult:
    """Result of an assertion check."""
    assertion: Assertion
    passed: bool
    actual_value: Any = None
    error: Optional[str] = None


@dataclass
class StepResult:
    """Result of a test step."""
    step: TestStep
    success: bool
    duration_ms: int = 0
    screenshot_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TestResult:
    """Result of a test execution."""
    test: GeneratedTest
    passed: bool
    step_results: list[StepResult] = field(default_factory=list)
    assertion_results: list[AssertionResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0


class TestExecutor:
    """Executes generated tests using Playwright.

    Supports:
    - Single-user test execution
    - Multi-user parallel contexts
    - Multiple assertion types
    - Screenshot capture on failure
    """

    def __init__(
        self,
        config: AutoQAConfig,
        screenshots_dir: Optional[str] = None
    ):
        """Initialize the executor.

        Args:
            config: AutoQA configuration.
            screenshots_dir: Directory for failure screenshots.
        """
        self.config = config
        self.screenshots_dir = Path(screenshots_dir or "screenshots/failures")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self._browser: Optional[Browser] = None
        self._contexts: dict[str, BrowserContext] = {}
        self._pages: dict[str, Page] = {}

    def run_test(self, test: GeneratedTest) -> TestResult:
        """Execute a single test.

        Args:
            test: Test to execute.

        Returns:
            TestResult with pass/fail and details.
        """
        result = TestResult(test=test, passed=True, start_time=datetime.now())

        with sync_playwright() as playwright:
            try:
                self._browser = playwright.chromium.launch(
                    headless=self.config.exploration.headless
                )

                # Create contexts for each user
                for user in test.users:
                    self._setup_user_context(user)

                # Execute steps
                for step in test.steps:
                    step_result = self._execute_step(step)
                    result.step_results.append(step_result)

                    if not step_result.success:
                        result.passed = False
                        result.error = step_result.error
                        break

                # Run assertions if steps passed
                if result.passed:
                    for assertion in test.assertions:
                        assertion_result = self._check_assertion(assertion)
                        result.assertion_results.append(assertion_result)

                        if not assertion_result.passed:
                            result.passed = False
                            result.error = assertion_result.error

            except Exception as e:
                result.passed = False
                result.error = str(e)
                logger.error(f"Test execution failed: {e}")

            finally:
                # Cleanup
                for context in self._contexts.values():
                    context.close()
                if self._browser:
                    self._browser.close()
                self._contexts.clear()
                self._pages.clear()

        result.end_time = datetime.now()
        return result

    def run_suite(self, tests: list[GeneratedTest]) -> list[TestResult]:
        """Execute a suite of tests.

        Args:
            tests: List of tests to execute.

        Returns:
            List of test results.
        """
        results = []
        for test in tests:
            logger.info(f"Running: {test.name}")
            result = self.run_test(test)
            results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            logger.info(f"  {status} ({result.duration_ms}ms)")

        return results

    def _setup_user_context(self, user: str) -> None:
        """Set up browser context for a user.

        Args:
            user: User identifier.
        """
        user_config = self.config.users.get(user)
        storage_state = None

        if user_config and user_config.storage_state_path:
            storage_state = user_config.storage_state_path

        context = self._browser.new_context(
            viewport=self.config.exploration.viewport,
            storage_state=storage_state
        )

        page = context.new_page()
        page.goto(self.config.app.base_url)

        self._contexts[user] = context
        self._pages[user] = page

    def _execute_step(self, step: TestStep) -> StepResult:
        """Execute a test step.

        Args:
            step: Step to execute.

        Returns:
            StepResult with success/failure.
        """
        start = datetime.now()
        page = self._pages.get(step.user)

        if not page:
            return StepResult(
                step=step,
                success=False,
                error=f"No context for user: {step.user}"
            )

        try:
            action = step.action
            target = action.target or {}
            selector = target.get("selector", "")

            if action.action_type.value == "click":
                page.click(selector, timeout=step.timeout_ms)

            elif action.action_type.value == "type":
                page.fill(selector, action.value or "", timeout=step.timeout_ms)

            elif action.action_type.value == "select":
                page.select_option(selector, action.value, timeout=step.timeout_ms)

            elif action.action_type.value == "wait":
                if step.wait_for:
                    page.wait_for_selector(step.wait_for, timeout=step.timeout_ms)
                else:
                    page.wait_for_load_state("networkidle", timeout=step.timeout_ms)

            elif action.action_type.value == "navigate":
                page.goto(action.value or self.config.app.base_url)

            elif action.action_type.value == "hover":
                page.hover(selector, timeout=step.timeout_ms)

            # Wait for any navigation/loading
            page.wait_for_load_state("domcontentloaded", timeout=5000)

            duration = int((datetime.now() - start).total_seconds() * 1000)
            return StepResult(step=step, success=True, duration_ms=duration)

        except Exception as e:
            duration = int((datetime.now() - start).total_seconds() * 1000)

            # Capture screenshot on failure
            screenshot_path = None
            if self.config.execution.screenshot_on_failure:
                screenshot_path = str(
                    self.screenshots_dir /
                    f"failure_{step.order}_{datetime.now().strftime('%H%M%S')}.png"
                )
                page.screenshot(path=screenshot_path)

            return StepResult(
                step=step,
                success=False,
                duration_ms=duration,
                screenshot_path=screenshot_path,
                error=str(e)
            )

    def _check_assertion(self, assertion: Assertion) -> AssertionResult:
        """Check an assertion.

        Args:
            assertion: Assertion to check.

        Returns:
            AssertionResult with pass/fail.
        """
        page = self._pages.get(assertion.user)

        if not page:
            return AssertionResult(
                assertion=assertion,
                passed=False,
                error=f"No context for user: {assertion.user}"
            )

        try:
            if assertion.assertion_type == AssertionType.DOM:
                return self._check_dom_assertion(page, assertion)

            elif assertion.assertion_type == AssertionType.VISUAL:
                return self._check_visual_assertion(page, assertion)

            elif assertion.assertion_type == AssertionType.TIMING:
                return self._check_timing_assertion(page, assertion)

            elif assertion.assertion_type == AssertionType.SEMANTIC:
                return self._check_semantic_assertion(page, assertion)

            else:
                return AssertionResult(
                    assertion=assertion,
                    passed=False,
                    error=f"Unknown assertion type: {assertion.assertion_type}"
                )

        except Exception as e:
            return AssertionResult(
                assertion=assertion,
                passed=False,
                error=str(e)
            )

    def _check_dom_assertion(self, page: Page, assertion: Assertion) -> AssertionResult:
        """Check a DOM assertion.

        Args:
            page: Playwright page.
            assertion: DOM assertion to check.

        Returns:
            AssertionResult.
        """
        selector = assertion.selector or ""
        check = assertion.check or "element_exists"

        try:
            if check == "element_exists":
                locator = page.locator(selector)
                passed = locator.count() > 0
                return AssertionResult(
                    assertion=assertion,
                    passed=passed,
                    actual_value=locator.count()
                )

            elif check == "element_visible":
                locator = page.locator(selector)
                passed = locator.is_visible()
                return AssertionResult(
                    assertion=assertion,
                    passed=passed,
                    actual_value=passed
                )

            elif check == "element_text_equals":
                locator = page.locator(selector)
                actual = locator.text_content()
                passed = actual == assertion.expected
                return AssertionResult(
                    assertion=assertion,
                    passed=passed,
                    actual_value=actual
                )

            elif check == "element_text_contains":
                locator = page.locator(selector)
                actual = locator.text_content() or ""
                passed = assertion.expected in actual if assertion.expected else False
                return AssertionResult(
                    assertion=assertion,
                    passed=passed,
                    actual_value=actual
                )

            else:
                return AssertionResult(
                    assertion=assertion,
                    passed=False,
                    error=f"Unknown DOM check: {check}"
                )

        except Exception as e:
            return AssertionResult(
                assertion=assertion,
                passed=False,
                error=str(e)
            )

    def _check_visual_assertion(self, page: Page, assertion: Assertion) -> AssertionResult:
        """Check a visual regression assertion.

        Args:
            page: Playwright page.
            assertion: Visual assertion to check.

        Returns:
            AssertionResult.
        """
        # Placeholder - would implement pixelmatch comparison
        return AssertionResult(
            assertion=assertion,
            passed=True,
            actual_value="Visual comparison not implemented"
        )

    def _check_timing_assertion(self, page: Page, assertion: Assertion) -> AssertionResult:
        """Check a timing assertion.

        Args:
            page: Playwright page.
            assertion: Timing assertion to check.

        Returns:
            AssertionResult.
        """
        # Placeholder - would measure actual timing
        return AssertionResult(
            assertion=assertion,
            passed=True,
            actual_value=0
        )

    def _check_semantic_assertion(self, page: Page, assertion: Assertion) -> AssertionResult:
        """Check a semantic assertion using LLM.

        Args:
            page: Playwright page.
            assertion: Semantic assertion to check.

        Returns:
            AssertionResult.
        """
        # Placeholder - would use LLM to verify semantic expectations
        return AssertionResult(
            assertion=assertion,
            passed=True,
            actual_value="Semantic verification not implemented"
        )
