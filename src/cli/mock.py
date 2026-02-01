"""Mock web application testing CLI."""

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from ..mock.server import MockServer, MockServerConfig
from ..mock.scenarios import Scenario, ScenarioManager

logger = logging.getLogger(__name__)


@dataclass
class MockTestConfig:
    """Configuration for mock testing."""
    scenario: str = "login"  # Built-in scenario or path to custom
    host: str = "127.0.0.1"
    port: int = 5000
    headless: bool = True
    slow_mo: int = 0
    timeout_ms: int = 30000
    viewport: dict = None
    screenshots_dir: str = "screenshots/mock"
    auto_explore: bool = False
    generate_tests: bool = False

    def __post_init__(self):
        if self.viewport is None:
            self.viewport = {"width": 1280, "height": 720}


class MockTestRunner:
    """Run tests against mock web application.

    Features:
    - Built-in test scenarios (login, todo, form, collaboration)
    - Custom scenario support
    - Automatic exploration
    - Test generation from exploration
    - Multi-user testing with WebSocket
    """

    # Built-in test suites for each scenario
    BUILT_IN_TESTS = {
        "login": [
            {
                "name": "Successful login",
                "steps": [
                    {"action": "navigate", "url": "/login"},
                    {"action": "type", "selector": "#email", "value": "test@example.com"},
                    {"action": "type", "selector": "#password", "value": "password123"},
                    {"action": "click", "selector": "#submit-btn"},
                    {"action": "wait", "selector": "#welcome"},
                ],
                "assertions": [
                    {"type": "url", "contains": "/dashboard"},
                    {"type": "visible", "selector": "#logout-btn"},
                ]
            },
            {
                "name": "Logout flow",
                "steps": [
                    {"action": "navigate", "url": "/dashboard"},
                    {"action": "click", "selector": "#logout-btn"},
                ],
                "assertions": [
                    {"type": "url", "contains": "/"},
                    {"type": "visible", "selector": "#login-link"},
                ]
            }
        ],
        "todo": [
            {
                "name": "Add todo item",
                "steps": [
                    {"action": "navigate", "url": "/todo"},
                    {"action": "type", "selector": "#new-todo", "value": "Test todo item"},
                    {"action": "click", "selector": "#add-btn"},
                ],
                "assertions": [
                    {"type": "text", "selector": "#todo-list", "contains": "Test todo item"},
                ]
            },
            {
                "name": "Toggle todo completion",
                "steps": [
                    {"action": "navigate", "url": "/todo"},
                    {"action": "type", "selector": "#new-todo", "value": "Toggle test"},
                    {"action": "click", "selector": "#add-btn"},
                    {"action": "wait", "selector": ".todo-item"},
                    {"action": "click", "selector": ".todo-item input[type=checkbox]"},
                ],
                "assertions": [
                    {"type": "visible", "selector": ".todo-item"},
                ]
            }
        ],
        "form": [
            {
                "name": "Complete registration flow",
                "steps": [
                    {"action": "navigate", "url": "/"},
                    {"action": "type", "selector": "#first-name", "value": "John"},
                    {"action": "type", "selector": "#last-name", "value": "Doe"},
                    {"action": "type", "selector": "#email", "value": "john@example.com"},
                    {"action": "click", "selector": "#next-btn"},
                    {"action": "wait", "selector": "#password"},
                    {"action": "type", "selector": "#password", "value": "SecurePass123!"},
                    {"action": "type", "selector": "#confirm-password", "value": "SecurePass123!"},
                    {"action": "select", "selector": "#country", "value": "USA"},
                    {"action": "click", "selector": "#submit-btn"},
                ],
                "assertions": [
                    {"type": "url", "contains": "/success"},
                    {"type": "visible", "selector": "#success-message"},
                ]
            },
            {
                "name": "Password mismatch validation",
                "steps": [
                    {"action": "navigate", "url": "/step2"},
                    {"action": "type", "selector": "#password", "value": "password1"},
                    {"action": "type", "selector": "#confirm-password", "value": "password2"},
                    {"action": "click", "selector": "#submit-btn"},
                ],
                "assertions": [
                    {"type": "visible", "selector": "#error-message"},
                    {"type": "text", "selector": "#error-message", "contains": "match"},
                ]
            }
        ],
        "collaboration": [
            {
                "name": "User joins collaboration space",
                "steps": [
                    {"action": "navigate", "url": "/collab/user_a"},
                    {"action": "wait", "selector": "#editor"},
                ],
                "assertions": [
                    {"type": "visible", "selector": "#editor"},
                    {"type": "visible", "selector": "#chat"},
                ]
            }
        ]
    }

    def __init__(self, config: MockTestConfig):
        """Initialize the runner.

        Args:
            config: Mock test configuration.
        """
        self.config = config
        self.scenario_manager = ScenarioManager()
        self.server: Optional[MockServer] = None
        self._browser: Optional[Browser] = None
        self._contexts: dict[str, BrowserContext] = {}
        self._pages: dict[str, Page] = {}
        self._results: list[dict] = []

        Path(self.config.screenshots_dir).mkdir(parents=True, exist_ok=True)

    def setup(self) -> str:
        """Set up mock server and browser.

        Returns:
            Server URL.
        """
        # Load scenario
        scenario = None
        if Path(self.config.scenario).exists():
            scenario = self.scenario_manager.load_from_file(self.config.scenario)
        else:
            scenario = self.scenario_manager.get(self.config.scenario)

        # Start mock server
        server_config = MockServerConfig(
            host=self.config.host,
            port=self.config.port
        )
        self.server = MockServer(scenario=scenario, config=server_config)
        url = self.server.start(background=True)

        # Start browser
        playwright = sync_playwright().start()
        self._browser = playwright.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo
        )

        return url

    def teardown(self) -> None:
        """Clean up resources."""
        for context in self._contexts.values():
            context.close()
        if self._browser:
            self._browser.close()
        if self.server:
            self.server.stop()

    def create_user_session(self, user_id: str) -> Page:
        """Create a browser session for a user.

        Args:
            user_id: User identifier.

        Returns:
            Page object for the user.
        """
        context = self._browser.new_context(
            viewport=self.config.viewport
        )
        page = context.new_page()
        page.set_default_timeout(self.config.timeout_ms)

        self._contexts[user_id] = context
        self._pages[user_id] = page

        return page

    def run_test(self, test: dict, user_id: str = "default") -> dict:
        """Run a single test.

        Args:
            test: Test definition.
            user_id: User to run test as.

        Returns:
            Test result.
        """
        result = {
            "name": test.get("name", "Unnamed"),
            "passed": True,
            "steps": [],
            "assertions": [],
            "error": None,
            "duration_ms": 0
        }

        start_time = datetime.now()
        page = self._pages.get(user_id)

        if not page:
            page = self.create_user_session(user_id)

        base_url = self.server.get_url()

        try:
            # Execute steps
            for step in test.get("steps", []):
                step_result = self._execute_step(page, step, base_url)
                result["steps"].append(step_result)

                if not step_result["success"]:
                    result["passed"] = False
                    result["error"] = step_result.get("error")
                    break

            # Run assertions if steps passed
            if result["passed"]:
                for assertion in test.get("assertions", []):
                    assertion_result = self._check_assertion(page, assertion)
                    result["assertions"].append(assertion_result)

                    if not assertion_result["passed"]:
                        result["passed"] = False

        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)

        result["duration_ms"] = int((datetime.now() - start_time).total_seconds() * 1000)
        return result

    def _execute_step(self, page: Page, step: dict, base_url: str) -> dict:
        """Execute a test step.

        Args:
            page: Playwright page.
            step: Step definition.
            base_url: Server base URL.

        Returns:
            Step result.
        """
        result = {"action": step.get("action"), "success": True}

        try:
            action = step.get("action")

            if action == "navigate":
                url = step.get("url", "/")
                if not url.startswith("http"):
                    url = f"{base_url}{url}"
                page.goto(url)
                page.wait_for_load_state("networkidle", timeout=5000)

            elif action == "click":
                page.click(step["selector"])
                page.wait_for_load_state("networkidle", timeout=5000)

            elif action == "type":
                page.fill(step["selector"], step["value"])

            elif action == "select":
                page.select_option(step["selector"], step["value"])

            elif action == "wait":
                page.wait_for_selector(step["selector"], timeout=step.get("timeout", 5000))

            elif action == "screenshot":
                name = step.get("name", f"step_{datetime.now().strftime('%H%M%S')}")
                path = Path(self.config.screenshots_dir) / f"{name}.png"
                page.screenshot(path=str(path))
                result["screenshot"] = str(path)

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

            # Take failure screenshot
            try:
                path = Path(self.config.screenshots_dir) / f"failure_{datetime.now().strftime('%H%M%S')}.png"
                page.screenshot(path=str(path))
                result["screenshot"] = str(path)
            except Exception:
                pass

        return result

    def _check_assertion(self, page: Page, assertion: dict) -> dict:
        """Check an assertion.

        Args:
            page: Playwright page.
            assertion: Assertion definition.

        Returns:
            Assertion result.
        """
        result = {"type": assertion.get("type"), "passed": True}

        try:
            assertion_type = assertion.get("type")

            if assertion_type == "url":
                actual = page.url
                expected = assertion.get("contains", "")
                result["passed"] = expected in actual
                result["actual"] = actual
                result["expected"] = expected

            elif assertion_type == "visible":
                selector = assertion["selector"]
                result["passed"] = page.locator(selector).is_visible()
                result["selector"] = selector

            elif assertion_type == "text":
                selector = assertion["selector"]
                actual = page.locator(selector).text_content() or ""
                expected = assertion.get("contains", "")
                result["passed"] = expected in actual
                result["actual"] = actual
                result["expected"] = expected

            elif assertion_type == "count":
                selector = assertion["selector"]
                actual = page.locator(selector).count()
                expected = assertion.get("count", 0)
                result["passed"] = actual == expected
                result["actual"] = actual
                result["expected"] = expected

        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)

        return result

    def run_scenario_tests(self) -> list[dict]:
        """Run all tests for the current scenario.

        Returns:
            List of test results.
        """
        scenario_name = self.config.scenario
        tests = self.BUILT_IN_TESTS.get(scenario_name, [])

        results = []
        for test in tests:
            logger.info(f"Running: {test['name']}")
            result = self.run_test(test)
            results.append(result)

            status = "PASS" if result["passed"] else "FAIL"
            logger.info(f"  {status} ({result['duration_ms']}ms)")

        self._results = results
        return results

    def run_custom_tests(self, tests_file: str) -> list[dict]:
        """Run tests from a JSON file.

        Args:
            tests_file: Path to tests JSON.

        Returns:
            List of test results.
        """
        with open(tests_file) as f:
            tests = json.load(f)

        results = []
        for test in tests:
            logger.info(f"Running: {test.get('name', 'Unnamed')}")
            result = self.run_test(test)
            results.append(result)

            status = "PASS" if result["passed"] else "FAIL"
            logger.info(f"  {status} ({result['duration_ms']}ms)")

        self._results = results
        return results

    def explore_scenario(self) -> dict:
        """Automatically explore the mock scenario.

        Returns:
            Exploration results.
        """
        from ..exploration.crawler import Crawler
        from ..core.models import ExplorationConfig

        config = ExplorationConfig(
            headless=self.config.headless,
            viewport=self.config.viewport,
            max_depth=5,
            max_states=20,
            max_actions_per_state=5
        )

        crawler = Crawler(
            config=config,
            screenshots_dir=self.config.screenshots_dir
        )

        base_url = self.server.get_url()
        graph = crawler.explore(base_url)

        return {
            "states": graph.state_count,
            "transitions": graph.transition_count,
            "graph": graph
        }

    def get_summary(self) -> dict:
        """Get test run summary.

        Returns:
            Summary statistics.
        """
        if not self._results:
            return {"total": 0, "passed": 0, "failed": 0}

        passed = sum(1 for r in self._results if r["passed"])
        failed = len(self._results) - passed

        return {
            "total": len(self._results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self._results) if self._results else 0,
            "results": self._results
        }

    def save_results(self, path: str) -> None:
        """Save test results to file.

        Args:
            path: Output file path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            json.dump({
                "scenario": self.config.scenario,
                "run_at": datetime.now().isoformat(),
                "summary": self.get_summary()
            }, f, indent=2)


def run_mock_tests(config: MockTestConfig) -> dict:
    """Run mock tests and return results.

    Args:
        config: Test configuration.

    Returns:
        Test summary.
    """
    runner = MockTestRunner(config)

    try:
        url = runner.setup()
        logger.info(f"Mock server started at: {url}")
        logger.info(f"Running scenario: {config.scenario}")

        if config.auto_explore:
            exploration = runner.explore_scenario()
            logger.info(f"Explored {exploration['states']} states, {exploration['transitions']} transitions")

        results = runner.run_scenario_tests()
        summary = runner.get_summary()

        logger.info(f"\nResults: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1%})")

        return summary

    finally:
        runner.teardown()


def run_interactive_mock(config: MockTestConfig) -> None:
    """Run interactive mock testing session.

    Args:
        config: Test configuration.
    """
    runner = MockTestRunner(config)

    print(f"\n{'='*60}")
    print("AutoQA Mock Testing - Interactive Mode")
    print(f"{'='*60}")
    print(f"Scenario: {config.scenario}")
    print("\nAvailable scenarios: login, todo, form, collaboration")
    print("\nCommands:")
    print("  run                     - Run all scenario tests")
    print("  run <test_name>         - Run specific test")
    print("  explore                 - Auto-explore the app")
    print("  list                    - List available tests")
    print("  reset                   - Reset server state")
    print("  url                     - Show server URL")
    print("  scenario <name>         - Switch scenario")
    print("  results                 - Show test results")
    print("  save <path>             - Save results")
    print("  quit                    - Exit")
    print(f"{'='*60}\n")

    try:
        url = runner.setup()
        print(f"Mock server running at: {url}\n")

        while True:
            try:
                cmd = input("mock> ").strip()
                if not cmd:
                    continue

                parts = cmd.split(maxsplit=1)
                action = parts[0].lower()

                if action == "quit" or action == "exit":
                    break

                elif action == "run":
                    if len(parts) > 1:
                        test_name = parts[1]
                        tests = runner.BUILT_IN_TESTS.get(config.scenario, [])
                        test = next((t for t in tests if t["name"] == test_name), None)
                        if test:
                            result = runner.run_test(test)
                            status = "PASS" if result["passed"] else "FAIL"
                            print(f"{status}: {result['name']} ({result['duration_ms']}ms)")
                        else:
                            print(f"Test not found: {test_name}")
                    else:
                        results = runner.run_scenario_tests()
                        summary = runner.get_summary()
                        print(f"\n{summary['passed']}/{summary['total']} passed")

                elif action == "explore":
                    print("Exploring...")
                    exploration = runner.explore_scenario()
                    print(f"Found {exploration['states']} states, {exploration['transitions']} transitions")

                elif action == "list":
                    tests = runner.BUILT_IN_TESTS.get(config.scenario, [])
                    print(f"\nTests for '{config.scenario}':")
                    for test in tests:
                        print(f"  - {test['name']}")

                elif action == "reset":
                    runner.server.reset()
                    print("Server state reset")

                elif action == "url":
                    print(f"Server URL: {runner.server.get_url()}")

                elif action == "scenario" and len(parts) > 1:
                    new_scenario = parts[1]
                    if new_scenario in runner.BUILT_IN_TESTS:
                        config.scenario = new_scenario
                        print(f"Switched to scenario: {new_scenario}")
                    else:
                        print(f"Unknown scenario: {new_scenario}")
                        print(f"Available: {', '.join(runner.BUILT_IN_TESTS.keys())}")

                elif action == "results":
                    summary = runner.get_summary()
                    print(f"\nTotal: {summary['total']}, Passed: {summary['passed']}, Failed: {summary['failed']}")
                    for r in summary.get("results", []):
                        status = "PASS" if r["passed"] else "FAIL"
                        print(f"  [{status}] {r['name']}")

                elif action == "save" and len(parts) > 1:
                    runner.save_results(parts[1])
                    print(f"Results saved to: {parts[1]}")

                else:
                    print(f"Unknown command: {action}")

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")

    finally:
        runner.teardown()
        print("\nMock server stopped.")
