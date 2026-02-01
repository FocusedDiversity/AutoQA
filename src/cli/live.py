"""Live web application testing CLI."""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


@dataclass
class LiveTestConfig:
    """Configuration for live testing."""
    base_url: str
    headless: bool = True
    slow_mo: int = 0
    timeout_ms: int = 30000
    viewport: dict = None
    storage_state: Optional[str] = None
    screenshots_dir: str = "screenshots/live"
    video_dir: Optional[str] = None
    trace_dir: Optional[str] = None
    user_agent: Optional[str] = None

    def __post_init__(self):
        if self.viewport is None:
            self.viewport = {"width": 1280, "height": 720}


class LiveTestRunner:
    """Run tests against live web applications.

    Features:
    - Interactive exploration mode
    - Record and replay
    - Screenshot on each action
    - Video recording
    - Trace capture for debugging
    """

    def __init__(self, config: LiveTestConfig):
        """Initialize the runner.

        Args:
            config: Live test configuration.
        """
        self.config = config
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._actions: list[dict] = []
        self._screenshots: list[str] = []

        # Ensure directories exist
        Path(self.config.screenshots_dir).mkdir(parents=True, exist_ok=True)
        if self.config.video_dir:
            Path(self.config.video_dir).mkdir(parents=True, exist_ok=True)
        if self.config.trace_dir:
            Path(self.config.trace_dir).mkdir(parents=True, exist_ok=True)

    def start(self) -> str:
        """Start the browser and navigate to base URL.

        Returns:
            Current page URL.
        """
        playwright = sync_playwright().start()

        self._browser = playwright.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo
        )

        context_options = {
            "viewport": self.config.viewport,
            "storage_state": self.config.storage_state
        }

        if self.config.video_dir:
            context_options["record_video_dir"] = self.config.video_dir

        if self.config.user_agent:
            context_options["user_agent"] = self.config.user_agent

        self._context = self._browser.new_context(**context_options)

        if self.config.trace_dir:
            self._context.tracing.start(screenshots=True, snapshots=True)

        self._page = self._context.new_page()
        self._page.set_default_timeout(self.config.timeout_ms)

        self._page.goto(self.config.base_url)
        self._record_action("navigate", {"url": self.config.base_url})

        return self._page.url

    def stop(self) -> dict:
        """Stop the browser and return session summary.

        Returns:
            Session summary with actions and screenshots.
        """
        if self.config.trace_dir and self._context:
            trace_path = Path(self.config.trace_dir) / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            self._context.tracing.stop(path=str(trace_path))

        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()

        return {
            "actions": self._actions,
            "screenshots": self._screenshots,
            "total_actions": len(self._actions)
        }

    def click(self, selector: str, screenshot: bool = True) -> bool:
        """Click an element.

        Args:
            selector: CSS selector.
            screenshot: Take screenshot after.

        Returns:
            True if successful.
        """
        try:
            self._page.click(selector)
            self._page.wait_for_load_state("networkidle", timeout=5000)
            self._record_action("click", {"selector": selector})

            if screenshot:
                self._take_screenshot(f"click_{len(self._actions)}")

            return True
        except Exception as e:
            logger.error(f"Click failed on {selector}: {e}")
            self._take_screenshot(f"error_click_{len(self._actions)}")
            return False

    def type_text(self, selector: str, text: str, screenshot: bool = True) -> bool:
        """Type text into an input.

        Args:
            selector: CSS selector.
            text: Text to type.
            screenshot: Take screenshot after.

        Returns:
            True if successful.
        """
        try:
            self._page.fill(selector, text)
            self._record_action("type", {"selector": selector, "text": text})

            if screenshot:
                self._take_screenshot(f"type_{len(self._actions)}")

            return True
        except Exception as e:
            logger.error(f"Type failed on {selector}: {e}")
            return False

    def select(self, selector: str, value: str, screenshot: bool = True) -> bool:
        """Select an option from a dropdown.

        Args:
            selector: CSS selector.
            value: Option value to select.
            screenshot: Take screenshot after.

        Returns:
            True if successful.
        """
        try:
            self._page.select_option(selector, value)
            self._record_action("select", {"selector": selector, "value": value})

            if screenshot:
                self._take_screenshot(f"select_{len(self._actions)}")

            return True
        except Exception as e:
            logger.error(f"Select failed on {selector}: {e}")
            return False

    def navigate(self, url: str, screenshot: bool = True) -> bool:
        """Navigate to a URL.

        Args:
            url: URL to navigate to.
            screenshot: Take screenshot after.

        Returns:
            True if successful.
        """
        try:
            if not url.startswith("http"):
                url = f"{self.config.base_url.rstrip('/')}/{url.lstrip('/')}"

            self._page.goto(url)
            self._page.wait_for_load_state("networkidle", timeout=10000)
            self._record_action("navigate", {"url": url})

            if screenshot:
                self._take_screenshot(f"navigate_{len(self._actions)}")

            return True
        except Exception as e:
            logger.error(f"Navigation failed to {url}: {e}")
            return False

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> bool:
        """Wait for an element to appear.

        Args:
            selector: CSS selector.
            timeout_ms: Timeout in milliseconds.

        Returns:
            True if element appeared.
        """
        try:
            self._page.wait_for_selector(selector, timeout=timeout_ms)
            self._record_action("wait", {"selector": selector})
            return True
        except Exception as e:
            logger.warning(f"Wait timed out for {selector}")
            return False

    def get_text(self, selector: str) -> Optional[str]:
        """Get text content of an element.

        Args:
            selector: CSS selector.

        Returns:
            Text content or None.
        """
        try:
            return self._page.locator(selector).text_content()
        except Exception as e:
            logger.warning(f"Failed to get text from {selector}: {e}")
            return None

    def get_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """Get an attribute value.

        Args:
            selector: CSS selector.
            attribute: Attribute name.

        Returns:
            Attribute value or None.
        """
        try:
            return self._page.locator(selector).get_attribute(attribute)
        except Exception as e:
            logger.warning(f"Failed to get attribute {attribute} from {selector}: {e}")
            return None

    def is_visible(self, selector: str) -> bool:
        """Check if an element is visible.

        Args:
            selector: CSS selector.

        Returns:
            True if visible.
        """
        try:
            return self._page.locator(selector).is_visible()
        except Exception:
            return False

    def get_elements(self, selector: str) -> list[dict]:
        """Get information about all matching elements.

        Args:
            selector: CSS selector.

        Returns:
            List of element info dictionaries.
        """
        elements = []
        try:
            locators = self._page.locator(selector).all()
            for i, loc in enumerate(locators):
                elements.append({
                    "index": i,
                    "text": loc.text_content(),
                    "tag": loc.evaluate("el => el.tagName.toLowerCase()"),
                    "visible": loc.is_visible()
                })
        except Exception as e:
            logger.warning(f"Failed to get elements for {selector}: {e}")

        return elements

    def screenshot(self, name: Optional[str] = None) -> str:
        """Take a screenshot.

        Args:
            name: Optional screenshot name.

        Returns:
            Screenshot path.
        """
        return self._take_screenshot(name or f"manual_{len(self._screenshots)}")

    def current_url(self) -> str:
        """Get current page URL.

        Returns:
            Current URL.
        """
        return self._page.url

    def current_title(self) -> str:
        """Get current page title.

        Returns:
            Page title.
        """
        return self._page.title()

    def execute_script(self, script: str) -> Any:
        """Execute JavaScript on the page.

        Args:
            script: JavaScript code.

        Returns:
            Script result.
        """
        try:
            result = self._page.evaluate(script)
            self._record_action("script", {"script": script[:100]})
            return result
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return None

    def assert_text(self, selector: str, expected: str) -> bool:
        """Assert element contains expected text.

        Args:
            selector: CSS selector.
            expected: Expected text.

        Returns:
            True if assertion passes.
        """
        actual = self.get_text(selector)
        passed = actual is not None and expected in actual
        self._record_action("assert", {
            "type": "text",
            "selector": selector,
            "expected": expected,
            "actual": actual,
            "passed": passed
        })
        return passed

    def assert_visible(self, selector: str) -> bool:
        """Assert element is visible.

        Args:
            selector: CSS selector.

        Returns:
            True if visible.
        """
        passed = self.is_visible(selector)
        self._record_action("assert", {
            "type": "visible",
            "selector": selector,
            "passed": passed
        })
        return passed

    def assert_url(self, expected: str) -> bool:
        """Assert current URL matches expected.

        Args:
            expected: Expected URL or pattern.

        Returns:
            True if matches.
        """
        actual = self.current_url()
        passed = expected in actual
        self._record_action("assert", {
            "type": "url",
            "expected": expected,
            "actual": actual,
            "passed": passed
        })
        return passed

    def _record_action(self, action_type: str, details: dict) -> None:
        """Record an action.

        Args:
            action_type: Type of action.
            details: Action details.
        """
        self._actions.append({
            "type": action_type,
            "timestamp": datetime.now().isoformat(),
            "url": self._page.url if self._page else None,
            **details
        })

    def _take_screenshot(self, name: str) -> str:
        """Take and save a screenshot.

        Args:
            name: Screenshot name.

        Returns:
            Screenshot path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.config.screenshots_dir) / f"{name}_{timestamp}.png"
        self._page.screenshot(path=str(path))
        self._screenshots.append(str(path))
        return str(path)

    def save_actions(self, path: str) -> None:
        """Save recorded actions to file.

        Args:
            path: Output file path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            json.dump({
                "base_url": self.config.base_url,
                "recorded_at": datetime.now().isoformat(),
                "actions": self._actions
            }, f, indent=2)

        logger.info(f"Actions saved to {path}")

    def replay_actions(self, actions_file: str) -> dict:
        """Replay recorded actions.

        Args:
            actions_file: Path to actions JSON file.

        Returns:
            Replay results.
        """
        with open(actions_file) as f:
            data = json.load(f)

        results = {"passed": 0, "failed": 0, "errors": []}

        for action in data.get("actions", []):
            action_type = action.get("type")
            try:
                if action_type == "navigate":
                    self.navigate(action["url"], screenshot=False)
                elif action_type == "click":
                    self.click(action["selector"], screenshot=False)
                elif action_type == "type":
                    self.type_text(action["selector"], action["text"], screenshot=False)
                elif action_type == "select":
                    self.select(action["selector"], action["value"], screenshot=False)
                elif action_type == "wait":
                    self.wait_for(action["selector"])

                results["passed"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "action": action,
                    "error": str(e)
                })

        return results


def run_interactive(config: LiveTestConfig) -> None:
    """Run interactive live testing session.

    Args:
        config: Test configuration.
    """
    runner = LiveTestRunner(config)

    print(f"\n{'='*60}")
    print("AutoQA Live Testing - Interactive Mode")
    print(f"{'='*60}")
    print(f"Target: {config.base_url}")
    print("\nCommands:")
    print("  click <selector>        - Click an element")
    print("  type <selector> <text>  - Type text into input")
    print("  select <selector> <val> - Select dropdown option")
    print("  goto <url>              - Navigate to URL")
    print("  wait <selector>         - Wait for element")
    print("  text <selector>         - Get element text")
    print("  screenshot [name]       - Take screenshot")
    print("  elements <selector>     - List matching elements")
    print("  url                     - Show current URL")
    print("  assert-text <sel> <txt> - Assert text content")
    print("  assert-visible <sel>    - Assert element visible")
    print("  save <path>             - Save recorded actions")
    print("  replay <path>           - Replay saved actions")
    print("  quit                    - Exit session")
    print(f"{'='*60}\n")

    try:
        url = runner.start()
        print(f"Started at: {url}\n")

        while True:
            try:
                cmd = input("autoqa> ").strip()
                if not cmd:
                    continue

                parts = cmd.split(maxsplit=2)
                action = parts[0].lower()

                if action == "quit" or action == "exit":
                    break

                elif action == "click" and len(parts) >= 2:
                    if runner.click(parts[1]):
                        print(f"Clicked: {parts[1]}")
                    else:
                        print("Click failed")

                elif action == "type" and len(parts) >= 3:
                    if runner.type_text(parts[1], parts[2]):
                        print(f"Typed into: {parts[1]}")
                    else:
                        print("Type failed")

                elif action == "select" and len(parts) >= 3:
                    if runner.select(parts[1], parts[2]):
                        print(f"Selected: {parts[2]}")
                    else:
                        print("Select failed")

                elif action == "goto" and len(parts) >= 2:
                    if runner.navigate(parts[1]):
                        print(f"Navigated to: {runner.current_url()}")
                    else:
                        print("Navigation failed")

                elif action == "wait" and len(parts) >= 2:
                    if runner.wait_for(parts[1]):
                        print(f"Found: {parts[1]}")
                    else:
                        print("Wait timed out")

                elif action == "text" and len(parts) >= 2:
                    text = runner.get_text(parts[1])
                    print(f"Text: {text}")

                elif action == "screenshot":
                    name = parts[1] if len(parts) >= 2 else None
                    path = runner.screenshot(name)
                    print(f"Screenshot saved: {path}")

                elif action == "elements" and len(parts) >= 2:
                    elements = runner.get_elements(parts[1])
                    for el in elements:
                        print(f"  [{el['index']}] <{el['tag']}> {el['text'][:50] if el['text'] else ''}")

                elif action == "url":
                    print(f"URL: {runner.current_url()}")
                    print(f"Title: {runner.current_title()}")

                elif action == "assert-text" and len(parts) >= 3:
                    if runner.assert_text(parts[1], parts[2]):
                        print("PASS: Text assertion")
                    else:
                        print("FAIL: Text assertion")

                elif action == "assert-visible" and len(parts) >= 2:
                    if runner.assert_visible(parts[1]):
                        print("PASS: Visibility assertion")
                    else:
                        print("FAIL: Visibility assertion")

                elif action == "save" and len(parts) >= 2:
                    runner.save_actions(parts[1])

                elif action == "replay" and len(parts) >= 2:
                    results = runner.replay_actions(parts[1])
                    print(f"Replay: {results['passed']} passed, {results['failed']} failed")

                else:
                    print(f"Unknown command: {action}")

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")

    finally:
        summary = runner.stop()
        print(f"\nSession ended. {summary['total_actions']} actions recorded.")
