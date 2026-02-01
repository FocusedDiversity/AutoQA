"""Playwright-based UI crawler for state discovery."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from ..core.models import (
    Action,
    ActionType,
    ExplorationConfig,
    InteractiveElement,
)
from ..state.graph import StateGraph

logger = logging.getLogger(__name__)


class Crawler:
    """Playwright-based UI crawler for discovering application states.

    Systematically explores a web application by:
    1. Detecting interactive elements on each page
    2. Executing actions and capturing resulting states
    3. Building a state graph of the application
    """

    # Selectors for interactive elements
    INTERACTIVE_SELECTORS = [
        "button",
        "a[href]",
        "input",
        "select",
        "textarea",
        "[role='button']",
        "[role='link']",
        "[role='menuitem']",
        "[role='tab']",
        "[onclick]",
        "[data-testid]",
    ]

    def __init__(
        self,
        config: ExplorationConfig,
        state_graph: Optional[StateGraph] = None,
        screenshots_dir: Optional[str] = None
    ):
        """Initialize the crawler.

        Args:
            config: Exploration configuration.
            state_graph: State graph to populate. Creates new if not provided.
            screenshots_dir: Directory for screenshots.
        """
        self.config = config
        self.state_graph = state_graph or StateGraph()
        self.screenshots_dir = Path(screenshots_dir or "screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._start_time: Optional[datetime] = None
        self._on_state_discovered: Optional[Callable] = None

    def set_callback(self, on_state_discovered: Callable[[str, bool], None]) -> None:
        """Set callback for state discovery events.

        Args:
            on_state_discovered: Callback(state_id, is_new).
        """
        self._on_state_discovered = on_state_discovered

    def explore(
        self,
        start_url: str,
        user_context: str = "default",
        storage_state: Optional[str] = None
    ) -> StateGraph:
        """Explore an application starting from a URL.

        Args:
            start_url: URL to start exploration from.
            user_context: User context identifier.
            storage_state: Path to browser storage state (for auth).

        Returns:
            Populated state graph.
        """
        self._start_time = datetime.now()

        with sync_playwright() as playwright:
            self._browser = playwright.chromium.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo
            )

            self._context = self._browser.new_context(
                viewport=self.config.viewport,
                storage_state=storage_state
            )

            page = self._context.new_page()
            page.goto(start_url)

            # Capture initial state
            initial_state_id = self._capture_state(page, user_context, depth=0)

            # Start exploration
            self._explore_from_state(page, initial_state_id, user_context, depth=0)

            self._browser.close()

        return self.state_graph

    def _explore_from_state(
        self,
        page: Page,
        current_state_id: str,
        user_context: str,
        depth: int
    ) -> None:
        """Recursively explore from a state.

        Args:
            page: Playwright page.
            current_state_id: Current state ID.
            user_context: User context.
            depth: Current exploration depth.
        """
        # Check bounds
        if depth >= self.config.max_depth:
            logger.debug(f"Max depth {self.config.max_depth} reached")
            return

        if self.state_graph.state_count >= self.config.max_states:
            logger.info(f"Max states {self.config.max_states} reached")
            return

        if self._is_timeout():
            logger.info("Exploration timeout reached")
            return

        # Discover interactive elements
        elements = self._discover_elements(page)
        logger.debug(f"Found {len(elements)} interactive elements at depth {depth}")

        # Try each element
        actions_tried = 0
        for element in elements:
            if actions_tried >= self.config.max_actions_per_state:
                break

            if self._should_skip_element(element):
                continue

            # Execute action
            action = self._create_action(element, user_context)
            success = self._execute_action(page, element, action)

            if success:
                actions_tried += 1

                # Capture new state
                new_state_id, is_new = self._capture_state_after_action(
                    page, user_context, depth + 1
                )

                # Record transition
                self.state_graph.add_transition(
                    current_state_id,
                    new_state_id,
                    action
                )

                # Explore new state if it's actually new
                if is_new:
                    self._explore_from_state(page, new_state_id, user_context, depth + 1)

                # Go back to current state
                page.go_back()
                page.wait_for_load_state("networkidle", timeout=5000)

    def _discover_elements(self, page: Page) -> list[InteractiveElement]:
        """Discover interactive elements on the current page.

        Args:
            page: Playwright page.

        Returns:
            List of interactive elements.
        """
        elements = []

        try:
            raw_elements = page.evaluate("""() => {
                const selectors = %s;
                const results = [];

                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0 && rect.top < window.innerHeight) {
                            // Generate a stable selector
                            let stableSelector = '';
                            if (el.getAttribute('data-testid')) {
                                stableSelector = `[data-testid="${el.getAttribute('data-testid')}"]`;
                            } else if (el.id) {
                                stableSelector = `#${el.id}`;
                            } else if (el.getAttribute('aria-label')) {
                                stableSelector = `[aria-label="${el.getAttribute('aria-label')}"]`;
                            } else {
                                // Fallback to nth-child path
                                let path = [];
                                let current = el;
                                while (current && current !== document.body) {
                                    const parent = current.parentElement;
                                    if (parent) {
                                        const index = Array.from(parent.children).indexOf(current);
                                        path.unshift(`${current.tagName.toLowerCase()}:nth-child(${index + 1})`);
                                    }
                                    current = parent;
                                }
                                stableSelector = path.join(' > ');
                            }

                            results.push({
                                selector: stableSelector,
                                tag: el.tagName.toLowerCase(),
                                element_type: el.type || null,
                                text: (el.textContent || '').trim().slice(0, 100),
                                role: el.getAttribute('role'),
                                visible: true,
                                rect: {
                                    x: rect.x,
                                    y: rect.y,
                                    width: rect.width,
                                    height: rect.height
                                }
                            });
                        }
                    });
                });

                return results;
            }""" % str(self.INTERACTIVE_SELECTORS))

            for el in raw_elements:
                elements.append(InteractiveElement(**el))

        except Exception as e:
            logger.warning(f"Element discovery failed: {e}")

        return elements

    def _should_skip_element(self, element: InteractiveElement) -> bool:
        """Check if an element should be skipped.

        Args:
            element: Element to check.

        Returns:
            True if element should be skipped.
        """
        # Skip elements matching ignored selectors
        for ignored in self.config.ignored_selectors:
            if ignored in element.selector:
                return True

        # Skip external links
        if element.tag == "a" and element.text:
            text_lower = element.text.lower()
            if any(skip in text_lower for skip in ["logout", "sign out", "external"]):
                return True

        return False

    def _create_action(self, element: InteractiveElement, user: str) -> Action:
        """Create an action for an element.

        Args:
            element: Target element.
            user: User context.

        Returns:
            Action to execute.
        """
        action_type = ActionType.CLICK

        if element.tag == "input":
            if element.element_type in ["text", "email", "password", "search", "tel", "url"]:
                action_type = ActionType.TYPE
        elif element.tag == "select":
            action_type = ActionType.SELECT
        elif element.tag == "textarea":
            action_type = ActionType.TYPE

        return Action(
            action_type=action_type,
            target={
                "selector": element.selector,
                "text": element.text,
                "tag": element.tag
            },
            user=user,
            value=self._generate_test_value(element) if action_type == ActionType.TYPE else None
        )

    def _generate_test_value(self, element: InteractiveElement) -> str:
        """Generate test data for input elements.

        Args:
            element: Input element.

        Returns:
            Test value string.
        """
        element_type = element.element_type or "text"

        test_values = {
            "email": "test@example.com",
            "password": "TestPassword123!",
            "tel": "555-123-4567",
            "url": "https://example.com",
            "number": "42",
            "search": "test search",
            "text": "Test input value"
        }

        return test_values.get(element_type, "Test value")

    def _execute_action(self, page: Page, element: InteractiveElement, action: Action) -> bool:
        """Execute an action on an element.

        Args:
            page: Playwright page.
            element: Target element.
            action: Action to execute.

        Returns:
            True if action succeeded.
        """
        try:
            locator = page.locator(element.selector).first

            if action.action_type == ActionType.CLICK:
                locator.click(timeout=5000)
            elif action.action_type == ActionType.TYPE:
                locator.fill(action.value or "", timeout=5000)
            elif action.action_type == ActionType.SELECT:
                # Select first option
                options = locator.locator("option").all_text_contents()
                if options:
                    locator.select_option(options[0], timeout=5000)

            page.wait_for_load_state("networkidle", timeout=5000)
            return True

        except Exception as e:
            logger.debug(f"Action failed on {element.selector}: {e}")
            return False

    def _capture_state(
        self,
        page: Page,
        user_context: str,
        depth: int
    ) -> str:
        """Capture the current page state.

        Args:
            page: Playwright page.
            user_context: User context.
            depth: Discovery depth.

        Returns:
            State ID.
        """
        url = page.url
        dom_content = page.content()

        # Take screenshot
        screenshot_path = self.screenshots_dir / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        page.screenshot(path=str(screenshot_path))

        state_id, is_new = self.state_graph.add_state(
            url=url,
            dom_content=dom_content,
            screenshot_path=str(screenshot_path),
            user_context=user_context,
            discovery_depth=depth
        )

        # Update state with interactive elements
        if is_new:
            state = self.state_graph.get_state(state_id)
            if state:
                state.interactive_elements = self._discover_elements(page)

        if self._on_state_discovered:
            self._on_state_discovered(state_id, is_new)

        return state_id

    def _capture_state_after_action(
        self,
        page: Page,
        user_context: str,
        depth: int
    ) -> tuple[str, bool]:
        """Capture state after an action.

        Args:
            page: Playwright page.
            user_context: User context.
            depth: Discovery depth.

        Returns:
            Tuple of (state_id, is_new).
        """
        url = page.url
        dom_content = page.content()

        screenshot_path = self.screenshots_dir / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        page.screenshot(path=str(screenshot_path))

        state_id, is_new = self.state_graph.add_state(
            url=url,
            dom_content=dom_content,
            screenshot_path=str(screenshot_path),
            user_context=user_context,
            discovery_depth=depth
        )

        if self._on_state_discovered:
            self._on_state_discovered(state_id, is_new)

        return (state_id, is_new)

    def _is_timeout(self) -> bool:
        """Check if exploration has timed out.

        Returns:
            True if timeout exceeded.
        """
        if not self._start_time:
            return False

        elapsed = (datetime.now() - self._start_time).total_seconds() / 60
        return elapsed >= self.config.timeout_minutes
