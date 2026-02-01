"""Selector generation and optimization utilities."""

import re
from enum import Enum
from typing import Optional


class SelectorStrategy(Enum):
    """Selector generation strategy."""
    DATA_TESTID = "data-testid"
    ID = "id"
    ARIA_LABEL = "aria-label"
    CSS_PATH = "css-path"
    XPATH = "xpath"
    TEXT = "text"


class SelectorGenerator:
    """Generates stable, maintainable CSS selectors.

    Prioritizes:
    1. data-testid attributes (most stable)
    2. Unique IDs
    3. aria-label for accessibility
    4. CSS path as fallback
    """

    # Priority order for selector strategies
    STRATEGY_PRIORITY = [
        SelectorStrategy.DATA_TESTID,
        SelectorStrategy.ID,
        SelectorStrategy.ARIA_LABEL,
        SelectorStrategy.CSS_PATH,
    ]

    def __init__(self, preferred_strategy: Optional[SelectorStrategy] = None):
        """Initialize the generator.

        Args:
            preferred_strategy: Preferred strategy to use when available.
        """
        self.preferred_strategy = preferred_strategy

    def generate(self, element_info: dict) -> str:
        """Generate the best selector for an element.

        Args:
            element_info: Dictionary with element attributes.

        Returns:
            CSS selector string.
        """
        # Try strategies in priority order
        strategies = self.STRATEGY_PRIORITY.copy()

        # Move preferred strategy to front
        if self.preferred_strategy and self.preferred_strategy in strategies:
            strategies.remove(self.preferred_strategy)
            strategies.insert(0, self.preferred_strategy)

        for strategy in strategies:
            selector = self._try_strategy(strategy, element_info)
            if selector:
                return selector

        # Ultimate fallback
        return self._generate_css_path(element_info)

    def _try_strategy(self, strategy: SelectorStrategy, info: dict) -> Optional[str]:
        """Try a specific selector strategy.

        Args:
            strategy: Strategy to try.
            info: Element information.

        Returns:
            Selector string or None.
        """
        if strategy == SelectorStrategy.DATA_TESTID:
            testid = info.get("data-testid") or info.get("testid")
            if testid:
                return f'[data-testid="{testid}"]'

        elif strategy == SelectorStrategy.ID:
            element_id = info.get("id")
            if element_id and self._is_stable_id(element_id):
                return f"#{element_id}"

        elif strategy == SelectorStrategy.ARIA_LABEL:
            aria_label = info.get("aria-label")
            if aria_label:
                return f'[aria-label="{self._escape_attr(aria_label)}"]'

        elif strategy == SelectorStrategy.TEXT:
            text = info.get("text")
            tag = info.get("tag", "*")
            if text and len(text) < 50:
                return f'{tag}:has-text("{self._escape_attr(text)}")'

        return None

    def _is_stable_id(self, element_id: str) -> bool:
        """Check if an ID is likely stable (not auto-generated).

        Args:
            element_id: ID to check.

        Returns:
            True if ID appears stable.
        """
        # Skip IDs that look auto-generated
        auto_patterns = [
            r"^[a-f0-9]{8,}$",  # UUID-like
            r"^\d+$",  # Numeric only
            r"^:[a-z]+\d+$",  # React-style :r0
            r".*_\d{10,}.*",  # Timestamp-like
            r"^ember\d+$",  # Ember.js
            r"^ng-\d+$",  # Angular
        ]

        for pattern in auto_patterns:
            if re.match(pattern, element_id, re.IGNORECASE):
                return False

        return True

    def _generate_css_path(self, info: dict) -> str:
        """Generate a CSS path selector.

        Args:
            info: Element information.

        Returns:
            CSS path selector.
        """
        if "selector" in info:
            return info["selector"]

        tag = info.get("tag", "div")
        classes = info.get("classes", [])

        if classes:
            # Use first meaningful class
            meaningful_classes = [
                c for c in classes
                if not re.match(r"^(css|sc|styled)-", c)  # Skip generated classes
            ]
            if meaningful_classes:
                return f"{tag}.{meaningful_classes[0]}"

        return tag

    def _escape_attr(self, value: str) -> str:
        """Escape a value for use in attribute selector.

        Args:
            value: Value to escape.

        Returns:
            Escaped value.
        """
        return value.replace('"', '\\"').replace("\n", " ").strip()

    def optimize_selector(self, selector: str) -> str:
        """Optimize a selector for stability and performance.

        Args:
            selector: Original selector.

        Returns:
            Optimized selector.
        """
        # Remove overly specific paths
        if " > " in selector:
            parts = selector.split(" > ")
            # Keep last 3 parts at most
            if len(parts) > 3:
                selector = " > ".join(parts[-3:])

        # Prefer ID over class chains
        if "#" not in selector and "." in selector:
            # Keep only meaningful classes
            parts = selector.split(".")
            meaningful = [p for p in parts if not re.match(r"^[a-z]{2,3}-[a-f0-9]+", p)]
            if meaningful:
                selector = ".".join(meaningful)

        return selector

    def validate_selector(self, selector: str) -> dict:
        """Validate a selector and provide recommendations.

        Args:
            selector: Selector to validate.

        Returns:
            Validation result with recommendations.
        """
        result = {
            "valid": True,
            "stability": "high",
            "warnings": [],
            "recommendations": []
        }

        # Check for fragile patterns
        if ":nth-child" in selector:
            result["stability"] = "low"
            result["warnings"].append("nth-child selectors are fragile")
            result["recommendations"].append("Use data-testid instead")

        if len(selector.split(" > ")) > 5:
            result["stability"] = "medium"
            result["warnings"].append("Deep CSS path may break on layout changes")

        if re.search(r"\.[a-z]{2,3}-[a-f0-9]{6,}", selector):
            result["stability"] = "low"
            result["warnings"].append("Contains CSS-in-JS generated class names")

        if selector.startswith("body > "):
            result["warnings"].append("Selector starts from body, very fragile")
            result["stability"] = "low"

        return result
