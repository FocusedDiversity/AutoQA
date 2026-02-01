"""LLM-guided navigation for intelligent exploration."""

import logging
from typing import Optional

from ..core.models import InteractiveElement, ExplorationConfig
from ..llm.engine import LLMEngine
from ..state.graph import StateGraph

logger = logging.getLogger(__name__)


class Navigator:
    """LLM-guided navigator for intelligent exploration decisions.

    Uses LLM to:
    - Prioritize which elements to interact with
    - Decide when exploration is complete
    - Guide exploration toward user stories
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        state_graph: StateGraph,
        config: ExplorationConfig
    ):
        """Initialize the navigator.

        Args:
            llm_engine: LLM engine for decision making.
            state_graph: State graph being explored.
            config: Exploration configuration.
        """
        self.llm = llm_engine
        self.state_graph = state_graph
        self.config = config
        self._user_stories: list[str] = []

    def set_user_stories(self, stories: list[str]) -> None:
        """Set user stories to guide exploration.

        Args:
            stories: List of user story descriptions.
        """
        self._user_stories = stories

    def prioritize_elements(
        self,
        elements: list[InteractiveElement],
        current_url: str,
        exploration_depth: int
    ) -> list[InteractiveElement]:
        """Prioritize which elements to interact with.

        Args:
            elements: List of discovered elements.
            current_url: Current page URL.
            exploration_depth: Current depth.

        Returns:
            Prioritized list of elements.
        """
        if not elements:
            return []

        # Build element summary
        element_summary = "\n".join([
            f"{i+1}. [{e.tag}] {e.text or e.selector[:50]}"
            for i, e in enumerate(elements[:20])
        ])

        prompt = f"""You are exploring a web application. Current page: {current_url}
Depth: {exploration_depth}
States discovered so far: {self.state_graph.state_count}

Interactive elements found:
{element_summary}

{"User stories to cover:" + chr(10) + chr(10).join(self._user_stories) if self._user_stories else "No specific user stories provided."}

Rank the top 5 elements to interact with first, based on:
1. Likelihood to discover new states
2. Relevance to user stories (if any)
3. Importance (main navigation > secondary > forms)

Return just the numbers (1-indexed) separated by commas.
Example: 3, 1, 7, 5, 2
"""

        try:
            response = self.llm.complete(prompt, max_tokens=50)
            indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]

            # Reorder elements
            prioritized = []
            for idx in indices:
                if 0 <= idx < len(elements):
                    prioritized.append(elements[idx])

            # Add remaining elements
            for e in elements:
                if e not in prioritized:
                    prioritized.append(e)

            return prioritized

        except Exception as e:
            logger.warning(f"Prioritization failed: {e}")
            return elements

    def should_explore_state(
        self,
        state_id: str,
        depth: int
    ) -> bool:
        """Decide if a state should be further explored.

        Args:
            state_id: State to evaluate.
            depth: Current exploration depth.

        Returns:
            True if state should be explored.
        """
        state = self.state_graph.get_state(state_id)
        if not state:
            return False

        # Already explored too many states
        if self.state_graph.state_count >= self.config.max_states:
            return False

        # Too deep
        if depth >= self.config.max_depth:
            return False

        # Has unexplored elements
        transitions = self.state_graph.get_transitions_from(state_id)
        if len(transitions) < len(state.interactive_elements) // 2:
            return True

        return False

    def is_exploration_complete(self) -> tuple[bool, str]:
        """Check if exploration should stop.

        Returns:
            Tuple of (is_complete, reason).
        """
        stats = self.state_graph.get_coverage_stats()

        # Check basic bounds
        if stats["total_states"] >= self.config.max_states:
            return (True, f"Max states ({self.config.max_states}) reached")

        # Check coverage
        if stats["coverage_ratio"] > 0.8:
            return (True, "High coverage ratio achieved")

        # Check for stagnation
        unexplored = self.state_graph.get_unexplored_states()
        if not unexplored:
            return (True, "No unexplored states remain")

        # If user stories provided, check coverage
        if self._user_stories:
            covered = self._estimate_story_coverage()
            if covered >= 0.9:
                return (True, "User stories sufficiently covered")

        return (False, "Exploration in progress")

    def _estimate_story_coverage(self) -> float:
        """Estimate coverage of user stories.

        Returns:
            Coverage ratio between 0 and 1.
        """
        if not self._user_stories:
            return 1.0

        # Get all state URLs and key text
        state_urls = []
        state_text = []
        for state in self.state_graph._states.values():
            state_urls.append(state.url)
            # Would extract key text from DOM here

        prompt = f"""Given these discovered pages/states:
{chr(10).join(state_urls[:20])}

And these user stories to cover:
{chr(10).join(self._user_stories)}

Estimate what percentage (0-100) of the user stories are likely covered by the discovered states.
Return just a number.
"""

        try:
            response = self.llm.complete(prompt, max_tokens=10)
            coverage = int(response.strip().replace("%", ""))
            return coverage / 100
        except Exception:
            return 0.5

    def suggest_next_actions(
        self,
        current_state_id: str
    ) -> list[dict]:
        """Suggest next actions to take from current state.

        Args:
            current_state_id: Current state ID.

        Returns:
            List of suggested actions with reasoning.
        """
        state = self.state_graph.get_state(current_state_id)
        if not state:
            return []

        existing_transitions = self.state_graph.get_transitions_from(current_state_id)
        explored_selectors = {t.action.target.get("selector") for t in existing_transitions}

        unexplored = [
            e for e in state.interactive_elements
            if e.selector not in explored_selectors
        ]

        if not unexplored:
            return []

        element_summary = "\n".join([
            f"- [{e.tag}] {e.text or e.selector[:50]}"
            for e in unexplored[:10]
        ])

        prompt = f"""Current page: {state.url}
Unexplored elements:
{element_summary}

Which elements should be explored next, and why?
Format: selector | reason

Return top 3 suggestions.
"""

        try:
            response = self.llm.complete(prompt, max_tokens=200)
            suggestions = []

            for line in response.strip().split("\n"):
                if "|" in line:
                    selector, reason = line.split("|", 1)
                    suggestions.append({
                        "selector": selector.strip(),
                        "reason": reason.strip()
                    })

            return suggestions[:3]

        except Exception as e:
            logger.warning(f"Suggestion failed: {e}")
            return [{"selector": e.selector, "reason": "Default"} for e in unexplored[:3]]
