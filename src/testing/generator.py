"""LLM-powered test generation from state graph."""

import json
import logging
from typing import Optional

from ..core.models import (
    Action,
    ActionType,
    Assertion,
    AssertionType,
    GeneratedTest,
    TestStep,
    UserStory,
)
from ..llm.engine import LLMEngine
from ..state.graph import StateGraph

logger = logging.getLogger(__name__)


class TestGenerator:
    """Generates test scripts from state graph and user stories.

    Uses LLM to:
    - Analyze state graph paths
    - Generate test cases covering user stories
    - Create assertions for each test
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        state_graph: StateGraph,
        coverage_target: str = "exhaustive"
    ):
        """Initialize the test generator.

        Args:
            llm_engine: LLM engine for generation.
            state_graph: State graph to generate tests from.
            coverage_target: Coverage level (happy_path, edge_cases, exhaustive).
        """
        self.llm = llm_engine
        self.state_graph = state_graph
        self.coverage_target = coverage_target

    def generate_tests(
        self,
        user_stories: Optional[list[UserStory]] = None,
        max_tests_per_story: int = 10
    ) -> list[GeneratedTest]:
        """Generate tests for user stories.

        Args:
            user_stories: Stories to generate tests for.
            max_tests_per_story: Maximum tests per story.

        Returns:
            List of generated tests.
        """
        all_tests = []

        if user_stories:
            for story in user_stories:
                tests = self._generate_tests_for_story(story, max_tests_per_story)
                all_tests.extend(tests)
        else:
            # Generate tests from graph paths without stories
            tests = self._generate_tests_from_graph(max_tests_per_story * 5)
            all_tests.extend(tests)

        return all_tests

    def _generate_tests_for_story(
        self,
        story: UserStory,
        max_tests: int
    ) -> list[GeneratedTest]:
        """Generate tests for a single user story.

        Args:
            story: User story to cover.
            max_tests: Maximum tests to generate.

        Returns:
            List of generated tests.
        """
        # Build graph summary
        graph_summary = self._build_graph_summary()

        prompt = f"""Generate test cases for this user story based on the application state graph.

USER STORY:
Title: {story.title}
Description: {story.description}

ACCEPTANCE CRITERIA:
{chr(10).join(f"- {ac}" for ac in story.acceptance_criteria)}

APPLICATION STATE GRAPH:
{graph_summary}

COVERAGE TARGET: {self.coverage_target}

Generate {max_tests} test cases in JSON format. Each test should have:
- name: descriptive test name
- description: what the test verifies
- users: list of users involved (e.g., ["user_a"] or ["user_a", "user_b"] for multi-user)
- steps: array of steps with order, user, action (type, target, value), expected_state
- assertions: array with type (dom/semantic/timing), check, selector, expected

Example test:
{{
  "name": "User creates task successfully",
  "description": "Verify user can create a new task",
  "users": ["user_a"],
  "steps": [
    {{"order": 1, "user": "user_a", "action": {{"type": "click", "target": "button.create"}}, "expected_state": "form_open"}}
  ],
  "assertions": [
    {{"type": "dom", "check": "element_visible", "selector": ".success-message"}}
  ],
  "tags": ["happy_path", "core"]
}}

Return a JSON array of tests.
"""

        try:
            response = self.llm.complete(prompt, max_tokens=4000)
            tests_data = self._parse_json_response(response)

            tests = []
            for test_data in tests_data[:max_tests]:
                test = self._create_test_from_data(test_data, story.id)
                if test:
                    tests.append(test)

            return tests

        except Exception as e:
            logger.error(f"Test generation failed for story {story.id}: {e}")
            return []

    def _generate_tests_from_graph(self, max_tests: int) -> list[GeneratedTest]:
        """Generate tests directly from graph paths.

        Args:
            max_tests: Maximum tests to generate.

        Returns:
            List of generated tests.
        """
        graph_summary = self._build_graph_summary()

        prompt = f"""Analyze this application state graph and generate comprehensive test cases.

STATE GRAPH:
{graph_summary}

COVERAGE TARGET: {self.coverage_target}

Generate {max_tests} test cases that cover:
1. Happy paths through the application
2. Edge cases (empty inputs, invalid data)
3. Error scenarios (if visible in the graph)
4. Multi-user scenarios (if applicable)

Return tests in JSON format with: name, description, users, steps, assertions, tags.
"""

        try:
            response = self.llm.complete(prompt, max_tokens=4000)
            tests_data = self._parse_json_response(response)

            tests = []
            for test_data in tests_data[:max_tests]:
                test = self._create_test_from_data(test_data)
                if test:
                    tests.append(test)

            return tests

        except Exception as e:
            logger.error(f"Test generation from graph failed: {e}")
            return []

    def _build_graph_summary(self) -> str:
        """Build a summary of the state graph for LLM.

        Returns:
            Text summary of the graph.
        """
        lines = []

        # States summary
        lines.append("STATES:")
        for state_id, state in list(self.state_graph._states.items())[:20]:
            elements_count = len(state.interactive_elements)
            lines.append(f"  - {state.url} (ID: {state_id[:8]}, {elements_count} elements)")

        # Transitions summary
        lines.append("\nTRANSITIONS:")
        for u, v, data in list(self.state_graph.graph.edges(data=True))[:30]:
            action = data.get("action", {})
            action_type = action.get("action_type", "unknown")
            target = action.get("target", {}).get("text", "")[:30]
            lines.append(f"  - {u[:8]} --[{action_type}: {target}]--> {v[:8]}")

        # Stats
        stats = self.state_graph.get_coverage_stats()
        lines.append(f"\nSTATS: {stats['total_states']} states, {stats['total_transitions']} transitions")

        return "\n".join(lines)

    def _parse_json_response(self, response: str) -> list[dict]:
        """Parse JSON from LLM response.

        Args:
            response: Raw LLM response.

        Returns:
            List of test data dictionaries.
        """
        # Try to find JSON array in response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.index("```json") + 7
            end = response.index("```", start)
            response = response[start:end]
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            response = response[start:end]

        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "tests" in data:
                return data["tests"]
            else:
                return [data]
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response")
            return []

    def _create_test_from_data(
        self,
        data: dict,
        source_story: Optional[str] = None
    ) -> Optional[GeneratedTest]:
        """Create a GeneratedTest from parsed data.

        Args:
            data: Parsed test data.
            source_story: Source story ID.

        Returns:
            GeneratedTest or None if invalid.
        """
        try:
            steps = []
            for step_data in data.get("steps", []):
                action_data = step_data.get("action", {})
                action = Action(
                    action_type=ActionType(action_data.get("type", "click")),
                    target={"selector": action_data.get("target", "")},
                    value=action_data.get("value"),
                    user=step_data.get("user", "user_a")
                )
                step = TestStep(
                    order=step_data.get("order", len(steps) + 1),
                    user=step_data.get("user", "user_a"),
                    action=action,
                    expected_state=step_data.get("expected_state")
                )
                steps.append(step)

            assertions = []
            for assert_data in data.get("assertions", []):
                assertion = Assertion(
                    assertion_type=AssertionType(assert_data.get("type", "dom")),
                    user=assert_data.get("user", "user_a"),
                    check=assert_data.get("check"),
                    selector=assert_data.get("selector"),
                    expected=assert_data.get("expected"),
                    threshold=assert_data.get("threshold"),
                    max_value=assert_data.get("max_value")
                )
                assertions.append(assertion)

            return GeneratedTest(
                name=data.get("name", "Unnamed test"),
                description=data.get("description"),
                users=data.get("users", ["user_a"]),
                steps=steps,
                assertions=assertions,
                tags=data.get("tags", []),
                source_story=source_story
            )

        except Exception as e:
            logger.warning(f"Failed to create test: {e}")
            return None

    def generate_assertions_for_state(
        self,
        state_id: str,
        expected_behavior: str
    ) -> list[Assertion]:
        """Generate assertions for a specific state.

        Args:
            state_id: State to generate assertions for.
            expected_behavior: Description of expected behavior.

        Returns:
            List of assertions.
        """
        state = self.state_graph.get_state(state_id)
        if not state:
            return []

        prompt = f"""Generate test assertions for this UI state.

URL: {state.url}
Expected behavior: {expected_behavior}

Interactive elements visible:
{chr(10).join(f"- {e.tag}: {e.text or e.selector[:40]}" for e in state.interactive_elements[:10])}

Generate assertions that verify the expected behavior.
Return JSON array with type (dom/semantic/visual/timing), check, selector, expected.
"""

        try:
            response = self.llm.complete(prompt, max_tokens=500)
            assertions_data = self._parse_json_response(response)

            assertions = []
            for data in assertions_data:
                assertion = Assertion(
                    assertion_type=AssertionType(data.get("type", "dom")),
                    check=data.get("check"),
                    selector=data.get("selector"),
                    expected=data.get("expected")
                )
                assertions.append(assertion)

            return assertions

        except Exception as e:
            logger.warning(f"Assertion generation failed: {e}")
            return []
