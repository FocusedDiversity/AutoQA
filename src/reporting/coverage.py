"""Test coverage analysis and reporting."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.models import GeneratedTest, UserStory
from ..state.graph import StateGraph
from ..testing.executor import TestResult

logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Coverage metrics for a category."""
    total: int = 0
    covered: int = 0
    uncovered_items: list[str] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        """Calculate coverage percentage."""
        if self.total == 0:
            return 100.0
        return (self.covered / self.total) * 100


@dataclass
class CoverageReport:
    """Complete coverage analysis report."""
    generated_at: datetime = field(default_factory=datetime.now)
    state_coverage: CoverageMetrics = field(default_factory=CoverageMetrics)
    transition_coverage: CoverageMetrics = field(default_factory=CoverageMetrics)
    story_coverage: CoverageMetrics = field(default_factory=CoverageMetrics)
    element_coverage: CoverageMetrics = field(default_factory=CoverageMetrics)
    path_coverage: CoverageMetrics = field(default_factory=CoverageMetrics)

    @property
    def overall_percentage(self) -> float:
        """Calculate overall coverage percentage."""
        total = (
            self.state_coverage.total +
            self.transition_coverage.total +
            self.story_coverage.total
        )
        covered = (
            self.state_coverage.covered +
            self.transition_coverage.covered +
            self.story_coverage.covered
        )
        if total == 0:
            return 100.0
        return (covered / total) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall_percentage": self.overall_percentage,
            "state_coverage": {
                "total": self.state_coverage.total,
                "covered": self.state_coverage.covered,
                "percentage": self.state_coverage.percentage,
                "uncovered": self.state_coverage.uncovered_items[:10]
            },
            "transition_coverage": {
                "total": self.transition_coverage.total,
                "covered": self.transition_coverage.covered,
                "percentage": self.transition_coverage.percentage,
                "uncovered": self.transition_coverage.uncovered_items[:10]
            },
            "story_coverage": {
                "total": self.story_coverage.total,
                "covered": self.story_coverage.covered,
                "percentage": self.story_coverage.percentage,
                "uncovered": self.story_coverage.uncovered_items[:10]
            },
            "element_coverage": {
                "total": self.element_coverage.total,
                "covered": self.element_coverage.covered,
                "percentage": self.element_coverage.percentage
            },
            "path_coverage": {
                "total": self.path_coverage.total,
                "covered": self.path_coverage.covered,
                "percentage": self.path_coverage.percentage
            }
        }


class CoverageAnalyzer:
    """Analyzes test coverage against state graph and user stories.

    Measures:
    - State coverage: Which states are tested
    - Transition coverage: Which transitions are exercised
    - Story coverage: Which user stories have tests
    - Element coverage: Which interactive elements are tested
    - Path coverage: Which paths through the app are covered
    """

    def __init__(
        self,
        state_graph: StateGraph,
        user_stories: Optional[list[UserStory]] = None
    ):
        """Initialize the analyzer.

        Args:
            state_graph: Application state graph.
            user_stories: User stories to check coverage against.
        """
        self.state_graph = state_graph
        self.user_stories = user_stories or []

    def analyze(
        self,
        tests: list[GeneratedTest],
        results: Optional[list[TestResult]] = None
    ) -> CoverageReport:
        """Analyze test coverage.

        Args:
            tests: Generated tests.
            results: Optional test execution results.

        Returns:
            CoverageReport with metrics.
        """
        report = CoverageReport()

        # State coverage
        report.state_coverage = self._analyze_state_coverage(tests)

        # Transition coverage
        report.transition_coverage = self._analyze_transition_coverage(tests)

        # Story coverage
        report.story_coverage = self._analyze_story_coverage(tests)

        # Element coverage
        report.element_coverage = self._analyze_element_coverage(tests)

        # Path coverage
        report.path_coverage = self._analyze_path_coverage(tests)

        return report

    def _analyze_state_coverage(self, tests: list[GeneratedTest]) -> CoverageMetrics:
        """Analyze state coverage.

        Args:
            tests: Generated tests.

        Returns:
            CoverageMetrics for states.
        """
        all_states = set(self.state_graph._states.keys())
        covered_states = set()

        # Extract states referenced in tests
        for test in tests:
            for step in test.steps:
                if step.expected_state:
                    # Try to find matching state
                    for state_id, state in self.state_graph._states.items():
                        if step.expected_state in state.url or step.expected_state == state_id[:8]:
                            covered_states.add(state_id)

        uncovered = all_states - covered_states
        return CoverageMetrics(
            total=len(all_states),
            covered=len(covered_states),
            uncovered_items=[
                f"{sid[:8]}: {self.state_graph._states[sid].url}"
                for sid in list(uncovered)[:10]
            ]
        )

    def _analyze_transition_coverage(self, tests: list[GeneratedTest]) -> CoverageMetrics:
        """Analyze transition coverage.

        Args:
            tests: Generated tests.

        Returns:
            CoverageMetrics for transitions.
        """
        all_transitions = list(self.state_graph.graph.edges())
        covered_transitions = set()

        # Extract actions from tests and match to transitions
        test_selectors = set()
        for test in tests:
            for step in test.steps:
                selector = step.action.target.get("selector", "")
                if selector:
                    test_selectors.add(selector)

        # Match selectors to transitions
        for u, v, data in self.state_graph.graph.edges(data=True):
            action = data.get("action", {})
            if isinstance(action, dict):
                target = action.get("target", {})
                selector = target.get("selector", "") if isinstance(target, dict) else ""
            else:
                selector = action.target.get("selector", "") if hasattr(action, "target") else ""

            if selector in test_selectors:
                covered_transitions.add((u, v))

        uncovered = set(all_transitions) - covered_transitions
        return CoverageMetrics(
            total=len(all_transitions),
            covered=len(covered_transitions),
            uncovered_items=[
                f"{u[:8]} -> {v[:8]}"
                for u, v in list(uncovered)[:10]
            ]
        )

    def _analyze_story_coverage(self, tests: list[GeneratedTest]) -> CoverageMetrics:
        """Analyze user story coverage.

        Args:
            tests: Generated tests.

        Returns:
            CoverageMetrics for stories.
        """
        if not self.user_stories:
            return CoverageMetrics(total=0, covered=0)

        covered_stories = set()
        for test in tests:
            if test.source_story:
                covered_stories.add(test.source_story)

        uncovered = [
            s.id for s in self.user_stories
            if s.id not in covered_stories
        ]

        return CoverageMetrics(
            total=len(self.user_stories),
            covered=len(covered_stories),
            uncovered_items=[
                f"{s.id}: {s.title}"
                for s in self.user_stories if s.id in uncovered
            ][:10]
        )

    def _analyze_element_coverage(self, tests: list[GeneratedTest]) -> CoverageMetrics:
        """Analyze interactive element coverage.

        Args:
            tests: Generated tests.

        Returns:
            CoverageMetrics for elements.
        """
        # Collect all interactive elements from states
        all_elements = set()
        for state in self.state_graph._states.values():
            for element in state.interactive_elements:
                all_elements.add(element.selector)

        # Collect tested elements
        tested_elements = set()
        for test in tests:
            for step in test.steps:
                selector = step.action.target.get("selector", "")
                if selector:
                    tested_elements.add(selector)

        return CoverageMetrics(
            total=len(all_elements),
            covered=len(tested_elements & all_elements)
        )

    def _analyze_path_coverage(self, tests: list[GeneratedTest]) -> CoverageMetrics:
        """Analyze path coverage through the application.

        Args:
            tests: Generated tests.

        Returns:
            CoverageMetrics for paths.
        """
        # Get unique paths from state graph
        unique_paths = self.state_graph.find_all_paths(max_depth=5)
        covered_paths = 0

        # Approximate path coverage by test step sequences
        for test in tests:
            if len(test.steps) >= 2:
                covered_paths += 1

        return CoverageMetrics(
            total=len(unique_paths) if unique_paths else 0,
            covered=min(covered_paths, len(unique_paths) if unique_paths else 0)
        )

    def get_uncovered_states(self, tests: list[GeneratedTest]) -> list[str]:
        """Get list of uncovered state IDs.

        Args:
            tests: Generated tests.

        Returns:
            List of uncovered state IDs.
        """
        metrics = self._analyze_state_coverage(tests)
        return metrics.uncovered_items

    def suggest_additional_tests(self, tests: list[GeneratedTest]) -> list[dict]:
        """Suggest additional tests to improve coverage.

        Args:
            tests: Current tests.

        Returns:
            List of test suggestions.
        """
        suggestions = []

        # Check uncovered states
        state_metrics = self._analyze_state_coverage(tests)
        for uncovered in state_metrics.uncovered_items[:5]:
            suggestions.append({
                "type": "state_coverage",
                "description": f"Add test covering state: {uncovered}",
                "priority": "high"
            })

        # Check uncovered transitions
        transition_metrics = self._analyze_transition_coverage(tests)
        for uncovered in transition_metrics.uncovered_items[:5]:
            suggestions.append({
                "type": "transition_coverage",
                "description": f"Add test for transition: {uncovered}",
                "priority": "medium"
            })

        # Check uncovered stories
        story_metrics = self._analyze_story_coverage(tests)
        for uncovered in story_metrics.uncovered_items[:5]:
            suggestions.append({
                "type": "story_coverage",
                "description": f"Add test for user story: {uncovered}",
                "priority": "high"
            })

        return suggestions

    def save_report(self, report: CoverageReport, output_path: str) -> None:
        """Save coverage report to file.

        Args:
            report: Coverage report.
            output_path: Output file path.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Coverage report saved to: {output_path}")
