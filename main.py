#!/usr/bin/env python3
"""AutoQA - LLM-Powered Semi-Supervised UI Testing Framework.

Commands:
    explore     Crawl and explore a web application to build state graph
    generate    Generate test scripts from state graph
    execute     Execute generated tests
    analyze     Analyze test coverage
    report      Generate test reports
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("autoqa")


def cmd_explore(args):
    """Explore a web application and build state graph."""
    from src.core.config import AutoQAConfig
    from src.exploration.crawler import Crawler
    from src.state.graph import StateGraph

    logger.info(f"Starting exploration of {args.url}")

    # Load config
    config = AutoQAConfig.from_yaml(args.config) if args.config else AutoQAConfig()

    # Override with CLI args
    if args.headless is not None:
        config.exploration.headless = args.headless
    if args.max_states:
        config.exploration.max_states = args.max_states
    if args.max_depth:
        config.exploration.max_depth = args.max_depth

    # Set base URL
    config.app.base_url = args.url

    # Create state graph
    state_graph = StateGraph()

    # Create crawler
    crawler = Crawler(
        config=config.exploration,
        state_graph=state_graph,
        screenshots_dir=args.screenshots_dir
    )

    # Set up progress callback
    def on_state_discovered(state_id: str, is_new: bool):
        status = "NEW" if is_new else "EXISTING"
        logger.info(f"  [{status}] State: {state_id[:8]}...")

    crawler.set_callback(on_state_discovered)

    # Run exploration
    graph = crawler.explore(
        start_url=args.url,
        storage_state=args.storage_state
    )

    # Save state graph
    output_path = args.output or "state_graph.json"
    graph.save(output_path)

    stats = graph.get_coverage_stats()
    logger.info(f"\nExploration complete:")
    logger.info(f"  States discovered: {stats['total_states']}")
    logger.info(f"  Transitions found: {stats['total_transitions']}")
    logger.info(f"  Graph saved to: {output_path}")

    return 0


def cmd_generate(args):
    """Generate tests from state graph."""
    from src.core.config import AutoQAConfig
    from src.core.models import UserStory
    from src.llm.engine import LLMEngine
    from src.state.graph import StateGraph
    from src.testing.generator import TestGenerator

    logger.info("Generating tests from state graph...")

    # Load config
    config = AutoQAConfig.from_yaml(args.config) if args.config else AutoQAConfig()

    # Load state graph
    state_graph = StateGraph.load(args.graph)
    logger.info(f"Loaded graph with {state_graph.state_count} states")

    # Create LLM engine
    llm_engine = LLMEngine(config=config.llm)

    # Load user stories if provided
    user_stories = []
    if args.stories:
        with open(args.stories) as f:
            stories_data = json.load(f)
            for story_data in stories_data:
                user_stories.append(UserStory(**story_data))
        logger.info(f"Loaded {len(user_stories)} user stories")

    # Create generator
    generator = TestGenerator(
        llm_engine=llm_engine,
        state_graph=state_graph,
        coverage_target=args.coverage_target
    )

    # Generate tests
    tests = generator.generate_tests(
        user_stories=user_stories if user_stories else None,
        max_tests_per_story=args.max_tests
    )

    logger.info(f"Generated {len(tests)} tests")

    # Save tests
    output_path = args.output or "generated_tests.json"
    tests_data = [
        {
            "name": t.name,
            "description": t.description,
            "users": t.users,
            "steps": [
                {
                    "order": s.order,
                    "user": s.user,
                    "action": {
                        "type": s.action.action_type.value,
                        "target": s.action.target,
                        "value": s.action.value
                    },
                    "expected_state": s.expected_state
                }
                for s in t.steps
            ],
            "assertions": [
                {
                    "type": a.assertion_type.value,
                    "user": a.user,
                    "check": a.check,
                    "selector": a.selector,
                    "expected": a.expected
                }
                for a in t.assertions
            ],
            "tags": t.tags,
            "source_story": t.source_story
        }
        for t in tests
    ]

    with open(output_path, "w") as f:
        json.dump(tests_data, f, indent=2)

    logger.info(f"Tests saved to: {output_path}")
    return 0


def cmd_execute(args):
    """Execute generated tests."""
    from src.core.config import AutoQAConfig
    from src.core.models import Action, ActionType, Assertion, AssertionType, GeneratedTest, TestStep
    from src.testing.executor import TestExecutor
    from src.reporting.allure_reporter import AllureReporter

    logger.info("Executing tests...")

    # Load config
    config = AutoQAConfig.from_yaml(args.config) if args.config else AutoQAConfig()

    # Override with CLI args
    if args.headless is not None:
        config.exploration.headless = args.headless

    # Load tests
    with open(args.tests) as f:
        tests_data = json.load(f)

    # Convert to GeneratedTest objects
    tests = []
    for td in tests_data:
        steps = []
        for sd in td.get("steps", []):
            action_data = sd.get("action", {})
            action = Action(
                action_type=ActionType(action_data.get("type", "click")),
                target=action_data.get("target", {}),
                value=action_data.get("value"),
                user=sd.get("user", "user_a")
            )
            step = TestStep(
                order=sd.get("order", len(steps) + 1),
                user=sd.get("user", "user_a"),
                action=action,
                expected_state=sd.get("expected_state")
            )
            steps.append(step)

        assertions = []
        for ad in td.get("assertions", []):
            assertion = Assertion(
                assertion_type=AssertionType(ad.get("type", "dom")),
                user=ad.get("user", "user_a"),
                check=ad.get("check"),
                selector=ad.get("selector"),
                expected=ad.get("expected")
            )
            assertions.append(assertion)

        test = GeneratedTest(
            name=td.get("name", "Unnamed"),
            description=td.get("description"),
            users=td.get("users", ["user_a"]),
            steps=steps,
            assertions=assertions,
            tags=td.get("tags", []),
            source_story=td.get("source_story")
        )
        tests.append(test)

    logger.info(f"Loaded {len(tests)} tests")

    # Create executor
    executor = TestExecutor(
        config=config,
        screenshots_dir=args.screenshots_dir
    )

    # Run tests
    results = executor.run_suite(tests)

    # Generate report
    if args.allure:
        reporter = AllureReporter(results_dir=args.allure_dir)
        summary = reporter.report_suite(results)
        logger.info(f"\nAllure results written to: {args.allure_dir}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    logger.info(f"\nExecution complete:")
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Pass rate: {passed/len(results)*100:.1f}%")

    return 0 if failed == 0 else 1


def cmd_analyze(args):
    """Analyze test coverage."""
    from src.core.models import Action, ActionType, Assertion, AssertionType, GeneratedTest, TestStep, UserStory
    from src.state.graph import StateGraph
    from src.reporting.coverage import CoverageAnalyzer

    logger.info("Analyzing test coverage...")

    # Load state graph
    state_graph = StateGraph.load(args.graph)

    # Load tests
    with open(args.tests) as f:
        tests_data = json.load(f)

    # Convert to GeneratedTest objects (simplified)
    tests = []
    for td in tests_data:
        steps = []
        for sd in td.get("steps", []):
            action_data = sd.get("action", {})
            action = Action(
                action_type=ActionType(action_data.get("type", "click")),
                target=action_data.get("target", {}),
                value=action_data.get("value")
            )
            step = TestStep(
                order=sd.get("order", len(steps) + 1),
                user=sd.get("user", "user_a"),
                action=action,
                expected_state=sd.get("expected_state")
            )
            steps.append(step)

        assertions = []
        for ad in td.get("assertions", []):
            assertion = Assertion(
                assertion_type=AssertionType(ad.get("type", "dom")),
                check=ad.get("check"),
                selector=ad.get("selector"),
                expected=ad.get("expected")
            )
            assertions.append(assertion)

        test = GeneratedTest(
            name=td.get("name", "Unnamed"),
            description=td.get("description"),
            users=td.get("users", ["user_a"]),
            steps=steps,
            assertions=assertions,
            tags=td.get("tags", []),
            source_story=td.get("source_story")
        )
        tests.append(test)

    # Load user stories if provided
    user_stories = []
    if args.stories:
        with open(args.stories) as f:
            stories_data = json.load(f)
            for story_data in stories_data:
                user_stories.append(UserStory(**story_data))

    # Analyze coverage
    analyzer = CoverageAnalyzer(
        state_graph=state_graph,
        user_stories=user_stories if user_stories else None
    )

    report = analyzer.analyze(tests)

    # Print report
    logger.info("\nCoverage Report:")
    logger.info(f"  Overall: {report.overall_percentage:.1f}%")
    logger.info(f"  States: {report.state_coverage.percentage:.1f}% ({report.state_coverage.covered}/{report.state_coverage.total})")
    logger.info(f"  Transitions: {report.transition_coverage.percentage:.1f}% ({report.transition_coverage.covered}/{report.transition_coverage.total})")
    logger.info(f"  Stories: {report.story_coverage.percentage:.1f}% ({report.story_coverage.covered}/{report.story_coverage.total})")

    # Save report if output specified
    if args.output:
        analyzer.save_report(report, args.output)
        logger.info(f"\nReport saved to: {args.output}")

    # Get suggestions
    if args.suggest:
        suggestions = analyzer.suggest_additional_tests(tests)
        if suggestions:
            logger.info("\nSuggested improvements:")
            for s in suggestions[:5]:
                logger.info(f"  [{s['priority']}] {s['description']}")

    return 0


def cmd_report(args):
    """Generate reports from test results."""
    from src.reporting.allure_reporter import AllureReporter

    logger.info("Generating reports...")

    if args.format == "allure":
        import subprocess
        result = subprocess.run([
            "allure", "generate", args.results_dir,
            "-o", args.output_dir,
            "--clean"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Allure report generated in: {args.output_dir}")
            if args.open:
                subprocess.run(["allure", "open", args.output_dir])
        else:
            logger.error(f"Allure generation failed: {result.stderr}")
            return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="AutoQA - LLM-Powered Semi-Supervised UI Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Explore a web application")
    explore_parser.add_argument("url", help="URL to start exploration from")
    explore_parser.add_argument("-o", "--output", help="Output path for state graph JSON")
    explore_parser.add_argument("--storage-state", help="Path to browser storage state for auth")
    explore_parser.add_argument("--headless", type=bool, help="Run in headless mode")
    explore_parser.add_argument("--max-states", type=int, help="Maximum states to discover")
    explore_parser.add_argument("--max-depth", type=int, help="Maximum exploration depth")
    explore_parser.add_argument("--screenshots-dir", default="screenshots", help="Screenshots directory")
    explore_parser.set_defaults(func=cmd_explore)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate tests from state graph")
    generate_parser.add_argument("graph", help="Path to state graph JSON")
    generate_parser.add_argument("-o", "--output", help="Output path for generated tests")
    generate_parser.add_argument("--stories", help="Path to user stories JSON")
    generate_parser.add_argument("--coverage-target", default="exhaustive",
                                  choices=["happy_path", "edge_cases", "exhaustive"])
    generate_parser.add_argument("--max-tests", type=int, default=10, help="Max tests per story")
    generate_parser.set_defaults(func=cmd_generate)

    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute generated tests")
    execute_parser.add_argument("tests", help="Path to tests JSON file")
    execute_parser.add_argument("--headless", type=bool, help="Run in headless mode")
    execute_parser.add_argument("--screenshots-dir", default="screenshots/failures")
    execute_parser.add_argument("--allure", action="store_true", help="Generate Allure results")
    execute_parser.add_argument("--allure-dir", default="allure-results")
    execute_parser.set_defaults(func=cmd_execute)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze test coverage")
    analyze_parser.add_argument("graph", help="Path to state graph JSON")
    analyze_parser.add_argument("tests", help="Path to tests JSON file")
    analyze_parser.add_argument("-o", "--output", help="Output path for coverage report")
    analyze_parser.add_argument("--stories", help="Path to user stories JSON")
    analyze_parser.add_argument("--suggest", action="store_true", help="Suggest improvements")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--format", default="allure", choices=["allure"])
    report_parser.add_argument("--results-dir", default="allure-results")
    report_parser.add_argument("--output-dir", default="allure-report")
    report_parser.add_argument("--open", action="store_true", help="Open report in browser")
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
