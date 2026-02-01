"""AutoQA - LLM-Powered Semi-Supervised UI Testing Framework.

Modules:
    core        - Core data models and configuration
    exploration - Web application crawling and exploration
    state       - State graph management and fingerprinting
    llm         - LLM engine integration
    testing     - Test generation and execution
    realtime    - WebSocket and multi-user validation
    reporting   - Test reports and coverage analysis
    utils       - Utility functions and helpers
"""

__version__ = "0.1.0"
__author__ = "AutoQA Team"

from .core.config import AutoQAConfig
from .core.models import (
    GeneratedTest,
    TestStep,
    Action,
    ActionType,
    Assertion,
    AssertionType,
    StateNode,
    UserStory,
)
from .state.graph import StateGraph
from .exploration.crawler import Crawler
from .testing.generator import TestGenerator
from .testing.executor import TestExecutor

__all__ = [
    # Config
    "AutoQAConfig",
    # Models
    "GeneratedTest",
    "TestStep",
    "Action",
    "ActionType",
    "Assertion",
    "AssertionType",
    "StateNode",
    "UserStory",
    # Core classes
    "StateGraph",
    "Crawler",
    "TestGenerator",
    "TestExecutor",
]
