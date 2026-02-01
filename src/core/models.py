"""Core data models for AutoQA."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid


class ActionType(str, Enum):
    """Types of UI actions."""
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    WAIT = "wait"
    HOVER = "hover"
    CUSTOM = "custom"


class AssertionType(str, Enum):
    """Types of test assertions."""
    DOM = "dom"
    VISUAL = "visual"
    SEMANTIC = "semantic"
    TIMING = "timing"


class StoryStatus(str, Enum):
    """Status of user stories in the curation workflow."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class InteractiveElement(BaseModel):
    """An interactive element discovered on a page."""
    selector: str
    tag: str
    element_type: Optional[str] = None
    text: Optional[str] = None
    role: Optional[str] = None
    visible: bool = True
    rect: Optional[dict[str, float]] = None


class Action(BaseModel):
    """A user action on the UI."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType
    target: Optional[dict[str, Any]] = None
    value: Optional[str] = None
    user: str = "default"
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Optional[dict[str, Any]] = None


class StateFingerprint(BaseModel):
    """Fingerprint for state identification."""
    fast_hash: str
    semantic_signature: Optional[str] = None
    similarity_threshold: float = 0.95


class StateNode(BaseModel):
    """A unique application state in the state graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint: StateFingerprint
    url: str
    title: Optional[str] = None
    dom_snapshot_path: Optional[str] = None
    screenshot_path: Optional[str] = None
    interactive_elements: list[InteractiveElement] = Field(default_factory=list)
    user_context: str = "default"
    discovered_at: datetime = Field(default_factory=datetime.now)
    discovery_depth: int = 0
    visit_count: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class Transition(BaseModel):
    """A transition between states caused by an action."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_state: str
    to_state: str
    action: Action
    user: str = "default"
    duration_ms: int = 0
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)


class UserStory(BaseModel):
    """A user story with acceptance criteria."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    acceptance_criteria: list[str] = Field(default_factory=list)
    inferred_from: list[str] = Field(default_factory=list)  # State IDs
    confidence: float = 1.0
    status: StoryStatus = StoryStatus.DRAFT
    reviewer_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TestStep(BaseModel):
    """A step in a generated test."""
    order: int
    user: str = "default"
    action: Action
    expected_state: Optional[str] = None
    wait_for: Optional[str] = None
    timeout_ms: int = 5000


class Assertion(BaseModel):
    """A test assertion."""
    assertion_type: AssertionType
    user: str = "default"
    check: Optional[str] = None
    selector: Optional[str] = None
    expected: Optional[str] = None
    threshold: Optional[float] = None
    max_value: Optional[int] = None


class GeneratedTest(BaseModel):
    """A generated test case."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    users: list[str] = Field(default_factory=lambda: ["default"])
    preconditions: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[TestStep] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source_story: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.now)
    estimated_complexity: str = "medium"


class ExplorationConfig(BaseModel):
    """Configuration for exploration bounds."""
    max_depth: int = 10
    max_states: int = 500
    timeout_minutes: int = 30
    max_actions_per_state: int = 20
    ignored_selectors: list[str] = Field(default_factory=list)
    headless: bool = True
    slow_mo: int = 0
    viewport: dict[str, int] = Field(default_factory=lambda: {"width": 1280, "height": 720})
