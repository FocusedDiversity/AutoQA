# Semi-Supervised UI Testing Solution Plan

## Executive Summary

A scalable, LLM-powered framework for automated UI testing of complex web applications, with special emphasis on multi-user collaboration scenarios. The system combines automated exploration, state graph modeling, and AI-driven test generation to achieve exhaustive test coverage.

**Key Capabilities:**
- Automated discovery of application states and transitions
- Multi-user simulation with real-time collaboration testing
- LLM-driven test generation, execution, and failure analysis
- Pluggable architecture supporting multiple LLM providers
- Hybrid state fingerprinting for efficient deduplication

---

## Problem Statement

Testing complex collaboration applications (Slack-like, project management tools, etc.) presents unique challenges:

1. **Combinatorial explosion** of user interactions and state permutations
2. **Multi-user scenarios** where actions by one user affect others in real-time
3. **Real-time synchronization** via WebSockets requiring timing-sensitive validation
4. **Manual test creation** cannot keep pace with feature development

**Goal:** Build a semi-supervised testing pipeline where initial planning guides the system, while automation and LLMs handle exhaustive permutation testing.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           AutoQA Framework                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐                                                     │
│  │     INPUTS      │                                                     │
│  │                 │                                                     │
│  │ • User Stories  │                                                     │
│  │ • Acceptance    │                                                     │
│  │   Criteria      │                                                     │
│  │ • App URL       │                                                     │
│  │ • Auth Config   │                                                     │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      EXPLORATION ENGINE                          │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │    │
│  │  │  Playwright  │  │  LLM-Guided  │  │  Multi-User        │    │    │
│  │  │  Crawler     │  │  Navigator   │  │  Context Manager   │    │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────┘    │    │
│  │                                                                  │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                │                                         │
│                                ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       STATE MANAGER                              │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │    │
│  │  │    State     │  │   Hybrid     │  │   Transition       │    │    │
│  │  │    Graph     │  │   Fingerprint│  │   Recorder         │    │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────┘    │    │
│  │                                                                  │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                │                                         │
│                                ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        LLM ENGINE                                │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │    │
│  │  │   Provider   │  │    Test      │  │    Failure         │    │    │
│  │  │   Adapter    │  │   Generator  │  │    Analyzer        │    │    │
│  │  │ (Pluggable)  │  │              │  │                    │    │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────┘    │    │
│  │                                                                  │    │
│  └─────────────────────────────┬───────────────────────────────────┘    │
│                                │                                         │
│           ┌────────────────────┼────────────────────┐                   │
│           ▼                    ▼                    ▼                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  TEST EXECUTOR  │  │  REAL-TIME      │  │   ASSERTION     │         │
│  │                 │  │  VALIDATOR      │  │   ENGINE        │         │
│  │ • Parallel run  │  │ • WebSocket     │  │ • DOM diff      │         │
│  │ • Multi-context │  │   intercept     │  │ • Visual regr.  │         │
│  │ • Retry logic   │  │ • Event timing  │  │ • LLM semantic  │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        REPORTING                                 │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │    │
│  │  │   Allure     │  │   Coverage   │  │   Feedback Loop    │    │    │
│  │  │   Reports    │  │   Metrics    │  │   (to LLM Engine)  │    │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────┘    │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. Exploration Engine

**Purpose:** Automatically discover all reachable states and actions within the application.

#### 1.1 Playwright Crawler

```python
# Core responsibilities:
# - Navigate to starting URL
# - Detect interactive elements (buttons, links, forms, etc.)
# - Execute actions and record outcomes
# - Capture DOM snapshots and screenshots

class Crawler:
    def discover_elements(page) -> List[Element]
    def execute_action(element, action_type) -> ActionResult
    def capture_state(page) -> StateSnapshot
```

#### 1.2 LLM-Guided Navigator

```python
# Uses LLM to make intelligent exploration decisions:
# - Prioritize unexplored areas
# - Understand semantic meaning of UI elements
# - Determine which actions are meaningful vs redundant

class LLMNavigator:
    def analyze_page(state: StateSnapshot) -> List[SuggestedAction]
    def prioritize_actions(actions, coverage_map) -> List[SuggestedAction]
    def is_exploration_complete(graph, user_stories) -> bool
```

#### 1.3 Multi-User Context Manager

```python
# Manages parallel browser sessions for different users:
# - Create isolated browser contexts per user
# - Coordinate actions between users
# - Track per-user and shared state

class MultiUserManager:
    def create_user_context(user_config) -> UserContext
    def execute_coordinated_sequence(sequence: List[UserAction]) -> SequenceResult
    def get_shared_state() -> SharedState
```

---

### 2. State Manager

**Purpose:** Build and maintain a graph representation of application states and transitions.

#### 2.1 Hybrid State Fingerprinting

```python
# Two-tier approach for state identification:
#
# Tier 1 (Fast): DOM-based hash
#   - URL path
#   - Key element selectors (nav, headings, forms)
#   - Visible text content hash
#
# Tier 2 (Accurate): LLM semantic verification
#   - When Tier 1 is uncertain (hash similarity 70-95%)
#   - LLM compares two states and determines if semantically equivalent

class StateFingerprinter:
    def compute_fast_hash(page) -> str
    def compute_semantic_signature(page, llm) -> str
    def are_states_equivalent(state_a, state_b) -> Tuple[bool, float]
```

#### 2.2 State Graph

```python
# Graph structure:
# - Nodes: Unique application states
# - Edges: Actions that transition between states
# - Metadata: User context, timestamps, screenshots

@dataclass
class StateNode:
    id: str
    fingerprint: str
    screenshot_path: str
    dom_snapshot: str
    url: str
    user_context: str  # Which user sees this state
    discovered_at: datetime

@dataclass
class Transition:
    from_state: str
    to_state: str
    action: Action
    user: str
    duration_ms: int
```

---

### 3. LLM Engine

**Purpose:** Provide AI capabilities for test generation, exploration guidance, and failure analysis.

#### 3.1 Provider Adapter (Pluggable)

```python
# Abstract interface for LLM providers:

class LLMProvider(ABC):
    @abstractmethod
    def complete(prompt: str, max_tokens: int) -> str

    @abstractmethod
    def analyze_image(image_path: str, prompt: str) -> str

# Implementations:
class ClaudeProvider(LLMProvider): ...
class OpenAIProvider(LLMProvider): ...
class OllamaProvider(LLMProvider): ...  # For local models

# Configuration:
llm_config = {
    "provider": "claude",  # or "openai", "ollama"
    "model": "claude-sonnet-4-20250514",
    "api_key_env": "ANTHROPIC_API_KEY",
    "max_tokens": 4096,
    "temperature": 0.2
}
```

#### 3.2 Test Generator

```python
# Generates test scripts from graph + user stories:

class TestGenerator:
    def generate_tests(
        graph: StateGraph,
        user_stories: List[UserStory],
        coverage_target: str  # "happy_path", "edge_cases", "exhaustive"
    ) -> List[GeneratedTest]

    def generate_assertions(
        state: StateNode,
        expected_behavior: str
    ) -> List[Assertion]

# Output format:
@dataclass
class GeneratedTest:
    name: str
    description: str
    steps: List[TestStep]
    assertions: List[Assertion]
    users_involved: List[str]
    estimated_complexity: str
```

#### 3.3 Failure Analyzer

```python
# Analyzes test failures and suggests fixes:

class FailureAnalyzer:
    def analyze_failure(
        test: GeneratedTest,
        error: TestError,
        screenshots: List[str]
    ) -> FailureAnalysis

    def suggest_fix(analysis: FailureAnalysis) -> SuggestedFix
    def is_flaky(test: GeneratedTest, history: List[TestRun]) -> bool

@dataclass
class FailureAnalysis:
    root_cause: str  # "bug", "flaky", "test_error", "env_issue"
    explanation: str
    affected_component: str
    confidence: float
```

---

### 4. Test Executor

**Purpose:** Run generated tests with parallel execution and proper isolation.

```python
class TestExecutor:
    def __init__(self, parallel_workers: int = 4):
        self.parallel_workers = parallel_workers

    def run_test(test: GeneratedTest) -> TestResult
    def run_suite(tests: List[GeneratedTest]) -> SuiteResult

    # Multi-user test execution:
    def run_multi_user_test(
        test: GeneratedTest,
        user_contexts: Dict[str, UserContext]
    ) -> MultiUserTestResult
```

---

### 5. Real-Time Validator

**Purpose:** Verify WebSocket events and live UI updates between users.

```python
class RealTimeValidator:
    def intercept_websocket(page) -> WebSocketInterceptor

    def wait_for_event(
        interceptor: WebSocketInterceptor,
        event_pattern: str,
        timeout_ms: int
    ) -> WebSocketEvent

    def measure_propagation_time(
        sender_context: UserContext,
        receiver_context: UserContext,
        action: Action
    ) -> PropagationMetrics

    def verify_sync(
        contexts: List[UserContext],
        expected_state: SharedState
    ) -> SyncVerificationResult

@dataclass
class PropagationMetrics:
    action_timestamp: datetime
    event_received_timestamp: datetime
    ui_updated_timestamp: datetime
    total_latency_ms: int
    within_threshold: bool
```

**Real-Time Test Scenarios:**

| Scenario | User A Action | Expected User B Behavior | Validation |
|----------|---------------|--------------------------|------------|
| Message send | Sends message | Message appears (no refresh) | DOM mutation + WS event |
| Typing indicator | Starts typing | "User A is typing..." shown | WS event timing < 500ms |
| Presence update | Goes offline | Status changes to offline | WS event + UI update |
| Concurrent edit | Edits document | Sees changes merge/conflict UI | Conflict resolution logic |
| Notification | @mentions User B | Notification badge appears | WS event + visual check |

---

### 6. Assertion Engine

**Purpose:** Multi-layer verification system combining fast checks with semantic understanding.

#### 6.1 DOM Assertions

```python
class DOMAssertion:
    def element_exists(selector: str) -> bool
    def element_text_equals(selector: str, expected: str) -> bool
    def element_visible(selector: str) -> bool
    def dom_diff(before: str, after: str) -> DOMDiff
```

#### 6.2 Visual Regression

```python
class VisualAssertion:
    def capture_screenshot(page, region: Optional[Rect]) -> Screenshot
    def compare_screenshots(baseline: Screenshot, current: Screenshot) -> VisualDiff
    def get_diff_percentage(diff: VisualDiff) -> float
    def update_baseline(test_name: str, screenshot: Screenshot) -> None
```

#### 6.3 LLM Semantic Assertions

```python
class SemanticAssertion:
    def verify_behavior(
        screenshot: Screenshot,
        expected_behavior: str,
        llm: LLMProvider
    ) -> SemanticVerification

    def extract_visible_data(screenshot: Screenshot) -> Dict
    def compare_semantic_state(expected: str, actual: Screenshot) -> bool

# Example usage:
result = semantic.verify_behavior(
    screenshot=current_screenshot,
    expected_behavior="User should see a success message confirming task creation",
    llm=claude_provider
)
# Returns: SemanticVerification(passed=True, confidence=0.95, explanation="...")
```

---

### 7. Data Factory

**Purpose:** Generate and manage test data fixtures programmatically.

```python
from faker import Faker

class DataFactory:
    def __init__(self):
        self.faker = Faker()
        self.created_entities = []

    def create_user(self, **overrides) -> User:
        user = User(
            email=overrides.get('email', self.faker.email()),
            name=overrides.get('name', self.faker.name()),
            password=overrides.get('password', self.faker.password()),
        )
        self.created_entities.append(('user', user))
        return user

    def create_task(self, assignee: User, **overrides) -> Task:
        task = Task(
            title=overrides.get('title', self.faker.sentence()),
            description=overrides.get('description', self.faker.paragraph()),
            assignee=assignee,
            status='open'
        )
        self.created_entities.append(('task', task))
        return task

    def cleanup(self):
        """Remove all created test data via API"""
        for entity_type, entity in reversed(self.created_entities):
            self._delete_entity(entity_type, entity)
```

---

### 8. Reporting

**Purpose:** Aggregate results and provide actionable insights.

#### 8.1 Allure Integration

```python
class AllureReporter:
    def attach_screenshot(name: str, screenshot: bytes) -> None
    def add_step(name: str, status: str) -> None
    def set_test_status(status: str, message: str) -> None
    def generate_report(output_dir: str) -> None
```

#### 8.2 Coverage Metrics

```python
@dataclass
class CoverageReport:
    total_states_discovered: int
    total_transitions_discovered: int
    user_stories_covered: int
    user_stories_total: int
    story_coverage_percentage: float
    uncovered_acceptance_criteria: List[str]

    # Multi-user metrics
    user_combinations_tested: int
    real_time_scenarios_tested: int
```

#### 8.3 Feedback Loop

```python
class FeedbackLoop:
    def analyze_run(suite_result: SuiteResult) -> RunAnalysis
    def identify_gaps(coverage: CoverageReport, graph: StateGraph) -> List[Gap]
    def generate_additional_tests(gaps: List[Gap]) -> List[GeneratedTest]
    def update_exploration_priorities(analysis: RunAnalysis) -> None
```

---

## Data Schemas

### Action Format

```json
{
  "id": "action_uuid",
  "type": "click|type|select|scroll|navigate|custom",
  "target": {
    "selector": "button[data-testid='submit']",
    "text": "Submit",
    "role": "button"
  },
  "value": "optional input value",
  "user": "user_a",
  "timestamp": "2024-01-15T10:30:00Z",
  "context": {
    "page_url": "/tasks/new",
    "preceding_state_id": "state_123"
  }
}
```

### State Representation

```json
{
  "id": "state_uuid",
  "fingerprint": {
    "fast_hash": "abc123...",
    "semantic_signature": "Task list view showing 3 tasks, filter set to 'My Tasks'"
  },
  "url": "/tasks",
  "user_context": "user_a",
  "dom_snapshot_path": "snapshots/state_uuid.html",
  "screenshot_path": "screenshots/state_uuid.png",
  "interactive_elements": [
    {"selector": "button.create-task", "type": "button", "text": "Create Task"},
    {"selector": "input.search", "type": "input", "placeholder": "Search..."}
  ],
  "metadata": {
    "discovered_at": "2024-01-15T10:30:00Z",
    "discovery_depth": 3,
    "visit_count": 5
  }
}
```

### Generated Test Format

```json
{
  "id": "test_uuid",
  "name": "User A creates task, User B receives notification",
  "description": "Verifies real-time notification delivery for task assignment",
  "users": ["user_a", "user_b"],
  "preconditions": [
    {"type": "user_logged_in", "user": "user_a"},
    {"type": "user_logged_in", "user": "user_b"},
    {"type": "users_in_same_workspace"}
  ],
  "steps": [
    {
      "order": 1,
      "user": "user_a",
      "action": {"type": "click", "target": "button.create-task"},
      "expected_state": "task_creation_modal_open"
    },
    {
      "order": 2,
      "user": "user_a",
      "action": {"type": "type", "target": "input.task-title", "value": "Test Task"},
      "expected_state": null
    },
    {
      "order": 3,
      "user": "user_a",
      "action": {"type": "select", "target": "select.assignee", "value": "user_b"},
      "expected_state": null
    },
    {
      "order": 4,
      "user": "user_a",
      "action": {"type": "click", "target": "button.submit"},
      "expected_state": "task_created_success"
    },
    {
      "order": 5,
      "user": "user_b",
      "action": {"type": "wait", "condition": "notification_appears", "timeout_ms": 5000},
      "expected_state": "notification_visible"
    }
  ],
  "assertions": [
    {
      "type": "dom",
      "user": "user_a",
      "check": "element_visible",
      "selector": ".success-message"
    },
    {
      "type": "dom",
      "user": "user_b",
      "check": "element_visible",
      "selector": ".notification-badge"
    },
    {
      "type": "semantic",
      "user": "user_b",
      "expected": "Notification shows task 'Test Task' was assigned"
    },
    {
      "type": "timing",
      "metric": "notification_latency_ms",
      "max_value": 3000
    }
  ],
  "tags": ["multi-user", "real-time", "notifications", "P0"],
  "source": {
    "user_story": "US-123",
    "generated_by": "llm",
    "generation_timestamp": "2024-01-15T10:30:00Z"
  }
}
```

---

## Implementation Phases

### Phase 1: Foundation (MVP)

**Goal:** Basic exploration and test generation working end-to-end.

| Component | Deliverable |
|-----------|-------------|
| Exploration Engine | Single-user Playwright crawler with element detection |
| State Manager | Fast hash fingerprinting, basic graph storage |
| LLM Engine | Claude integration, simple test generation prompts |
| Test Executor | Sequential test runner with basic assertions |
| Reporting | Console output + JSON results |

**Exit Criteria:**
- Can crawl a simple web app and discover states
- Can generate basic happy-path tests from graph
- Can execute generated tests and report pass/fail

### Phase 2: Multi-User & Real-Time

**Goal:** Support collaboration testing scenarios.

| Component | Deliverable |
|-----------|-------------|
| Multi-User Context Manager | Parallel browser contexts, coordinated sequences |
| Real-Time Validator | WebSocket interception, propagation timing |
| Data Factory | Faker integration, API-based fixture creation |
| Test Generator | Multi-user test templates, real-time assertions |

**Exit Criteria:**
- Can simulate 2+ users interacting simultaneously
- Can verify real-time updates between users
- Can measure and assert on propagation latency

### Phase 3: Intelligence & Scale

**Goal:** LLM-driven autonomous testing loop.

| Component | Deliverable |
|-----------|-------------|
| LLM-Guided Navigator | Intelligent exploration prioritization |
| Hybrid Fingerprinting | LLM semantic verification for uncertain states |
| Failure Analyzer | Automatic failure triage and fix suggestions |
| Feedback Loop | Automatic gap detection and test generation |
| Visual Regression | Screenshot comparison baseline management |

**Exit Criteria:**
- System can autonomously explore and generate comprehensive tests
- Failure analysis correctly categorizes 90%+ of failures
- Coverage gaps automatically trigger additional test generation

### Phase 4: Production Readiness

**Goal:** Scalable, maintainable, production-grade system.

| Component | Deliverable |
|-----------|-------------|
| Provider Abstraction | Support for Claude, OpenAI, local models |
| Parallel Execution | Distributed test running across workers |
| Allure Integration | Rich visual reporting with trends |
| CI/CD Integration | GitHub Actions / Jenkins pipeline templates |
| Documentation | User guide, API docs, example configurations |

**Exit Criteria:**
- Can run 1000+ tests in parallel
- Full documentation and onboarding guide
- Proven on 3+ different web applications

---

## Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Browser Automation** | Playwright (Python) | UI interaction, multi-context, WS interception |
| **Test Framework** | pytest | Test organization, fixtures, assertions |
| **LLM Integration** | LangChain or direct API | Provider abstraction, prompt management |
| **Data Generation** | Faker | Realistic test data |
| **Graph Storage** | NetworkX + JSON | State graph representation |
| **Visualization** | Graphviz, Mermaid | Flow diagrams |
| **Reporting** | Allure, pytest-html | Test result visualization |
| **Visual Regression** | Pixelmatch, percy | Screenshot comparison |
| **Configuration** | Pydantic, YAML | Type-safe config management |

---

## Configuration Example

```yaml
# autoqa.config.yaml

app:
  base_url: "https://staging.myapp.com"
  login_url: "/auth/login"

users:
  user_a:
    email: "tester.a@example.com"
    password_env: "TEST_USER_A_PASSWORD"
  user_b:
    email: "tester.b@example.com"
    password_env: "TEST_USER_B_PASSWORD"

exploration:
  max_depth: 10
  max_states: 500
  timeout_minutes: 30
  ignored_selectors:
    - ".ad-banner"
    - "#cookie-popup"

llm:
  provider: "claude"
  model: "claude-sonnet-4-20250514"
  api_key_env: "ANTHROPIC_API_KEY"
  temperature: 0.2

test_generation:
  coverage_target: "exhaustive"  # "happy_path", "edge_cases", "exhaustive"
  include_negative_tests: true
  max_tests_per_flow: 20

real_time:
  enabled: true
  max_propagation_ms: 3000
  websocket_patterns:
    - "wss://*/socket"

assertions:
  dom_enabled: true
  visual_enabled: true
  visual_threshold: 0.01  # 1% diff allowed
  semantic_enabled: true

execution:
  parallel_workers: 4
  retry_count: 2
  screenshot_on_failure: true

reporting:
  allure_enabled: true
  output_dir: "./reports"
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| State Discovery | 95% of reachable states | Manual audit vs automated discovery |
| Test Coverage | 100% of user stories | Acceptance criteria mapped to tests |
| Failure Detection | 99% of real bugs caught | Comparison with manual QA findings |
| False Positive Rate | < 5% | Flaky tests / total tests |
| Generation Accuracy | 90% tests runnable without edit | Tests that pass syntax + basic execution |
| Real-Time Latency | < 3s propagation verified | Measured in multi-user tests |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM generates invalid tests | Wasted execution time | Syntax validation + dry-run before full suite |
| State explosion in complex apps | Memory/time limits exceeded | Configurable depth limits, smart pruning |
| Flaky tests due to timing | False failures | Retry logic, explicit waits, timing thresholds |
| WebSocket interception complexity | Missed real-time bugs | Fallback to DOM polling validation |
| LLM API costs | Budget overrun | Caching, batching, local model fallback |
| Application changes break tests | Maintenance burden | Self-healing selectors, semantic targeting |

---

## Open Questions

1. **Test Prioritization:** How should we rank which generated tests to run first?
2. **Baseline Management:** How do we handle visual regression baselines when UI intentionally changes?
3. **Conflict Resolution Testing:** How deep should we go into concurrent edit scenarios?
4. **Error Injection:** Should we test network failures, slow connections, etc.?
5. **Accessibility Testing:** Should this be integrated into the assertion engine?

---

## Next Steps

1. [ ] Set up project structure with core modules
2. [ ] Implement basic Playwright crawler
3. [ ] Build state fingerprinting with fast hash
4. [ ] Create LLM provider abstraction layer
5. [ ] Develop simple test generator with Claude
6. [ ] Build test executor with pytest integration
7. [ ] Add multi-user context management
8. [ ] Implement WebSocket interception for real-time tests
9. [ ] Create Allure reporting integration
10. [ ] Document API and create usage examples
