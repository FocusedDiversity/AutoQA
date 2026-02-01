"""Scenario definitions for mock web application behavior."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageDefinition:
    """Definition of a mock page."""
    path: str
    title: str
    template: str  # HTML template or template name
    elements: list[dict] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    on_load: Optional[str] = None  # JavaScript to run on load


@dataclass
class ActionHandler:
    """Handler for an action on a page."""
    selector: str
    action: str  # click, input, submit
    response: dict = field(default_factory=dict)  # What happens after action
    delay_ms: int = 0  # Simulated delay
    navigate_to: Optional[str] = None  # Page to navigate to
    update_element: Optional[dict] = None  # Element to update
    websocket_event: Optional[dict] = None  # WebSocket event to emit


@dataclass
class Scenario:
    """A complete test scenario with pages and behaviors."""
    name: str
    description: str = ""
    pages: list[PageDefinition] = field(default_factory=list)
    handlers: list[ActionHandler] = field(default_factory=list)
    initial_state: dict = field(default_factory=dict)
    users: list[str] = field(default_factory=list)
    websocket_enabled: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "pages": [
                {
                    "path": p.path,
                    "title": p.title,
                    "template": p.template,
                    "elements": p.elements,
                    "scripts": p.scripts,
                    "on_load": p.on_load
                }
                for p in self.pages
            ],
            "handlers": [
                {
                    "selector": h.selector,
                    "action": h.action,
                    "response": h.response,
                    "delay_ms": h.delay_ms,
                    "navigate_to": h.navigate_to,
                    "update_element": h.update_element,
                    "websocket_event": h.websocket_event
                }
                for h in self.handlers
            ],
            "initial_state": self.initial_state,
            "users": self.users,
            "websocket_enabled": self.websocket_enabled
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        """Create from dictionary."""
        pages = [
            PageDefinition(**p) for p in data.get("pages", [])
        ]
        handlers = [
            ActionHandler(**h) for h in data.get("handlers", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            pages=pages,
            handlers=handlers,
            initial_state=data.get("initial_state", {}),
            users=data.get("users", []),
            websocket_enabled=data.get("websocket_enabled", False)
        )


class ScenarioManager:
    """Manages test scenarios."""

    # Built-in scenario templates
    TEMPLATES = {
        "login": {
            "name": "Login Flow",
            "description": "Basic login/logout flow",
            "pages": [
                {
                    "path": "/",
                    "title": "Home",
                    "template": "home",
                    "elements": [
                        {"tag": "a", "id": "login-link", "text": "Login", "href": "/login"}
                    ]
                },
                {
                    "path": "/login",
                    "title": "Login",
                    "template": "login",
                    "elements": [
                        {"tag": "input", "id": "email", "type": "email", "name": "email"},
                        {"tag": "input", "id": "password", "type": "password", "name": "password"},
                        {"tag": "button", "id": "submit-btn", "type": "submit", "text": "Login"}
                    ]
                },
                {
                    "path": "/dashboard",
                    "title": "Dashboard",
                    "template": "dashboard",
                    "elements": [
                        {"tag": "h1", "id": "welcome", "text": "Welcome, User!"},
                        {"tag": "button", "id": "logout-btn", "text": "Logout"}
                    ]
                }
            ],
            "handlers": [
                {
                    "selector": "#submit-btn",
                    "action": "click",
                    "delay_ms": 500,
                    "navigate_to": "/dashboard"
                },
                {
                    "selector": "#logout-btn",
                    "action": "click",
                    "navigate_to": "/"
                }
            ]
        },
        "todo": {
            "name": "Todo App",
            "description": "Simple todo list application",
            "pages": [
                {
                    "path": "/",
                    "title": "Todo App",
                    "template": "todo",
                    "elements": [
                        {"tag": "input", "id": "new-todo", "type": "text", "placeholder": "Add new todo"},
                        {"tag": "button", "id": "add-btn", "text": "Add"},
                        {"tag": "ul", "id": "todo-list", "class": "todo-list"}
                    ]
                }
            ],
            "handlers": [
                {
                    "selector": "#add-btn",
                    "action": "click",
                    "delay_ms": 100,
                    "update_element": {
                        "selector": "#todo-list",
                        "action": "append",
                        "template": "<li class='todo-item'><input type='checkbox'><span>{{value}}</span><button class='delete'>X</button></li>"
                    }
                }
            ]
        },
        "collaboration": {
            "name": "Collaboration App",
            "description": "Multi-user real-time collaboration",
            "websocket_enabled": True,
            "users": ["user_a", "user_b"],
            "pages": [
                {
                    "path": "/",
                    "title": "Collaboration",
                    "template": "collab",
                    "elements": [
                        {"tag": "div", "id": "user-list", "class": "users"},
                        {"tag": "textarea", "id": "editor", "class": "editor"},
                        {"tag": "div", "id": "chat", "class": "chat-container"},
                        {"tag": "input", "id": "chat-input", "type": "text"},
                        {"tag": "button", "id": "send-btn", "text": "Send"}
                    ],
                    "scripts": ["websocket-client.js"]
                }
            ],
            "handlers": [
                {
                    "selector": "#send-btn",
                    "action": "click",
                    "websocket_event": {
                        "type": "chat_message",
                        "broadcast": True
                    }
                },
                {
                    "selector": "#editor",
                    "action": "input",
                    "websocket_event": {
                        "type": "editor_change",
                        "broadcast": True,
                        "debounce_ms": 300
                    }
                }
            ]
        },
        "form": {
            "name": "Complex Form",
            "description": "Multi-step form with validation",
            "pages": [
                {
                    "path": "/",
                    "title": "Registration - Step 1",
                    "template": "form-step1",
                    "elements": [
                        {"tag": "input", "id": "first-name", "type": "text", "required": True},
                        {"tag": "input", "id": "last-name", "type": "text", "required": True},
                        {"tag": "input", "id": "email", "type": "email", "required": True},
                        {"tag": "button", "id": "next-btn", "text": "Next"}
                    ]
                },
                {
                    "path": "/step2",
                    "title": "Registration - Step 2",
                    "template": "form-step2",
                    "elements": [
                        {"tag": "input", "id": "password", "type": "password", "required": True},
                        {"tag": "input", "id": "confirm-password", "type": "password", "required": True},
                        {"tag": "select", "id": "country", "options": ["USA", "Canada", "UK", "Other"]},
                        {"tag": "button", "id": "back-btn", "text": "Back"},
                        {"tag": "button", "id": "submit-btn", "text": "Submit"}
                    ]
                },
                {
                    "path": "/success",
                    "title": "Registration Complete",
                    "template": "form-success",
                    "elements": [
                        {"tag": "h1", "id": "success-message", "text": "Registration Successful!"},
                        {"tag": "a", "id": "home-link", "href": "/", "text": "Go Home"}
                    ]
                }
            ],
            "handlers": [
                {
                    "selector": "#next-btn",
                    "action": "click",
                    "navigate_to": "/step2"
                },
                {
                    "selector": "#back-btn",
                    "action": "click",
                    "navigate_to": "/"
                },
                {
                    "selector": "#submit-btn",
                    "action": "click",
                    "delay_ms": 1000,
                    "navigate_to": "/success"
                }
            ]
        }
    }

    def __init__(self, scenarios_dir: Optional[str] = None):
        """Initialize the manager.

        Args:
            scenarios_dir: Directory for custom scenario files.
        """
        self.scenarios_dir = Path(scenarios_dir) if scenarios_dir else None
        self._scenarios: dict[str, Scenario] = {}
        self._load_built_in_scenarios()

    def _load_built_in_scenarios(self) -> None:
        """Load built-in scenario templates."""
        for name, template in self.TEMPLATES.items():
            self._scenarios[name] = Scenario.from_dict(template)

    def load_from_file(self, path: str) -> Scenario:
        """Load a scenario from a JSON file.

        Args:
            path: Path to scenario file.

        Returns:
            Loaded scenario.
        """
        with open(path) as f:
            data = json.load(f)
        scenario = Scenario.from_dict(data)
        self._scenarios[scenario.name] = scenario
        return scenario

    def load_from_directory(self) -> list[Scenario]:
        """Load all scenarios from the scenarios directory.

        Returns:
            List of loaded scenarios.
        """
        if not self.scenarios_dir or not self.scenarios_dir.exists():
            return []

        scenarios = []
        for path in self.scenarios_dir.glob("*.json"):
            try:
                scenario = self.load_from_file(str(path))
                scenarios.append(scenario)
                logger.info(f"Loaded scenario: {scenario.name}")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        return scenarios

    def get(self, name: str) -> Optional[Scenario]:
        """Get a scenario by name.

        Args:
            name: Scenario name.

        Returns:
            Scenario or None.
        """
        return self._scenarios.get(name)

    def list_scenarios(self) -> list[str]:
        """List available scenario names.

        Returns:
            List of scenario names.
        """
        return list(self._scenarios.keys())

    def save_scenario(self, scenario: Scenario, path: Optional[str] = None) -> str:
        """Save a scenario to file.

        Args:
            scenario: Scenario to save.
            path: Optional output path.

        Returns:
            Path where scenario was saved.
        """
        if path is None:
            if self.scenarios_dir:
                path = str(self.scenarios_dir / f"{scenario.name}.json")
            else:
                path = f"{scenario.name}.json"

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            json.dump(scenario.to_dict(), f, indent=2)

        return str(output)

    def create_scenario(
        self,
        name: str,
        template: Optional[str] = None,
        **kwargs
    ) -> Scenario:
        """Create a new scenario.

        Args:
            name: Scenario name.
            template: Optional template to base on.
            **kwargs: Scenario parameters.

        Returns:
            Created scenario.
        """
        if template and template in self.TEMPLATES:
            data = self.TEMPLATES[template].copy()
            data["name"] = name
            data.update(kwargs)
            scenario = Scenario.from_dict(data)
        else:
            scenario = Scenario(name=name, **kwargs)

        self._scenarios[name] = scenario
        return scenario
