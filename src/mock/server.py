"""Mock web application server using Flask."""

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from flask import Flask, render_template_string, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room

from .scenarios import Scenario, ScenarioManager

logger = logging.getLogger(__name__)


# HTML Templates for common page types
TEMPLATES = {
    "base": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { margin-bottom: 20px; color: #333; }
        input, select, textarea { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px 0; }
        button:hover { background: #0056b3; }
        .btn-secondary { background: #6c757d; }
        .btn-danger { background: #dc3545; }
        a { color: #007bff; text-decoration: none; }
        ul { list-style: none; }
        li { padding: 10px; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 10px; }
        .error { color: #dc3545; }
        .success { color: #28a745; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        .users { display: flex; gap: 10px; margin-bottom: 20px; }
        .user-badge { padding: 5px 10px; background: #e9ecef; border-radius: 20px; font-size: 14px; }
        .editor { min-height: 200px; font-family: monospace; }
        .chat-container { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; }
        .chat-message { padding: 5px; margin: 5px 0; }
        .chat-message.own { text-align: right; background: #e3f2fd; }
        .chat-message.other { background: #f5f5f5; }
        [data-testid] { position: relative; }
    </style>
    {% block extra_styles %}{% endblock %}
</head>
<body>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
    <script>
        // Mock app state
        window.mockState = {{ state | tojson }};

        // Event handlers
        document.addEventListener('DOMContentLoaded', function() {
            {% block on_load %}{% endblock %}
        });
    </script>
    {% block extra_scripts %}{% endblock %}
</body>
</html>
""",
    "home": """
{% extends "base" %}
{% block content %}
<h1>Welcome to Mock App</h1>
<p>This is a mock web application for testing.</p>
<nav>
    {% for link in links %}
    <a href="{{ link.href }}" id="{{ link.id }}" data-testid="{{ link.id }}">{{ link.text }}</a>
    {% endfor %}
</nav>
{% endblock %}
""",
    "login": """
{% extends "base" %}
{% block content %}
<h1>Login</h1>
<form id="login-form" data-testid="login-form">
    <div class="form-group">
        <label for="email">Email</label>
        <input type="email" id="email" name="email" data-testid="email-input" required>
    </div>
    <div class="form-group">
        <label for="password">Password</label>
        <input type="password" id="password" name="password" data-testid="password-input" required>
    </div>
    <div id="error-message" class="error" style="display: none;"></div>
    <button type="submit" id="submit-btn" data-testid="login-button">Login</button>
</form>
{% endblock %}
{% block on_load %}
document.getElementById('login-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    fetch('/api/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({email, password})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            window.location.href = data.redirect || '/dashboard';
        } else {
            document.getElementById('error-message').textContent = data.error;
            document.getElementById('error-message').style.display = 'block';
        }
    });
});
{% endblock %}
""",
    "dashboard": """
{% extends "base" %}
{% block content %}
<h1 id="welcome" data-testid="welcome-message">Welcome, {{ user.name or 'User' }}!</h1>
<div id="dashboard-content">
    <p>You are now logged in.</p>
    <button id="logout-btn" data-testid="logout-button" onclick="logout()">Logout</button>
</div>
{% endblock %}
{% block on_load %}
function logout() {
    fetch('/api/logout', {method: 'POST'})
    .then(() => window.location.href = '/');
}
{% endblock %}
""",
    "todo": """
{% extends "base" %}
{% block content %}
<h1>Todo List</h1>
<div class="form-group" style="display: flex; gap: 10px;">
    <input type="text" id="new-todo" data-testid="new-todo-input" placeholder="Add new todo...">
    <button id="add-btn" data-testid="add-todo-button" onclick="addTodo()">Add</button>
</div>
<ul id="todo-list" data-testid="todo-list">
    {% for todo in todos %}
    <li class="todo-item" data-id="{{ todo.id }}" data-testid="todo-item-{{ todo.id }}">
        <input type="checkbox" {{ 'checked' if todo.completed else '' }} onchange="toggleTodo('{{ todo.id }}')">
        <span>{{ todo.text }}</span>
        <button class="btn-danger" onclick="deleteTodo('{{ todo.id }}')" data-testid="delete-{{ todo.id }}">X</button>
    </li>
    {% endfor %}
</ul>
{% endblock %}
{% block on_load %}
function addTodo() {
    const input = document.getElementById('new-todo');
    const text = input.value.trim();
    if (!text) return;

    fetch('/api/todos', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            location.reload();
        }
    });
    input.value = '';
}

function toggleTodo(id) {
    fetch('/api/todos/' + id + '/toggle', {method: 'POST'})
    .then(() => location.reload());
}

function deleteTodo(id) {
    fetch('/api/todos/' + id, {method: 'DELETE'})
    .then(() => location.reload());
}
{% endblock %}
""",
    "collab": """
{% extends "base" %}
{% block content %}
<h1>Collaboration Space</h1>
<div id="user-list" class="users" data-testid="user-list">
    <span>Connected users:</span>
    <div id="online-users"></div>
</div>
<div class="form-group">
    <label for="editor">Shared Editor</label>
    <textarea id="editor" class="editor" data-testid="shared-editor"></textarea>
</div>
<div id="chat" class="chat-container" data-testid="chat-container"></div>
<div style="display: flex; gap: 10px;">
    <input type="text" id="chat-input" data-testid="chat-input" placeholder="Type a message...">
    <button id="send-btn" data-testid="send-button">Send</button>
</div>
{% endblock %}
{% block extra_scripts %}
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.min.js"></script>
<script>
const socket = io();
const userId = '{{ user_id }}';

socket.on('connect', () => {
    socket.emit('join', {user_id: userId});
});

socket.on('user_joined', (data) => {
    updateUsers(data.users);
});

socket.on('user_left', (data) => {
    updateUsers(data.users);
});

socket.on('chat_message', (data) => {
    addChatMessage(data.user, data.message, data.user === userId);
});

socket.on('editor_change', (data) => {
    if (data.user !== userId) {
        document.getElementById('editor').value = data.content;
    }
});

function updateUsers(users) {
    document.getElementById('online-users').innerHTML =
        users.map(u => '<span class="user-badge">' + u + '</span>').join('');
}

function addChatMessage(user, message, isOwn) {
    const chat = document.getElementById('chat');
    const div = document.createElement('div');
    div.className = 'chat-message ' + (isOwn ? 'own' : 'other');
    div.innerHTML = '<strong>' + user + ':</strong> ' + message;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

document.getElementById('send-btn').addEventListener('click', () => {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message) {
        socket.emit('chat_message', {message});
        input.value = '';
    }
});

let editorTimeout;
document.getElementById('editor').addEventListener('input', (e) => {
    clearTimeout(editorTimeout);
    editorTimeout = setTimeout(() => {
        socket.emit('editor_change', {content: e.target.value});
    }, 300);
});
</script>
{% endblock %}
""",
    "form-step1": """
{% extends "base" %}
{% block content %}
<h1>Registration - Step 1 of 2</h1>
<form id="step1-form">
    <div class="form-group">
        <label for="first-name">First Name</label>
        <input type="text" id="first-name" name="first_name" data-testid="first-name-input" required>
    </div>
    <div class="form-group">
        <label for="last-name">Last Name</label>
        <input type="text" id="last-name" name="last_name" data-testid="last-name-input" required>
    </div>
    <div class="form-group">
        <label for="email">Email</label>
        <input type="email" id="email" name="email" data-testid="email-input" required>
    </div>
    <button type="button" id="next-btn" data-testid="next-button" onclick="nextStep()">Next</button>
</form>
{% endblock %}
{% block on_load %}
function nextStep() {
    const form = document.getElementById('step1-form');
    if (form.checkValidity()) {
        const data = new FormData(form);
        fetch('/api/form/step1', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(Object.fromEntries(data))
        })
        .then(() => window.location.href = '/step2');
    } else {
        form.reportValidity();
    }
}
{% endblock %}
""",
    "form-step2": """
{% extends "base" %}
{% block content %}
<h1>Registration - Step 2 of 2</h1>
<form id="step2-form">
    <div class="form-group">
        <label for="password">Password</label>
        <input type="password" id="password" name="password" data-testid="password-input" required minlength="8">
    </div>
    <div class="form-group">
        <label for="confirm-password">Confirm Password</label>
        <input type="password" id="confirm-password" name="confirm_password" data-testid="confirm-password-input" required>
    </div>
    <div class="form-group">
        <label for="country">Country</label>
        <select id="country" name="country" data-testid="country-select">
            <option value="">Select country...</option>
            <option value="USA">USA</option>
            <option value="Canada">Canada</option>
            <option value="UK">UK</option>
            <option value="Other">Other</option>
        </select>
    </div>
    <div id="error-message" class="error" style="display: none;"></div>
    <button type="button" id="back-btn" class="btn-secondary" data-testid="back-button" onclick="window.location.href='/'">Back</button>
    <button type="button" id="submit-btn" data-testid="submit-button" onclick="submitForm()">Submit</button>
</form>
{% endblock %}
{% block on_load %}
function submitForm() {
    const form = document.getElementById('step2-form');
    const password = document.getElementById('password').value;
    const confirm = document.getElementById('confirm-password').value;
    const errorEl = document.getElementById('error-message');

    if (password !== confirm) {
        errorEl.textContent = 'Passwords do not match';
        errorEl.style.display = 'block';
        return;
    }

    if (form.checkValidity()) {
        const data = new FormData(form);
        fetch('/api/form/step2', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(Object.fromEntries(data))
        })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                window.location.href = '/success';
            } else {
                errorEl.textContent = data.error;
                errorEl.style.display = 'block';
            }
        });
    } else {
        form.reportValidity();
    }
}
{% endblock %}
""",
    "form-success": """
{% extends "base" %}
{% block content %}
<div style="text-align: center; padding: 40px;">
    <h1 id="success-message" class="success" data-testid="success-message">Registration Successful!</h1>
    <p>Thank you for registering.</p>
    <a href="/" id="home-link" data-testid="home-link">Go Home</a>
</div>
{% endblock %}
"""
}


@dataclass
class MockServerConfig:
    """Configuration for the mock server."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    threaded: bool = True
    static_folder: Optional[str] = None


class MockServer:
    """Mock web application server for testing.

    Features:
    - Configurable pages and routes
    - Built-in scenario templates
    - WebSocket support for real-time testing
    - API endpoints for state manipulation
    - Screenshot-friendly styling
    """

    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        config: Optional[MockServerConfig] = None
    ):
        """Initialize the mock server.

        Args:
            scenario: Scenario to serve.
            config: Server configuration.
        """
        self.scenario = scenario
        self.config = config or MockServerConfig()
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'mock-secret-key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Server state
        self._state: dict[str, Any] = {}
        self._users: dict[str, dict] = {}
        self._todos: list[dict] = []
        self._form_data: dict = {}
        self._connected_users: set = set()

        self._setup_routes()
        self._setup_socketio()

        self._server_thread: Optional[threading.Thread] = None
        self._running = False

    def _setup_routes(self) -> None:
        """Set up Flask routes."""

        @self.app.route("/")
        def index():
            template = self._get_template("home")
            return render_template_string(
                TEMPLATES["base"].replace("{% block content %}{% endblock %}", template),
                title="Mock App",
                state=self._state,
                links=[{"id": "login-link", "href": "/login", "text": "Login"}],
                todos=self._todos
            )

        @self.app.route("/login")
        def login_page():
            return render_template_string(
                TEMPLATES["base"].replace(
                    "{% block content %}{% endblock %}",
                    self._get_template("login")
                ),
                title="Login",
                state=self._state
            )

        @self.app.route("/dashboard")
        def dashboard_page():
            return render_template_string(
                TEMPLATES["base"].replace(
                    "{% block content %}{% endblock %}",
                    self._get_template("dashboard")
                ),
                title="Dashboard",
                state=self._state,
                user={"name": self._state.get("current_user", "User")}
            )

        @self.app.route("/todo")
        def todo_page():
            return render_template_string(
                TEMPLATES["base"].replace(
                    "{% block content %}{% endblock %}",
                    self._get_template("todo")
                ),
                title="Todo App",
                state=self._state,
                todos=self._todos
            )

        @self.app.route("/collab")
        @self.app.route("/collab/<user_id>")
        def collab_page(user_id: str = "user_a"):
            return render_template_string(
                TEMPLATES["collab"],
                title="Collaboration",
                state=self._state,
                user_id=user_id
            )

        @self.app.route("/step2")
        def form_step2():
            return render_template_string(
                TEMPLATES["base"].replace(
                    "{% block content %}{% endblock %}",
                    self._get_template("form-step2")
                ),
                title="Registration - Step 2",
                state=self._state
            )

        @self.app.route("/success")
        def form_success():
            return render_template_string(
                TEMPLATES["base"].replace(
                    "{% block content %}{% endblock %}",
                    self._get_template("form-success")
                ),
                title="Success",
                state=self._state
            )

        # API Routes
        @self.app.route("/api/login", methods=["POST"])
        def api_login():
            data = request.json
            email = data.get("email", "")
            password = data.get("password", "")

            # Simple mock validation
            if email and password:
                self._state["logged_in"] = True
                self._state["current_user"] = email.split("@")[0]
                return jsonify({"success": True, "redirect": "/dashboard"})
            return jsonify({"success": False, "error": "Invalid credentials"})

        @self.app.route("/api/logout", methods=["POST"])
        def api_logout():
            self._state["logged_in"] = False
            self._state.pop("current_user", None)
            return jsonify({"success": True})

        @self.app.route("/api/todos", methods=["GET", "POST"])
        def api_todos():
            if request.method == "POST":
                data = request.json
                todo = {
                    "id": str(len(self._todos) + 1),
                    "text": data.get("text", ""),
                    "completed": False
                }
                self._todos.append(todo)
                return jsonify({"success": True, "todo": todo})
            return jsonify({"todos": self._todos})

        @self.app.route("/api/todos/<todo_id>", methods=["DELETE"])
        def api_delete_todo(todo_id: str):
            self._todos = [t for t in self._todos if t["id"] != todo_id]
            return jsonify({"success": True})

        @self.app.route("/api/todos/<todo_id>/toggle", methods=["POST"])
        def api_toggle_todo(todo_id: str):
            for todo in self._todos:
                if todo["id"] == todo_id:
                    todo["completed"] = not todo["completed"]
                    break
            return jsonify({"success": True})

        @self.app.route("/api/form/step1", methods=["POST"])
        def api_form_step1():
            self._form_data.update(request.json)
            return jsonify({"success": True})

        @self.app.route("/api/form/step2", methods=["POST"])
        def api_form_step2():
            self._form_data.update(request.json)
            return jsonify({"success": True})

        @self.app.route("/api/state", methods=["GET", "POST"])
        def api_state():
            if request.method == "POST":
                self._state.update(request.json)
                return jsonify({"success": True})
            return jsonify(self._state)

        @self.app.route("/api/reset", methods=["POST"])
        def api_reset():
            self._state.clear()
            self._todos.clear()
            self._form_data.clear()
            return jsonify({"success": True})

    def _setup_socketio(self) -> None:
        """Set up Socket.IO event handlers."""

        @self.socketio.on("connect")
        def handle_connect():
            logger.debug("Client connected")

        @self.socketio.on("disconnect")
        def handle_disconnect():
            logger.debug("Client disconnected")

        @self.socketio.on("join")
        def handle_join(data):
            user_id = data.get("user_id", "anonymous")
            self._connected_users.add(user_id)
            join_room("collab")
            emit("user_joined", {
                "user": user_id,
                "users": list(self._connected_users)
            }, room="collab")

        @self.socketio.on("leave")
        def handle_leave(data):
            user_id = data.get("user_id", "anonymous")
            self._connected_users.discard(user_id)
            leave_room("collab")
            emit("user_left", {
                "user": user_id,
                "users": list(self._connected_users)
            }, room="collab")

        @self.socketio.on("chat_message")
        def handle_chat(data):
            emit("chat_message", {
                "user": data.get("user_id", "anonymous"),
                "message": data.get("message", "")
            }, room="collab")

        @self.socketio.on("editor_change")
        def handle_editor(data):
            emit("editor_change", {
                "user": data.get("user_id", "anonymous"),
                "content": data.get("content", "")
            }, room="collab", include_self=False)

    def _get_template(self, name: str) -> str:
        """Get template content.

        Args:
            name: Template name.

        Returns:
            Template string.
        """
        if self.scenario:
            for page in self.scenario.pages:
                if page.template == name:
                    return page.template

        return TEMPLATES.get(name, TEMPLATES.get("home", ""))

    def start(self, background: bool = False) -> Optional[str]:
        """Start the mock server.

        Args:
            background: Run in background thread.

        Returns:
            Server URL if started.
        """
        url = f"http://{self.config.host}:{self.config.port}"

        if background:
            self._running = True
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self._server_thread.start()
            # Wait for server to start
            time.sleep(1)
            logger.info(f"Mock server started at {url}")
        else:
            logger.info(f"Starting mock server at {url}")
            self._run_server()

        return url

    def _run_server(self) -> None:
        """Run the server."""
        self.socketio.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            use_reloader=False,
            log_output=self.config.debug
        )

    def stop(self) -> None:
        """Stop the mock server."""
        self._running = False
        # Note: Flask-SocketIO doesn't have a clean shutdown
        # The daemon thread will be killed when main thread exits

    def get_url(self) -> str:
        """Get the server URL.

        Returns:
            Server URL string.
        """
        return f"http://{self.config.host}:{self.config.port}"

    def set_state(self, key: str, value: Any) -> None:
        """Set a state value.

        Args:
            key: State key.
            value: State value.
        """
        self._state[key] = value

    def get_state(self, key: str) -> Any:
        """Get a state value.

        Args:
            key: State key.

        Returns:
            State value.
        """
        return self._state.get(key)

    def reset(self) -> None:
        """Reset all server state."""
        self._state.clear()
        self._todos.clear()
        self._form_data.clear()
        self._connected_users.clear()

    def add_todo(self, text: str, completed: bool = False) -> dict:
        """Add a todo item.

        Args:
            text: Todo text.
            completed: Whether completed.

        Returns:
            Created todo.
        """
        todo = {
            "id": str(len(self._todos) + 1),
            "text": text,
            "completed": completed
        }
        self._todos.append(todo)
        return todo
