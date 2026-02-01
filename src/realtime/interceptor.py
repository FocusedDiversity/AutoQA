"""WebSocket message interception for real-time event validation."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from playwright.sync_api import Page, WebSocket

logger = logging.getLogger(__name__)


@dataclass
class WebSocketMessage:
    """Captured WebSocket message."""
    direction: str  # "sent" or "received"
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    url: str = ""

    @property
    def is_json(self) -> bool:
        """Check if data is JSON."""
        return isinstance(self.data, (dict, list))

    def get_event_type(self) -> Optional[str]:
        """Extract event type from message if present."""
        if isinstance(self.data, dict):
            return self.data.get("type") or self.data.get("event") or self.data.get("action")
        return None


@dataclass
class WebSocketCapture:
    """Collection of captured WebSocket messages."""
    url: str
    messages: list[WebSocketMessage] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    def filter_by_type(self, event_type: str) -> list[WebSocketMessage]:
        """Filter messages by event type."""
        return [m for m in self.messages if m.get_event_type() == event_type]

    def filter_by_direction(self, direction: str) -> list[WebSocketMessage]:
        """Filter messages by direction."""
        return [m for m in self.messages if m.direction == direction]

    @property
    def sent_messages(self) -> list[WebSocketMessage]:
        """Get all sent messages."""
        return self.filter_by_direction("sent")

    @property
    def received_messages(self) -> list[WebSocketMessage]:
        """Get all received messages."""
        return self.filter_by_direction("received")


class WebSocketInterceptor:
    """Intercepts and captures WebSocket messages for validation.

    Supports:
    - Multi-socket capture across pages
    - Message filtering by type/direction
    - Event correlation between users
    - Latency measurement
    """

    def __init__(self):
        """Initialize the interceptor."""
        self._captures: dict[str, WebSocketCapture] = {}
        self._active_sockets: dict[str, WebSocket] = {}
        self._message_handlers: list[Callable[[WebSocketMessage], None]] = []

    def attach_to_page(self, page: Page, user_id: str = "default") -> None:
        """Attach interceptor to a Playwright page.

        Args:
            page: Playwright page to monitor.
            user_id: User identifier for this page.
        """
        def on_websocket(ws: WebSocket) -> None:
            url = ws.url
            capture_key = f"{user_id}:{url}"

            self._active_sockets[capture_key] = ws
            self._captures[capture_key] = WebSocketCapture(url=url)

            logger.debug(f"WebSocket opened: {url} for user {user_id}")

            def on_frame_sent(payload: str) -> None:
                self._handle_message(capture_key, "sent", payload)

            def on_frame_received(payload: str) -> None:
                self._handle_message(capture_key, "received", payload)

            def on_close(_: WebSocket) -> None:
                logger.debug(f"WebSocket closed: {url}")
                self._active_sockets.pop(capture_key, None)

            ws.on("framesent", on_frame_sent)
            ws.on("framereceived", on_frame_received)
            ws.on("close", on_close)

        page.on("websocket", on_websocket)

    def _handle_message(self, capture_key: str, direction: str, payload: str) -> None:
        """Handle a WebSocket message.

        Args:
            capture_key: Capture identifier.
            direction: Message direction.
            payload: Raw message payload.
        """
        # Parse JSON if possible
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            data = payload

        message = WebSocketMessage(
            direction=direction,
            data=data,
            url=self._captures[capture_key].url if capture_key in self._captures else ""
        )

        if capture_key in self._captures:
            self._captures[capture_key].messages.append(message)

        # Call registered handlers
        for handler in self._message_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.warning(f"Message handler error: {e}")

    def add_message_handler(self, handler: Callable[[WebSocketMessage], None]) -> None:
        """Add a message handler callback.

        Args:
            handler: Callback function.
        """
        self._message_handlers.append(handler)

    def get_capture(self, user_id: str, url_pattern: Optional[str] = None) -> Optional[WebSocketCapture]:
        """Get captured messages for a user.

        Args:
            user_id: User identifier.
            url_pattern: Optional URL pattern to match.

        Returns:
            WebSocketCapture or None.
        """
        for key, capture in self._captures.items():
            if key.startswith(f"{user_id}:"):
                if url_pattern is None or url_pattern in capture.url:
                    return capture
        return None

    def get_all_captures(self) -> dict[str, WebSocketCapture]:
        """Get all captures.

        Returns:
            Dictionary of all captures.
        """
        return self._captures.copy()

    def wait_for_message(
        self,
        user_id: str,
        event_type: str,
        timeout_ms: int = 5000,
        direction: str = "received"
    ) -> Optional[WebSocketMessage]:
        """Wait for a specific message type.

        Args:
            user_id: User to wait for.
            event_type: Event type to match.
            timeout_ms: Timeout in milliseconds.
            direction: Message direction.

        Returns:
            Matching message or None.
        """
        import time

        start = datetime.now()
        timeout_sec = timeout_ms / 1000

        while (datetime.now() - start).total_seconds() < timeout_sec:
            capture = self.get_capture(user_id)
            if capture:
                messages = capture.filter_by_direction(direction)
                for msg in reversed(messages):  # Check newest first
                    if msg.get_event_type() == event_type:
                        return msg
            time.sleep(0.1)

        return None

    def clear(self) -> None:
        """Clear all captures."""
        self._captures.clear()

    def get_message_latencies(
        self,
        sender_id: str,
        receiver_id: str,
        event_type: str
    ) -> list[float]:
        """Calculate message propagation latencies.

        Args:
            sender_id: Sender user ID.
            receiver_id: Receiver user ID.
            event_type: Event type to measure.

        Returns:
            List of latencies in milliseconds.
        """
        sender_capture = self.get_capture(sender_id)
        receiver_capture = self.get_capture(receiver_id)

        if not sender_capture or not receiver_capture:
            return []

        sent_messages = [
            m for m in sender_capture.sent_messages
            if m.get_event_type() == event_type
        ]
        received_messages = [
            m for m in receiver_capture.received_messages
            if m.get_event_type() == event_type
        ]

        latencies = []
        for sent in sent_messages:
            # Find corresponding received message
            for received in received_messages:
                if received.timestamp > sent.timestamp:
                    latency = (received.timestamp - sent.timestamp).total_seconds() * 1000
                    latencies.append(latency)
                    break

        return latencies
