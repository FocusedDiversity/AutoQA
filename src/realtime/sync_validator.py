"""Multi-user synchronization validation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from playwright.sync_api import Page

from .interceptor import WebSocketInterceptor, WebSocketMessage

logger = logging.getLogger(__name__)


@dataclass
class SyncExpectation:
    """Expected synchronization behavior."""
    event_type: str
    from_user: str
    to_users: list[str]
    max_latency_ms: int = 1000
    payload_check: Optional[dict[str, Any]] = None


@dataclass
class SyncValidationResult:
    """Result of a sync validation check."""
    expectation: SyncExpectation
    passed: bool
    actual_latencies: dict[str, float] = field(default_factory=dict)
    missing_receivers: list[str] = field(default_factory=list)
    payload_mismatches: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if not self.actual_latencies:
            return 0.0
        return sum(self.actual_latencies.values()) / len(self.actual_latencies)


class SyncValidator:
    """Validates real-time synchronization between multiple users.

    Supports:
    - Event propagation verification
    - Latency threshold checking
    - Payload consistency validation
    - Multi-user state convergence testing
    """

    def __init__(self, interceptor: Optional[WebSocketInterceptor] = None):
        """Initialize the validator.

        Args:
            interceptor: WebSocket interceptor to use.
        """
        self.interceptor = interceptor or WebSocketInterceptor()
        self._pages: dict[str, Page] = {}

    def register_user(self, user_id: str, page: Page) -> None:
        """Register a user's page for validation.

        Args:
            user_id: User identifier.
            page: Playwright page.
        """
        self._pages[user_id] = page
        self.interceptor.attach_to_page(page, user_id)

    def validate_sync(
        self,
        expectation: SyncExpectation,
        timeout_ms: int = 5000
    ) -> SyncValidationResult:
        """Validate that an event syncs correctly between users.

        Args:
            expectation: Expected sync behavior.
            timeout_ms: Wait timeout.

        Returns:
            SyncValidationResult.
        """
        result = SyncValidationResult(
            expectation=expectation,
            passed=True
        )

        # Get sender's sent message
        sender_capture = self.interceptor.get_capture(expectation.from_user)
        if not sender_capture:
            result.passed = False
            result.error = f"No capture for sender: {expectation.from_user}"
            return result

        sent_messages = [
            m for m in sender_capture.sent_messages
            if m.get_event_type() == expectation.event_type
        ]

        if not sent_messages:
            result.passed = False
            result.error = f"No sent messages of type: {expectation.event_type}"
            return result

        sent_message = sent_messages[-1]  # Use most recent

        # Check each receiver
        for receiver_id in expectation.to_users:
            received_msg = self.interceptor.wait_for_message(
                receiver_id,
                expectation.event_type,
                timeout_ms=timeout_ms,
                direction="received"
            )

            if not received_msg:
                result.missing_receivers.append(receiver_id)
                result.passed = False
                continue

            # Calculate latency
            latency = (received_msg.timestamp - sent_message.timestamp).total_seconds() * 1000
            result.actual_latencies[receiver_id] = latency

            # Check latency threshold
            if latency > expectation.max_latency_ms:
                result.passed = False

            # Check payload if specified
            if expectation.payload_check:
                mismatch = self._check_payload(
                    received_msg.data,
                    expectation.payload_check
                )
                if mismatch:
                    result.payload_mismatches[receiver_id] = mismatch
                    result.passed = False

        if result.missing_receivers:
            result.error = f"Missing receivers: {result.missing_receivers}"

        return result

    def _check_payload(self, actual: Any, expected: dict[str, Any]) -> Optional[str]:
        """Check if actual payload matches expected fields.

        Args:
            actual: Actual payload.
            expected: Expected field values.

        Returns:
            Mismatch description or None.
        """
        if not isinstance(actual, dict):
            return f"Payload is not a dict: {type(actual)}"

        for key, expected_value in expected.items():
            if key not in actual:
                return f"Missing key: {key}"
            if actual[key] != expected_value:
                return f"Key {key}: expected {expected_value}, got {actual[key]}"

        return None

    def validate_state_convergence(
        self,
        selector: str,
        timeout_ms: int = 5000
    ) -> dict[str, Any]:
        """Validate that all users see the same UI state.

        Args:
            selector: CSS selector to check.
            timeout_ms: Wait timeout.

        Returns:
            Convergence result with values per user.
        """
        import time

        result = {
            "converged": False,
            "values": {},
            "final_value": None
        }

        start = datetime.now()
        timeout_sec = timeout_ms / 1000

        while (datetime.now() - start).total_seconds() < timeout_sec:
            values = {}
            for user_id, page in self._pages.items():
                try:
                    locator = page.locator(selector)
                    if locator.count() > 0:
                        values[user_id] = locator.first.text_content()
                except Exception as e:
                    logger.warning(f"Error getting value for {user_id}: {e}")
                    values[user_id] = None

            result["values"] = values

            # Check if all values are the same
            unique_values = set(v for v in values.values() if v is not None)
            if len(unique_values) == 1:
                result["converged"] = True
                result["final_value"] = unique_values.pop()
                break

            time.sleep(0.1)

        return result

    def trigger_and_validate(
        self,
        trigger_user: str,
        action: callable,
        expectation: SyncExpectation,
        timeout_ms: int = 5000
    ) -> SyncValidationResult:
        """Trigger an action and validate sync.

        Args:
            trigger_user: User who triggers the action.
            action: Callable that performs the action.
            expectation: Expected sync behavior.
            timeout_ms: Validation timeout.

        Returns:
            SyncValidationResult.
        """
        # Clear previous captures
        self.interceptor.clear()

        # Execute the action
        action()

        # Validate sync
        return self.validate_sync(expectation, timeout_ms)

    def get_latency_stats(self, event_type: str) -> dict[str, Any]:
        """Get latency statistics for an event type.

        Args:
            event_type: Event type to analyze.

        Returns:
            Statistics dictionary.
        """
        all_latencies = []

        users = list(self._pages.keys())
        for i, sender in enumerate(users):
            for receiver in users[i+1:]:
                latencies = self.interceptor.get_message_latencies(
                    sender, receiver, event_type
                )
                all_latencies.extend(latencies)

        if not all_latencies:
            return {"count": 0}

        return {
            "count": len(all_latencies),
            "min_ms": min(all_latencies),
            "max_ms": max(all_latencies),
            "avg_ms": sum(all_latencies) / len(all_latencies),
            "p95_ms": sorted(all_latencies)[int(len(all_latencies) * 0.95)] if len(all_latencies) >= 20 else max(all_latencies)
        }
