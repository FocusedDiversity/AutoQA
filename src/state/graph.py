"""State graph management for AutoQA."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import networkx as nx

from ..core.models import StateNode, Transition, Action, StateFingerprint
from .fingerprinter import StateFingerprinter


class StateGraph:
    """Manages the state graph of discovered application states.

    Uses NetworkX for graph operations with JSON persistence.
    """

    def __init__(
        self,
        fingerprinter: Optional[StateFingerprinter] = None,
        storage_path: Optional[str] = None
    ):
        """Initialize the state graph.

        Args:
            fingerprinter: StateFingerprinter for deduplication.
            storage_path: Path to persist the graph.
        """
        self.graph = nx.DiGraph()
        self.fingerprinter = fingerprinter or StateFingerprinter()
        self.storage_path = storage_path
        self._states: dict[str, StateNode] = {}
        self._fingerprint_index: dict[str, str] = {}  # fast_hash -> state_id

    @property
    def state_count(self) -> int:
        """Return number of unique states."""
        return len(self._states)

    @property
    def transition_count(self) -> int:
        """Return number of transitions."""
        return self.graph.number_of_edges()

    def add_state(
        self,
        url: str,
        dom_content: str,
        screenshot_path: Optional[str] = None,
        user_context: str = "default",
        discovery_depth: int = 0,
        metadata: Optional[dict] = None
    ) -> tuple[str, bool]:
        """Add a state to the graph, deduplicating if necessary.

        Args:
            url: Page URL.
            dom_content: DOM HTML content.
            screenshot_path: Optional screenshot path.
            user_context: User who discovered this state.
            discovery_depth: Depth from start state.
            metadata: Additional metadata.

        Returns:
            Tuple of (state_id, is_new). is_new is False if state was deduplicated.
        """
        # Compute fingerprint
        fingerprint = self.fingerprinter.compute_fingerprint(
            url, dom_content, screenshot_path
        )

        # Check for existing state with same fingerprint
        existing_id = self._fingerprint_index.get(fingerprint.fast_hash)
        if existing_id:
            # Update visit count
            self._states[existing_id].visit_count += 1
            return (existing_id, False)

        # Check for similar states
        for state_id, state in self._states.items():
            is_equiv, confidence = self.fingerprinter.are_states_equivalent(
                fingerprint, state.fingerprint
            )
            if is_equiv:
                # Deduplicate
                self._states[state_id].visit_count += 1
                # Also index this fingerprint to the existing state
                self._fingerprint_index[fingerprint.fast_hash] = state_id
                return (state_id, False)

        # New unique state
        state = StateNode(
            fingerprint=fingerprint,
            url=url,
            dom_snapshot_path=self._save_dom_snapshot(dom_content) if dom_content else None,
            screenshot_path=screenshot_path,
            user_context=user_context,
            discovery_depth=discovery_depth,
            metadata=metadata or {}
        )

        self._states[state.id] = state
        self._fingerprint_index[fingerprint.fast_hash] = state.id
        self.graph.add_node(state.id, **state.model_dump(mode="json"))

        return (state.id, True)

    def add_transition(
        self,
        from_state: str,
        to_state: str,
        action: Action,
        duration_ms: int = 0
    ) -> Transition:
        """Add a transition between states.

        Args:
            from_state: Source state ID.
            to_state: Target state ID.
            action: Action that caused the transition.
            duration_ms: Transition duration in milliseconds.

        Returns:
            The created Transition.
        """
        transition = Transition(
            from_state=from_state,
            to_state=to_state,
            action=action,
            user=action.user,
            duration_ms=duration_ms
        )

        self.graph.add_edge(
            from_state,
            to_state,
            **transition.model_dump(mode="json")
        )

        return transition

    def get_state(self, state_id: str) -> Optional[StateNode]:
        """Get a state by ID.

        Args:
            state_id: State ID.

        Returns:
            StateNode or None if not found.
        """
        return self._states.get(state_id)

    def get_transitions_from(self, state_id: str) -> list[Transition]:
        """Get all transitions from a state.

        Args:
            state_id: Source state ID.

        Returns:
            List of transitions.
        """
        transitions = []
        for _, to_state, data in self.graph.out_edges(state_id, data=True):
            transitions.append(Transition(**data))
        return transitions

    def get_transitions_to(self, state_id: str) -> list[Transition]:
        """Get all transitions to a state.

        Args:
            state_id: Target state ID.

        Returns:
            List of transitions.
        """
        transitions = []
        for from_state, _, data in self.graph.in_edges(state_id, data=True):
            transitions.append(Transition(**data))
        return transitions

    def find_paths(
        self,
        from_state: str,
        to_state: str,
        max_paths: int = 10
    ) -> list[list[str]]:
        """Find all paths between two states.

        Args:
            from_state: Source state ID.
            to_state: Target state ID.
            max_paths: Maximum number of paths to return.

        Returns:
            List of paths (each path is a list of state IDs).
        """
        try:
            paths = list(nx.all_simple_paths(
                self.graph, from_state, to_state, cutoff=20
            ))
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []

    def get_unexplored_states(self, min_depth: int = 0) -> list[StateNode]:
        """Get states that haven't been fully explored.

        Args:
            min_depth: Minimum depth to consider.

        Returns:
            List of states with few outgoing transitions.
        """
        unexplored = []
        for state_id, state in self._states.items():
            if state.discovery_depth >= min_depth:
                out_degree = self.graph.out_degree(state_id)
                element_count = len(state.interactive_elements)
                # State is unexplored if it has more elements than outgoing transitions
                if out_degree < element_count:
                    unexplored.append(state)
        return unexplored

    def get_coverage_stats(self) -> dict:
        """Get exploration coverage statistics.

        Returns:
            Dictionary with coverage metrics.
        """
        total_states = len(self._states)
        total_transitions = self.graph.number_of_edges()
        total_elements = sum(
            len(s.interactive_elements) for s in self._states.values()
        )

        return {
            "total_states": total_states,
            "total_transitions": total_transitions,
            "total_elements": total_elements,
            "avg_transitions_per_state": total_transitions / max(total_states, 1),
            "coverage_ratio": total_transitions / max(total_elements, 1),
            "max_depth": max((s.discovery_depth for s in self._states.values()), default=0),
            "isolated_states": len(list(nx.isolates(self.graph)))
        }

    def save(self, path: Optional[str] = None) -> str:
        """Save the graph to JSON.

        Args:
            path: Output path. Uses storage_path if not provided.

        Returns:
            Path where graph was saved.
        """
        save_path = path or self.storage_path or "state_graph.json"

        data = {
            "states": {sid: s.model_dump(mode="json") for sid, s in self._states.items()},
            "edges": [
                {
                    "from": u,
                    "to": v,
                    **d
                }
                for u, v, d in self.graph.edges(data=True)
            ],
            "fingerprint_index": self._fingerprint_index,
            "saved_at": datetime.now().isoformat()
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return save_path

    def load(self, path: Optional[str] = None) -> "StateGraph":
        """Load the graph from JSON.

        Args:
            path: Input path. Uses storage_path if not provided.

        Returns:
            Self for chaining.
        """
        load_path = path or self.storage_path
        if not load_path or not Path(load_path).exists():
            return self

        with open(load_path, "r") as f:
            data = json.load(f)

        # Restore states
        for state_id, state_data in data.get("states", {}).items():
            state_data["fingerprint"] = StateFingerprint(**state_data["fingerprint"])
            self._states[state_id] = StateNode(**state_data)
            self.graph.add_node(state_id, **state_data)

        # Restore edges
        for edge in data.get("edges", []):
            self.graph.add_edge(edge["from"], edge["to"], **edge)

        # Restore index
        self._fingerprint_index = data.get("fingerprint_index", {})

        return self

    def _save_dom_snapshot(self, dom_content: str) -> Optional[str]:
        """Save DOM snapshot to file.

        Args:
            dom_content: HTML content.

        Returns:
            Path to saved file.
        """
        if not self.storage_path:
            return None

        snapshots_dir = Path(self.storage_path).parent / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = snapshots_dir / f"dom_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.html"
        with open(snapshot_path, "w") as f:
            f.write(dom_content)

        return str(snapshot_path)

    def to_mermaid(self) -> str:
        """Export graph as Mermaid diagram.

        Returns:
            Mermaid flowchart string.
        """
        lines = ["flowchart TD"]

        for state_id, state in self._states.items():
            short_id = state_id[:8]
            label = state.url.split("/")[-1] or "home"
            lines.append(f'    {short_id}["{label}"]')

        for u, v, data in self.graph.edges(data=True):
            action_type = data.get("action", {}).get("action_type", "?")
            lines.append(f"    {u[:8]} -->|{action_type}| {v[:8]}")

        return "\n".join(lines)
