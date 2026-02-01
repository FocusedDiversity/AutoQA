"""Hybrid state fingerprinting for deduplication."""

import hashlib
import re
from typing import Optional
from urllib.parse import urlparse

from ..core.models import StateFingerprint


class StateFingerprinter:
    """Hybrid state fingerprinting using fast hash + optional LLM verification.

    Tier 1 (Fast): DOM-based hash computed from URL, key elements, and text
    Tier 2 (Accurate): LLM semantic verification when similarity is uncertain
    """

    def __init__(self, llm_engine=None, similarity_threshold: float = 0.95):
        """Initialize the fingerprinter.

        Args:
            llm_engine: Optional LLM engine for semantic verification.
            similarity_threshold: Threshold above which states are considered equal.
        """
        self.llm_engine = llm_engine
        self.similarity_threshold = similarity_threshold
        self._key_selectors = [
            "h1", "h2", "h3",
            "nav", "header", "main",
            "form", "[role='main']", "[role='navigation']"
        ]

    def compute_fingerprint(
        self,
        url: str,
        dom_content: str,
        screenshot_path: Optional[str] = None
    ) -> StateFingerprint:
        """Compute a fingerprint for a page state.

        Args:
            url: The page URL.
            dom_content: The DOM HTML content.
            screenshot_path: Optional path to screenshot for semantic analysis.

        Returns:
            StateFingerprint with fast hash and optional semantic signature.
        """
        fast_hash = self._compute_fast_hash(url, dom_content)

        semantic_signature = None
        if self.llm_engine and screenshot_path:
            semantic_signature = self._compute_semantic_signature(screenshot_path)

        return StateFingerprint(
            fast_hash=fast_hash,
            semantic_signature=semantic_signature,
            similarity_threshold=self.similarity_threshold
        )

    def _compute_fast_hash(self, url: str, dom_content: str) -> str:
        """Compute a fast deterministic hash from URL and DOM.

        Args:
            url: The page URL.
            dom_content: The DOM HTML content.

        Returns:
            16-character hex hash.
        """
        components = [
            self._normalize_url(url),
            self._extract_key_elements(dom_content),
            self._hash_visible_text(dom_content)
        ]

        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing dynamic parts.

        Args:
            url: Raw URL.

        Returns:
            Normalized URL path.
        """
        parsed = urlparse(url)
        path = parsed.path

        # Remove common dynamic segments (IDs, UUIDs, timestamps)
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        path = re.sub(r'/\d+', '/{id}', path)

        return path

    def _extract_key_elements(self, dom_content: str) -> str:
        """Extract key structural elements from DOM.

        Args:
            dom_content: HTML content.

        Returns:
            Hash of key element structure.
        """
        # Simple extraction - in production, use proper HTML parser
        key_patterns = []

        for selector in self._key_selectors:
            # Count occurrences of key elements
            if selector.startswith("["):
                # Attribute selector
                attr = selector[1:-1].replace("'", "")
                pattern = rf'{attr}[="\']([^"\']*)["\']'
            else:
                # Tag selector
                pattern = rf'<{selector}[^>]*>'

            matches = re.findall(pattern, dom_content, re.IGNORECASE)
            key_patterns.append(f"{selector}:{len(matches)}")

        return hashlib.md5("|".join(key_patterns).encode()).hexdigest()[:8]

    def _hash_visible_text(self, dom_content: str) -> str:
        """Hash the visible text content.

        Args:
            dom_content: HTML content.

        Returns:
            Hash of visible text.
        """
        # Strip HTML tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', dom_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Take first 1000 chars to avoid memory issues
        text = text[:1000]

        return hashlib.md5(text.encode()).hexdigest()[:8]

    def _compute_semantic_signature(self, screenshot_path: str) -> Optional[str]:
        """Use LLM to generate a semantic description of the state.

        Args:
            screenshot_path: Path to screenshot image.

        Returns:
            Semantic description string, or None if LLM unavailable.
        """
        if not self.llm_engine:
            return None

        prompt = """Describe this UI state in one concise sentence.
        Focus on: what page/view it is, key visible elements, and any active states.
        Example: "Dashboard showing 3 tasks with filter set to 'My Tasks'"
        """

        try:
            return self.llm_engine.analyze_image(screenshot_path, prompt)
        except Exception:
            return None

    def compute_similarity(self, fp1: StateFingerprint, fp2: StateFingerprint) -> float:
        """Compute similarity between two fingerprints.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.

        Returns:
            Similarity score between 0 and 1.
        """
        if fp1.fast_hash == fp2.fast_hash:
            return 1.0

        # Compare hash character by character for partial similarity
        matching = sum(c1 == c2 for c1, c2 in zip(fp1.fast_hash, fp2.fast_hash))
        return matching / len(fp1.fast_hash)

    def are_states_equivalent(
        self,
        fp1: StateFingerprint,
        fp2: StateFingerprint,
        force_llm: bool = False
    ) -> tuple[bool, float]:
        """Determine if two states are equivalent.

        Args:
            fp1: First state fingerprint.
            fp2: Second state fingerprint.
            force_llm: Force LLM verification regardless of similarity.

        Returns:
            Tuple of (is_equivalent, confidence_score).
        """
        similarity = self.compute_similarity(fp1, fp2)

        # Clear cases - no LLM needed
        if similarity < 0.70:
            return (False, 1.0 - similarity)

        if similarity > self.similarity_threshold and not force_llm:
            return (True, similarity)

        # Uncertain case (0.70-0.95) - use LLM if available
        if self.llm_engine and fp1.semantic_signature and fp2.semantic_signature:
            return self._verify_with_llm(fp1, fp2)

        # Default to similarity-based decision
        return (similarity > 0.85, similarity)

    def _verify_with_llm(
        self,
        fp1: StateFingerprint,
        fp2: StateFingerprint
    ) -> tuple[bool, float]:
        """Use LLM to verify state equivalence.

        Args:
            fp1: First state fingerprint.
            fp2: Second state fingerprint.

        Returns:
            Tuple of (is_equivalent, confidence).
        """
        prompt = f"""Compare these two UI state descriptions and determine if they represent the same logical state.

State A: {fp1.semantic_signature}
State B: {fp2.semantic_signature}

Answer with just 'yes' or 'no', followed by a confidence score (0.0-1.0).
Example: "yes 0.95" or "no 0.88"
"""

        try:
            response = self.llm_engine.complete(prompt)
            parts = response.strip().lower().split()

            is_equivalent = parts[0] == "yes"
            confidence = float(parts[1]) if len(parts) > 1 else 0.8

            return (is_equivalent, confidence)
        except Exception:
            # Fall back to similarity
            similarity = self.compute_similarity(fp1, fp2)
            return (similarity > 0.85, similarity)
