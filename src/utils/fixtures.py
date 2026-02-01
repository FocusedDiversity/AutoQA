"""Test data fixture generation."""

import random
import string
from datetime import datetime, timedelta
from typing import Any, Callable, Optional


class FixtureFactory:
    """Factory for generating test data fixtures.

    Provides consistent, realistic test data for:
    - User profiles
    - Form inputs
    - Edge cases
    - Randomized but reproducible data
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the factory.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        self._custom_generators: dict[str, Callable] = {}

    def register_generator(self, name: str, generator: Callable[[], Any]) -> None:
        """Register a custom data generator.

        Args:
            name: Generator name.
            generator: Callable that returns generated data.
        """
        self._custom_generators[name] = generator

    def generate(self, field_type: str, **kwargs) -> Any:
        """Generate data for a field type.

        Args:
            field_type: Type of field (email, name, text, etc.).
            **kwargs: Additional parameters.

        Returns:
            Generated data.
        """
        # Check custom generators first
        if field_type in self._custom_generators:
            return self._custom_generators[field_type]()

        # Built-in generators
        generators = {
            "email": self._generate_email,
            "name": self._generate_name,
            "first_name": self._generate_first_name,
            "last_name": self._generate_last_name,
            "phone": self._generate_phone,
            "password": self._generate_password,
            "username": self._generate_username,
            "text": self._generate_text,
            "paragraph": self._generate_paragraph,
            "number": self._generate_number,
            "date": self._generate_date,
            "url": self._generate_url,
            "uuid": self._generate_uuid,
            "boolean": self._generate_boolean,
            "choice": lambda: self._generate_choice(kwargs.get("choices", [])),
        }

        generator = generators.get(field_type, self._generate_text)
        return generator()

    def _generate_email(self) -> str:
        """Generate a random email address."""
        domains = ["example.com", "test.com", "autoqa.test"]
        name = self._generate_username()
        return f"{name}@{random.choice(domains)}"

    def _generate_name(self) -> str:
        """Generate a full name."""
        return f"{self._generate_first_name()} {self._generate_last_name()}"

    def _generate_first_name(self) -> str:
        """Generate a first name."""
        names = [
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
            "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo"
        ]
        return random.choice(names)

    def _generate_last_name(self) -> str:
        """Generate a last name."""
        names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones",
            "Garcia", "Miller", "Davis", "Wilson", "Taylor"
        ]
        return random.choice(names)

    def _generate_phone(self) -> str:
        """Generate a phone number."""
        return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

    def _generate_password(self) -> str:
        """Generate a secure password."""
        chars = string.ascii_letters + string.digits + "!@#$%"
        password = "".join(random.choice(chars) for _ in range(12))
        # Ensure at least one of each type
        password = (
            random.choice(string.ascii_uppercase) +
            random.choice(string.ascii_lowercase) +
            random.choice(string.digits) +
            random.choice("!@#$%") +
            password[4:]
        )
        return password

    def _generate_username(self) -> str:
        """Generate a username."""
        adjectives = ["quick", "lazy", "happy", "smart", "cool"]
        nouns = ["fox", "dog", "cat", "user", "tester"]
        return f"{random.choice(adjectives)}_{random.choice(nouns)}_{random.randint(1, 999)}"

    def _generate_text(self) -> str:
        """Generate short text."""
        words = ["test", "sample", "input", "data", "value", "content"]
        return " ".join(random.choices(words, k=random.randint(2, 5)))

    def _generate_paragraph(self) -> str:
        """Generate a paragraph of text."""
        sentences = []
        for _ in range(random.randint(3, 5)):
            words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
            sentence = " ".join(random.choices(words, k=random.randint(5, 10)))
            sentences.append(sentence.capitalize() + ".")
        return " ".join(sentences)

    def _generate_number(self) -> int:
        """Generate a random number."""
        return random.randint(1, 1000)

    def _generate_date(self) -> str:
        """Generate a date string."""
        base = datetime.now()
        offset = random.randint(-365, 365)
        date = base + timedelta(days=offset)
        return date.strftime("%Y-%m-%d")

    def _generate_url(self) -> str:
        """Generate a URL."""
        domains = ["example.com", "test.org", "sample.net"]
        paths = ["page", "item", "resource", "content"]
        return f"https://{random.choice(domains)}/{random.choice(paths)}/{random.randint(1, 100)}"

    def _generate_uuid(self) -> str:
        """Generate a UUID-like string."""
        import uuid
        return str(uuid.uuid4())

    def _generate_boolean(self) -> bool:
        """Generate a random boolean."""
        return random.choice([True, False])

    def _generate_choice(self, choices: list) -> Any:
        """Generate a random choice from list."""
        if not choices:
            return None
        return random.choice(choices)

    def generate_user_profile(self) -> dict:
        """Generate a complete user profile."""
        return {
            "first_name": self._generate_first_name(),
            "last_name": self._generate_last_name(),
            "email": self._generate_email(),
            "phone": self._generate_phone(),
            "username": self._generate_username(),
        }

    def generate_edge_cases(self, field_type: str) -> list[Any]:
        """Generate edge case values for a field type.

        Args:
            field_type: Type of field.

        Returns:
            List of edge case values.
        """
        common_edge_cases = [
            "",  # Empty
            " ",  # Whitespace only
            "   leading and trailing   ",  # Extra whitespace
            "a" * 256,  # Long input
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection
            "Test\nWith\nNewlines",  # Newlines
            "Unicode: 你好 مرحبا",  # Unicode
            "Emoji: 😀🎉",  # Emoji
            None,  # Null
        ]

        type_specific = {
            "email": [
                "notanemail",
                "missing@domain",
                "@nodomain.com",
                "spaces in@email.com",
                "valid@test.com" * 10,  # Very long
            ],
            "phone": [
                "123",  # Too short
                "abcdefghij",  # Letters
                "+1 (555) 123-4567",  # Formatted
                "555.123.4567",  # Dots
            ],
            "number": [
                0,
                -1,
                -999999,
                999999999,
                1.5,  # Float
                "NaN",  # Not a number string
            ],
            "password": [
                "short",  # Too short
                "nouppercase123!",
                "NOLOWERCASE123!",
                "NoSpecialChars1",
                "NoNumbers!!!",
            ],
            "date": [
                "2000-01-01",  # Old date
                "2099-12-31",  # Future date
                "1900-01-01",  # Very old
                "invalid-date",
                "02/30/2024",  # Invalid day
            ],
        }

        return common_edge_cases + type_specific.get(field_type, [])

    def generate_form_data(self, fields: dict[str, str]) -> dict[str, Any]:
        """Generate data for a form.

        Args:
            fields: Dictionary mapping field names to types.

        Returns:
            Dictionary of generated field values.
        """
        return {
            name: self.generate(field_type)
            for name, field_type in fields.items()
        }

    def generate_bulk(self, field_type: str, count: int) -> list[Any]:
        """Generate multiple values.

        Args:
            field_type: Type of field.
            count: Number of values.

        Returns:
            List of generated values.
        """
        return [self.generate(field_type) for _ in range(count)]
