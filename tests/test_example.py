"""Example test file demonstrating test structure."""

import pytest


class TestExample:
    """Example test class."""

    def test_addition(self):
        """Test basic addition."""
        assert 1 + 1 == 2

    def test_string_concat(self):
        """Test string concatenation."""
        assert "hello" + " " + "world" == "hello world"

    @pytest.mark.skip(reason="Example of skipped test")
    def test_skipped(self):
        """This test is skipped."""
        pass

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_double(self, input, expected):
        """Test parameterized doubling."""
        assert input * 2 == expected
