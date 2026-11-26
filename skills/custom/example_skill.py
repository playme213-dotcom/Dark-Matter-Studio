"""Example custom skill for Dark Matter Studio.

This file demonstrates how custom skills work. Place your own .py files
in skills/custom/ and they will be auto-loaded (if they pass tests).
"""


def greet(name: str = "World") -> str:
    """Greets a person by name."""
    return f"Hello, {name}! This is a custom skill."


def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


def test_greet() -> bool:
    """Test that greet works."""
    result = greet("Test")
    return "Test" in result


def test_add_numbers() -> bool:
    """Test that add_numbers works."""
    return add_numbers(2, 3) == 5
