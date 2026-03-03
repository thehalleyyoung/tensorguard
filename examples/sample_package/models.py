"""Data models demonstrating refinement type inference."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class User:
    """A user with validated fields."""

    name: str
    age: int
    email: Optional[str] = None

    def __post_init__(self) -> None:
        if self.age < 0:
            raise ValueError(f"Age must be non-negative, got {self.age}")
        if not self.name:
            raise ValueError("Name must not be empty")

    def is_adult(self) -> bool:
        return self.age >= 18


@dataclass
class Team:
    """A team containing users."""

    name: str
    members: List[User]

    def average_age(self) -> float:
        if len(self.members) == 0:
            raise ValueError("Cannot compute average age of empty team")
        return sum(m.age for m in self.members) / len(self.members)

    def find_member(self, name: str) -> Optional[User]:
        for m in self.members:
            if m.name == name:
                return m
        return None
