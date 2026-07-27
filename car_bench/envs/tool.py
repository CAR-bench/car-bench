import abc
from typing import Any


class Tool(abc.ABC):
    @staticmethod
    def reset() -> None:
        """Reset task-scoped tool state before a new environment run."""
        return None

    @staticmethod
    def invoke(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError
