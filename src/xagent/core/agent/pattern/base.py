import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional

from ...memory import MemoryStore
from ...tools.adapters.vibe import Tool
from ..context import AgentContext


def notify_condition(condition: asyncio.Condition) -> None:
    """Schedule a notify_all on an asyncio.Condition from sync code.

    Used by pause/resume/interrupt methods that are synchronous but need
    to wake up coroutines blocked on a Condition.
    """

    async def _notify() -> None:
        async with condition:
            condition.notify_all()

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_notify())
    except RuntimeError:
        pass


class AgentPattern(ABC):
    """
    Abstract interface for agent execution patterns (e.g., React, Plan, Reflect).
    Each pattern must implement the 'run' method.
    """

    @abstractmethod
    async def run(
        self,
        task: str,
        memory: MemoryStore,
        tools: list[Tool],
        context: Optional[AgentContext] = None,
    ) -> dict[str, Any]:
        """
        Execute the pattern with given task, memory, tools, and context.

        Returns:
            dict with at least a 'success' boolean field.
        """
