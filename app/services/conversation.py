"""
Conversation memory - stores chat history per conversation.
"""

from collections import defaultdict
from typing import Any


class ConversationMemory:
    """Simple in-memory conversation store. Replace with Redis/DB for production."""

    def __init__(self, max_turns: int = 10):
        self._history: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._max_turns = max_turns

    def add_turn(self, conversation_id: str, role: str, content: str):
        self._history[conversation_id].append({"role": role, "content": content})
        # Keep only last N turns
        if len(self._history[conversation_id]) > self._max_turns * 2:
            self._history[conversation_id] = self._history[conversation_id][-self._max_turns * 2:]

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        return self._history.get(conversation_id, [])

    def format_history(self, conversation_id: str) -> str:
        history = self.get_history(conversation_id)
        if not history:
            return ""

        formatted = []
        for turn in history[-self._max_turns * 2:]:
            if turn["role"] == "user":
                formatted.append(f"Q: {turn['content']}")
            else:
                formatted.append(f"A: {turn['content']}")

        return "\n".join(formatted)

    def clear(self, conversation_id: str):
        self._history.pop(conversation_id, None)


# Global instance
memory = ConversationMemory()
