"""Слой контекста: фрагменты, история и сообщения для модели."""

from .fragments import ContextFragment, ContextProvider, ContextState
from .history import FileRef, HistoryEntry
from .manager import ContextManager
from .summary import ReasoningSummarizer

__all__ = [
    "ContextFragment",
    "ContextManager",
    "ContextProvider",
    "ContextState",
    "FileRef",
    "HistoryEntry",
    "ReasoningSummarizer",
]
