"""Слой контекста: фрагменты, история и сообщения для модели."""

from .fragments import ContextFragment, ContextProvider, ContextState
from .history import FileRef, HistoryEntry
from .manager import ContextManager

__all__ = [
    "ContextFragment",
    "ContextManager",
    "ContextProvider",
    "ContextState",
    "FileRef",
    "HistoryEntry",
]
