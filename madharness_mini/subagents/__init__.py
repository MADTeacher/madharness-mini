"""Markdown-субагенты: discovery, tool providers и trace-сводки."""

from .loader import discover_subagents, load_subagent_file
from .runtime import summarize_subagent_trace, trace_path_for_observation
from .tools import AskUserToolProvider, OrchestratorToolProvider, effective_tools
from .types import Subagent, SubagentDiagnostic, SubagentIndex

__all__ = [
    "AskUserToolProvider",
    "OrchestratorToolProvider",
    "Subagent",
    "SubagentDiagnostic",
    "SubagentIndex",
    "discover_subagents",
    "effective_tools",
    "load_subagent_file",
    "summarize_subagent_trace",
    "trace_path_for_observation",
]
