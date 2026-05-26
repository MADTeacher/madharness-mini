"""Явный список встроенных инструментов харнесса."""

from .files import LIST_FILES_SPEC, READ_FILE_SPEC, WRITE_FILE_SPEC
from .patch import APPLY_PATCH_SPEC
from .search import SEARCH_CODE_SPEC
from .shell import RUN_SHELL_SPEC
from .specs import ToolSpec


class BuiltinToolProvider:
    """Provider встроенных инструментов в стабильном учебном порядке."""

    def specs(self, ctx: object) -> list[ToolSpec]:
        """Отдаём стандартный набор инструментов; ctx пока не нужен."""

        return [
            LIST_FILES_SPEC,
            READ_FILE_SPEC,
            WRITE_FILE_SPEC,
            APPLY_PATCH_SPEC,
            SEARCH_CODE_SPEC,
            RUN_SHELL_SPEC,
        ]
