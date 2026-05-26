"""Явный список встроенных инструментов харнесса."""

from .files import LIST_FILES_SPEC, READ_FILE_SPEC, WRITE_FILE_SPEC
from .images import READ_IMAGE_SPEC
from .patch import APPLY_PATCH_SPEC
from .search import SEARCH_CODE_SPEC
from .shell import RUN_SHELL_SPEC
from .specs import ToolSpec


def builtin_specs() -> list[ToolSpec]:
    """Возвращаем встроенные инструменты в стабильном порядке для модели.

    Чтобы добавить новый встроенный инструмент, создаём ToolSpec рядом с handler
    и добавляем его в этот список.
    """

    return [
        LIST_FILES_SPEC,
        READ_FILE_SPEC,
        READ_IMAGE_SPEC,
        WRITE_FILE_SPEC,
        APPLY_PATCH_SPEC,
        SEARCH_CODE_SPEC,
        RUN_SHELL_SPEC,
    ]
