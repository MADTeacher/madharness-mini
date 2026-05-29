"""ContextProvider и ToolProvider для раскрытия и активации Agent Skills."""

from __future__ import annotations

from collections.abc import Iterable

from ..context import ContextFragment, ContextState
from ..utils import obj
from ..tools.context import ToolContext
from ..tools.specs import ToolSpec
from .activation import SkillRuntime
from .catalog import render_skill_catalog_for_root
from .types import SkillIndex


class SkillCatalogProvider:
    """Добавляет в `run` компактный каталог навыков без полной загрузки инструкций."""

    def __init__(self, index: SkillIndex, workspace_root):
        self.index = index
        self.workspace_root = workspace_root

    def collect(self, state: ContextState) -> Iterable[ContextFragment]:
        """Возвращаем catalog-фрагмент, если в проекте есть хотя бы один skill."""

        text = render_skill_catalog_for_root(self.index, self.workspace_root)
        if not text:
            return []
        return [
            ContextFragment(
                id="skills:catalog",
                source="project skill catalog",
                text=text,
                priority=30,
                placement="system",
                transient=True,
            )
        ]


class SkillToolProvider:
    """Даёт модели инструмент `activate_skill` с enum найденных имён."""

    def __init__(self, runtime: SkillRuntime):
        self.runtime = runtime

    def specs(self, ctx: ToolContext) -> list[ToolSpec]:
        """Регистрируем `activate_skill` только когда есть доступные навыки."""

        names = self.runtime.index.names()
        if not names:
            return []
        return [
            ToolSpec(
                "activate_skill",
                "Activate one available Agent Skill by exact name. The harness adds the skill instructions to durable context and returns the skill root plus bundled resource list.",
                obj(
                    {
                        "name": {
                            "type": "string",
                            "enum": names,
                            "description": "Exact skill name from the available Agent Skills catalog.",
                        }
                    },
                    ["name"],
                ),
                self.activate_skill,
            )
        ]

    def activate_skill(self, ctx: ToolContext, args: dict) -> dict:
        """Handler инструмента: активация идёт через runtime текущего run."""

        return self.runtime.activate(str(args.get("name") or ""), "tool")
