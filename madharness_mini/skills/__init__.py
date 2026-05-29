"""Поддержка проектных Agent Skills без внешних зависимостей."""

from .activation import SkillRuntime
from .catalog import ExplicitSkillSelection, find_explicit_skill_selection
from .loader import discover_skills
from .provider import SkillCatalogProvider, SkillToolProvider
from .types import Skill, SkillDiagnostic, SkillIndex

__all__ = [
    "ExplicitSkillSelection",
    "Skill",
    "SkillCatalogProvider",
    "SkillDiagnostic",
    "SkillIndex",
    "SkillRuntime",
    "SkillToolProvider",
    "discover_skills",
    "find_explicit_skill_selection",
]
