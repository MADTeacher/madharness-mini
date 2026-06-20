"""Проверки безопасности путей и shell-команд перед инструментами."""

import shlex
from pathlib import Path

from .config import Config


class Policy:
    """Решаем, можно ли агенту трогать путь или запускать команду.

    Все файловые инструменты сначала прогоняют путь через safe_path;
    run_shell — через shell_allowed.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.root
        self.protected = list(cfg.data["protected_paths"])

    def safe_path(self, raw: str) -> tuple[Path | None, str | None]:
        """Проверяем относительный путь внутри workspace_root.

        Успех: (абсолютный Path, None). Отказ: (None, причина) — пустой путь,
        выход за root, попадание в protected_paths (.git, .env и т.д.).
        """

        if not raw:
            return None, "empty path"
        path = (self.root / raw).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return None, f"path outside workspace: {raw}"
        rel_parts = set(path.relative_to(self.root).parts)
        for item in self.protected:
            expanded = Path(item).expanduser()
            if expanded.is_absolute():
                try:
                    path.relative_to(expanded.resolve())
                    return None, f"protected path: {raw}"
                except ValueError:
                    pass
            name = item.strip("/").split("/")[-1]
            if name and name in rel_parts:
                return None, f"protected path: {raw}"
        return path, None

    def skill_root(self, raw: str) -> tuple[Path | None, str | None]:
        """Проверяем фиксированный каталог навыков внутри workspace.

        Discovery читает только `.madharness_mini/skills` и `.agents/skills`.
        Эти каталоги не проходят через protected_paths: они не являются целями
        пользовательских файловых инструментов, но всё равно обязаны оставаться
        внутри workspace_root.
        """

        path = (self.root / raw).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return None, f"skill root outside workspace: {raw}"
        return path, None

    def shell_allowed(self, command: str) -> tuple[bool, str]:
        """Решаем, можно ли выполнить одну простую shell-команду в workspace.

        Запрещаем пайпы, редиректы, sudo, rm -rf и похожие фрагменты;
        allow_shell=false в конфиге отключает shell целиком.
        Команда выполняется без shell-интерпретатора (через shlex + execve),
        поэтому фигурные скобки и обратные кавычки тоже запрещены вне кавычек:
        их раскрыл бы только bash, а без него они ушли бы в программу литералом
        (например, mkdir создал бы каталог с именем из нераскрытых скобок).
        """

        if not self.cfg.data.get("allow_shell", True):
            return False, "shell disabled by config"
        lowered = command.lower()
        denied = [
            "rm -rf",
            "sudo",
            "curl ",
            "wget ",
            "ssh ",
            "scp ",
            "chmod 777",
            "mkfs",
            " dd ",
        ]
        if any(fragment in f" {lowered} " for fragment in denied):
            return False, "risky shell command denied"
        # Метасимволы, которые раскрывает только shell: brace expansion и
        # command substitution. Без интерпретатора они становятся мусором в
        # аргументах. Обратные кавычки блокируем безусловно: в bash command
        # substitution раскрывается даже внутри двойных кавычек, а литералом
        # нужна редко. Фигурные скобки — только вне кавычек: внутри одинарных
        # или двойных bash их не раскрывает, и через execve они нужны литералом
        # (например, awk '{print $1}'). Глоббинг (*, ?, []) и $VAR намеренно не
        # блокируем — они часто полезны как литерал для find/grep/awk.
        if "`" in command or _has_unquoted(command, ["{", "}"]):
            return (
                False,
                "shell metacharacters not supported without a shell: "
                "pass literal arguments (no brace expansion or command substitution)",
            )
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return False, f"invalid shell command: {exc}"
        if not args:
            return False, "empty shell command"
        if any(token in command for token in ["|", ">", "<", "&&", "||", ";"]):
            return False, "shell control operators are denied"
        return True, ""


def _has_unquoted(command: str, chars: set[str] | list[str]) -> bool:
    """Есть ли среди символов chars хоть один вне одинарных/двойных кавычек.

    Простой обход строки с учётом переключения режима кавычек (как в bash);
    экранирование обратной чертой намеренно не учитываем — редкий случай для
    разрешённых harness-команд, а консервативная проверка безопаснее.
    """

    target = set(chars)
    in_single = False
    in_double = False
    for ch in command:
        if in_single:
            if ch == "'":
                in_single = False
        elif in_double:
            if ch == '"':
                in_double = False
        elif ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch in target:
            return True
    return False
