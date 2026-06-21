"""Проверки безопасности путей и shell-команд перед инструментами."""

import re
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
        """Решаем, можно ли выполнить shell-команду в workspace.

        Два режима исполнения переключаются настройкой allow_shell_interpreter:

        - True (по умолчанию): команда уходит в /bin/sh -c. Операторы && ; ||
          и конструкции |, 2>&1, >/dev/null работают. Это режим, в котором
          агент может связывать обычные проверки (node --version && npm --version,
          npx vitest run 2>&1) и не плодить лишних tool-вызовов.
        - False: прежний режим без интерпретатора. Команда режется через shlex
          и выполняется execve, поэтому control-операторы становятся литералами
          и запрещаются.

        Что блокируется в обоих режимах:
        - чёрный список разрушительных команд (rm -rf, sudo, curl, wget, ssh,
          scp, chmod 777, mkfs, dd);
        - обратные кавычки и фигурные скобки вне кавычек: command substitution
          и brace expansion раскрывает только bash, и почти всегда это либо
          ошибка, либо попытка инъекции;
        - редирект в файлы: > и >> применяются только к /dev/null и
          /dev/std{out,err}. Запись в любой другой файл запрещена — иначе shell
          обходит safe_path и apply_patch/write_file.
        """

        if not self.cfg.data.get("allow_shell", True):
            return False, "shell disabled by config"
        lowered = command.lower()
        # Чёрный список разрушительных команд действует в любом режиме.
        if any(fragment in f" {lowered} " for fragment in _DENYLIST):
            return False, "risky shell command denied"
        # Метасимволы, которые раскрывает только shell и которые почти всегда
        # означают ошибку или инъекцию: command substitution (обратные кавычки,
        # в том числе внутри двойных) и brace expansion (фигурные скобки вне
        # кавычек). Внутри одинарных/двойных кавычек фигурные скобки нужны как
        # литерал для awk/sed, поэтому разрешаем их там.
        if "`" in command or _has_unquoted(command, ["{", "}"]):
            return (
                False,
                "shell metacharacters not supported: "
                "no command substitution (backticks) or brace expansion",
            )
        # Редирект в файлы — отдельная проверка, общая для обоих режимов.
        redir_err = _check_redirects(command)
        if redir_err:
            return False, redir_err
        if self.cfg.data.get("allow_shell_interpreter", True):
            # Режим с интерпретатором: операторы разрешены, ловим только пустоту.
            if not command.strip():
                return False, "empty shell command"
            return True, ""
        # Прежний режим без интерпретатора: shlex + execve. Операторы здесь
        # становятся литералами, поэтому проверяем их явно.
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return False, f"invalid shell command: {exc}"
        if not args:
            return False, "empty shell command"
        if any(token in command for token in ["|", ">", "<", "&&", "||", ";"]):
            return False, "shell control operators are denied"
        return True, ""


# Команды, которые не запускать ни в каком режиме: разрушают workspace или
# выводят данные наружу. Проверяем подстрочно с пробелами по краям, чтобы
# не задеть innocuous-имена (например, "dd" в "adduser").
_DENYLIST = [
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


def _check_redirects(command: str) -> str | None:
    """Запрещаем редирект в файлы, кроме безопасных sink'ов.

    Разрешаем только /dev/null и /dev/std{out,err}: они нужны для подавления
    вывода (node x >/dev/null 2>&1). Любой другой файловый target давал бы
    shell-способ перезаписать исходник мимо apply_patch/write_file и safe_path,
    поэтому блокируем его с подсказкой пользоваться файловыми инструментами.

    Возвращает причину отказа или None, если редиректов нет / все безопасны.
    """

    # Ловим формы: > file, >> file, 2> file, 2>> file, &> file, 2>&1.
    # Цифры перед > и амперсанд не считаются target'ом, их пропускаем в группе.
    targets = re.findall(r"\d?&?>>?\s*([^\s|;&<>]+)", command)
    for raw in targets:
        target = raw.strip("'\"")
        if target in {"&1", "&2"}:
            # 2>&1 / 1>&2 — перенаправление дескрипторов, не файловый target.
            continue
        if target in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
            continue
        return (
            f"shell redirect into file denied ({raw}); "
            "use apply_patch or write_file for file writes, "
            "or redirect to /dev/null"
        )
    return None


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
