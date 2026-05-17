"""Проверки безопасности для путей и shell-команд агента."""

import shlex
from pathlib import Path

from .config import Config


class Policy:
    """Ограничивает инструменты рамками рабочей папки и безопасных команд."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.root
        self.protected = list(cfg.data["protected_paths"])

    def safe_path(self, raw: str) -> tuple[Path | None, str | None]:
        """Преобразовать пользовательский путь в абсолютный путь внутри workspace.

        Возвращает пару `(path, None)` при успехе или `(None, reason)`, если путь
        пустой, выходит за `workspace_root` или попадает в защищённую область.
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

    def shell_allowed(self, command: str) -> tuple[bool, str]:
        """Проверить, можно ли запускать shell-команду через инструмент агента."""

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
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return False, f"invalid shell command: {exc}"
        if not args:
            return False, "empty shell command"
        if any(token in command for token in ["|", ">", "<", "&&", "||", ";"]):
            return False, "shell control operators are denied"
        return True, ""
