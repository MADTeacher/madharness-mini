"""Инструмент чтения изображений для vision-моделей."""

import base64
from pathlib import Path
from typing import Any

from ..config import IMAGE_DETAIL_VALUES
from ..utils import fail, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec

IMAGE_MIME_BY_EXT = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def normalize_image_detail(value: Any) -> tuple[str | None, str | None]:
    """Проверяем detail до отправки запроса модели.

    Провайдеры могут по-разному поддерживать этот параметр, но внутри харнесса
    держим только документированные значения, чтобы опечатка не дошла до API.
    """

    detail = str(value or "auto").strip()
    if detail not in IMAGE_DETAIL_VALUES:
        allowed = ", ".join(sorted(IMAGE_DETAIL_VALUES))
        return None, f"invalid image detail: {detail}; allowed: {allowed}"
    return detail, None


def read_image(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Проверяем локальное изображение и сообщаем модели, доступно ли vision-вложение.

    Сами байты не возвращаем в observation: иначе base64 попал бы в историю tool
    и JSONL-трассу. Петля run_agent прикрепит изображение отдельно, если конфиг
    разрешает мультимодальный ввод.
    """

    path, err = ctx.policy.safe_path(args["path"])
    if err:
        return fail("read_image", err)
    if not path or not path.is_file():
        return fail("read_image", f"not a file: {args['path']}")
    detail, err = normalize_image_detail(
        args.get("detail") or ctx.cfg.data.get("image_detail")
    )
    if err:
        return fail("read_image", err)
    size = path.stat().st_size
    limit = int(ctx.cfg.data.get("max_image_bytes", 0))
    if size > limit:
        return fail(
            "read_image",
            f"image is too large: {size} bytes > {limit}",
            path=args["path"],
            bytes=size,
            max_image_bytes=limit,
        )
    data = path.read_bytes()
    mime_type, err = detect_image_mime(path, data)
    if err:
        return fail("read_image", err, path=args["path"], bytes=size)
    attached = bool(ctx.cfg.data.get("supports_image_input"))
    extra: dict[str, Any] = {}
    if attached:
        extra["_followup_messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Image from read_image is attached: {args['path']}",
                    },
                    image_content_part(data, mime_type, detail),
                ],
            }
        ]
    reason = (
        "image will be attached to the next model request"
        if attached
        else "supports_image_input is false"
    )
    return ok(
        "read_image",
        f"read image metadata for {args['path']}",
        path=args["path"],
        mime_type=mime_type,
        bytes=size,
        attached=attached,
        detail=detail,
        reason=reason,
        **extra,
    )


def detect_image_mime(path: Path, data: bytes) -> tuple[str | None, str | None]:
    """Определяем поддержанный MIME по расширению и сигнатуре файла."""

    ext = path.suffix.lower()
    mime_type = IMAGE_MIME_BY_EXT.get(ext)
    if not mime_type:
        return None, f"unsupported image type: {path.suffix or '(no extension)'}"
    if mime_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return mime_type, None
    if mime_type == "image/jpeg" and data.startswith(b"\xff\xd8\xff"):
        return mime_type, None
    if mime_type == "image/webp" and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return mime_type, None
    if mime_type == "image/gif" and data.startswith((b"GIF87a", b"GIF89a")):
        animated, err = gif_is_animated(data)
        if err:
            return None, err
        if animated:
            return None, "animated GIF is not supported"
        return mime_type, None
    return None, f"file signature does not match {mime_type}"


def gif_is_animated(data: bytes) -> tuple[bool, str | None]:
    """Считаем кадры GIF без внешних библиотек.

    Нам не нужен полноценный декодер: достаточно пройти блоки контейнера и
    остановиться, если найдено больше одного image descriptor.
    """

    if len(data) < 13:
        return False, "invalid GIF: header is too short"
    pos = 13
    packed = data[10]
    if packed & 0b10000000:
        pos += 3 * (2 ** ((packed & 0b00000111) + 1))
    frames = 0
    while pos < len(data):
        marker = data[pos]
        pos += 1
        if marker == 0x3B:
            return frames > 1, None
        if marker == 0x21:
            if pos >= len(data):
                return False, "invalid GIF: truncated extension"
            pos += 1
            pos, err = skip_gif_subblocks(data, pos)
            if err:
                return False, err
            continue
        if marker == 0x2C:
            frames += 1
            if frames > 1:
                return True, None
            if pos + 9 > len(data):
                return False, "invalid GIF: truncated image descriptor"
            packed = data[pos + 8]
            pos += 9
            if packed & 0b10000000:
                pos += 3 * (2 ** ((packed & 0b00000111) + 1))
            if pos >= len(data):
                return False, "invalid GIF: missing image data"
            pos += 1
            pos, err = skip_gif_subblocks(data, pos)
            if err:
                return False, err
            continue
        return False, "invalid GIF: unexpected block marker"
    return False, "invalid GIF: missing trailer"


def skip_gif_subblocks(data: bytes, pos: int) -> tuple[int, str | None]:
    """Пропускаем GIF sub-block sequence: длина, данные, нулевой терминатор."""

    while pos < len(data):
        size = data[pos]
        pos += 1
        if size == 0:
            return pos, None
        pos += size
        if pos > len(data):
            return pos, "invalid GIF: truncated sub-block"
    return pos, "invalid GIF: missing sub-block terminator"


def image_content_part(data: bytes, mime_type: str, detail: str) -> dict[str, Any]:
    """Готовим image_url part для Chat Completions, не сохраняя его в observation."""

    data_url = "data:{};base64,{}".format(
        mime_type,
        base64.b64encode(data).decode("ascii"),
    )
    return {
        "type": "image_url",
        "image_url": {"url": data_url, "detail": detail},
    }


READ_IMAGE_DESCRIPTION = """Read an image file from the workspace for visual inspection.

Use this for screenshots and other local images. The tool validates a supported
PNG, JPEG, WEBP, or non-animated GIF and returns metadata only. If
supports_image_input is true, the harness attaches the image to the next model
request; if false, you must not claim you visually inspected the pixels.
"""

READ_IMAGE_SPEC = ToolSpec(
    "read_image",
    READ_IMAGE_DESCRIPTION,
    obj(
        {
            "path": strp(
                req=True,
                desc="Workspace-relative image path to inspect.",
            ),
            "detail": {
                "type": "string",
                "enum": sorted(IMAGE_DETAIL_VALUES),
                "default": "auto",
                "description": "Vision detail level to send when image"
                " input is enabled.",
            },
        },
        ["path"],
    ),
    read_image,
)
