"""Общий цикл model/tool вызовов для parent и субагентов."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .context import ContextManager, FileRef
from .hooks import HookDecision, HookManager
from .model import ModelClient, ModelRateLimitError, ModelTransientError
from .tools import ToolRegistry
from .trace import Trace
from .utils import fail, parse_tool_args, paths_from_patch

# При 429 ждём Retry-After, но не дольше этой границы (секунды).
RATE_LIMIT_RETRY_MAX_SECONDS = 60

# Временные сетевые сбои провайдера повторяем коротко, чтобы не терять сессию
# из-за разового TLS/EOF/timeout, но и не зависать надолго в учебном CLI.
TRANSIENT_RETRY_DELAYS_SECONDS = (1, 3)

# После стольких неудачных apply_patch по одному пути подсказываем модели, что
# её память о файле рассинхронизирована, и стоит перечитать файл или сделать
# write_file целиком. Прерывает дорогой цикл read → apply_patch(fail) → repeat.
PATCH_RETRY_HINT_THRESHOLD = 2


def call_model_with_rate_limit_retry(
    client: ModelClient,
    trace: Trace,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **trace_data: Any,
) -> dict[str, Any]:
    """Зовём модель; при коротком 429 один раз ждём и повторяем запрос.

    Длинный Retry-After пробрасываем наверх — пользователь увидит ошибку в CLI.
    """

    try:
        return _call_model_with_transient_retry(client, trace, messages, tools, trace_data)
    except ModelRateLimitError as exc:
        wait_seconds = exc.retry_after_seconds
        if wait_seconds is not None and 0 < wait_seconds <= RATE_LIMIT_RETRY_MAX_SECONDS:
            trace.write(
                "model_rate_limit_retry",
                **trace_data,
                status=exc.status,
                retry_after=exc.retry_after,
                retry_after_seconds=wait_seconds,
            )
            time.sleep(wait_seconds)
            return _call_model_with_transient_retry(
                client,
                trace,
                messages,
                tools,
                trace_data,
            )
        raise


def _call_model_with_transient_retry(
    client: ModelClient,
    trace: Trace,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    trace_data: dict[str, Any],
) -> dict[str, Any]:
    """Повторяем короткие сетевые сбои и пишем каждую попытку в trace."""

    for attempt, delay in enumerate((*TRANSIENT_RETRY_DELAYS_SECONDS, None), 1):
        try:
            return client.chat(messages, tools)
        except ModelTransientError as exc:
            if delay is None:
                raise
            trace.write(
                "model_transient_retry",
                **trace_data,
                attempt=attempt,
                retry_after_seconds=delay,
                error=str(exc),
            )
            time.sleep(delay)
    raise RuntimeError("unreachable transient retry state")


def run_model_loop(
    client: ModelClient,
    trace: Trace,
    context: ContextManager,
    tools_registry: ToolRegistry,
    max_turns: int,
    *,
    stop_on_user_input: bool = False,
    hooks: HookManager | None = None,
    kind: str = "run",
) -> dict[str, Any]:
    """Ведём модель через ходы assistant/tool до финального результата."""

    # Счётчик неудачных apply_patch по пути за текущий запуск. Успешная правка
    # сбрасывает счётчик (модель исправилась). При достижении порога в observation
    # появляется подсказка перечитать файл — прерывает холостой цикл ошибок.
    failed_patches_by_path: dict[str, int] = {}

    for turn in range(max_turns):
        tool_schemas = tools_registry.schemas()
        try:
            messages = context.messages(tool_schemas)
        except RuntimeError as exc:
            trace.write(
                "context_error",
                turn=turn,
                error=str(exc),
                context_report=safe_context_report(context),
            )
            trace.write("session_end", result=f"error: {exc}")
            emit_session_error(hooks, kind, exc, turn=turn)
            raise
        context_report = context.report()
        model_call_id = f"{trace.id}:{turn}"
        trace.write(
            "model_call_started",
            turn=turn,
            model_call_id=model_call_id,
            tools_count=len(tool_schemas),
            context_report=context_report,
        )
        emit_hook(
            hooks,
            "before_model_call",
            kind=kind,
            data={
                "turn": turn,
                "model_call_id": model_call_id,
                "tools_count": len(tool_schemas),
                "context_report": context_report,
            },
        )
        try:
            raw = call_model_with_rate_limit_retry(
                client,
                trace,
                messages,
                tool_schemas,
                turn=turn,
                model_call_id=model_call_id,
            )
        except RuntimeError as exc:
            trace.write("model_error", turn=turn, error=str(exc))
            trace.write("session_end", result=f"error: {exc}")
            emit_session_error(hooks, kind, exc, turn=turn)
            raise
        message = raw["choices"][0]["message"]
        trace.write(
            "model_call_finished",
            turn=turn,
            model_call_id=model_call_id,
            model_response=model_response_summary(raw),
            message=message,
        )
        emit_hook(
            hooks,
            "after_model_call",
            kind=kind,
            data={"turn": turn, "message": model_message_summary(message)},
        )
        context.record_assistant(message)
        calls = message.get("tool_calls") or []
        if not calls:
            result = message.get("content") or ""
            trace.write("session_end", result=result)
            emit_hook(
                hooks,
                "session_end",
                kind=kind,
                data={
                    "status": "done",
                    "turns": turn + 1,
                    "result_preview": result[:1000],
                },
            )
            return {"status": "done", "result": result, "turns": turn + 1}
        for call in calls:
            # Берём имя инструмента заранее: если arguments битый и parse_tool_args
            # упадёт, у нас останется осмысленное имя для observation, а не 'tool_call'.
            fn = call.get("function") if isinstance(call, dict) else {}
            call_name = (fn.get("name") if isinstance(fn, dict) else "") or "tool_call"
            try:
                name, args = parse_tool_args(call)
                decision = emit_hook(
                    hooks,
                    "before_tool_call",
                    kind=kind,
                    data={
                        "turn": turn,
                        "call_id": str(call.get("id") or ""),
                        "tool": name,
                        "args": args,
                    },
                )
                if decision.ok:
                    obs = tools_registry.call(name, args)
                else:
                    obs = fail(
                        name,
                        f"blocked by hook: {decision.block}",
                        hook_blocked=True,
                    )
            except json.JSONDecodeError as exc:
                # Модель прислала arguments не валидным JSON — почти всегда это
                # обрезанная генерация (repetition loop + лимит токенов): строка
                # остаётся незакрытой. Подсказываем модели, что именно сломалось и
                # как восстановиться, иначе на следующем ходу она уткнётся в ту же
                # ошибку, а harness отправит провайдеру битый tool_call.
                name, args = call_name, {}
                obs = fail(
                    name,
                    f"tool call arguments are not valid JSON: {exc.msg}",
                    repair_hint=(
                        "Your previous tool call was truncated (likely a token limit "
                        "or a repetition loop). Repeat the call with complete, valid "
                        "JSON arguments. For large files prefer write_file in smaller "
                        "chunks or apply_patch."
                    ),
                )
                trace.write(
                    "tool_call_malformed",
                    turn=turn,
                    call_id=str(call.get("id") or ""),
                    tool=name,
                    error=str(exc),
                )
            except Exception as exc:
                name, args = "tool_call", {}
                obs = fail(name, f"invalid tool call: {exc}")
            followup_messages = obs.pop("_followup_messages", [])
            subagent_stop = obs.pop("_subagent_stop", "")
            apply_hidden_observation_effects(context, trace, obs)
            _maybe_apply_patch_retry_hint(name, args, obs, failed_patches_by_path)
            trace.write("tool_observation", tool=name, args=args, observation=obs)
            emit_hook(
                hooks,
                "after_tool_call",
                kind=kind,
                data={
                    "turn": turn,
                    "tool": name,
                    "args": args,
                    "observation": obs,
                },
            )
            if stop_on_user_input and subagent_stop == "needs_user_input":
                trace.write(
                    "session_end",
                    result=f"needs_user_input: {obs.get('question', '')}",
                )
                emit_hook(
                    hooks,
                    "session_end",
                    kind=kind,
                    data={
                        "status": "needs_user_input",
                        "turns": turn + 1,
                        "result_preview": str(obs.get("question", ""))[:1000],
                    },
                )
                return {
                    "status": "needs_user_input",
                    "result": obs.get("question", ""),
                    "observation": obs,
                    "turns": turn + 1,
                }
            context.record_tool_result(
                call,
                obs,
                followup_messages,
                file_refs=_file_refs_from_call(name, args, obs),
            )
            if is_parent_user_input_request(obs):
                result = render_user_input_request(obs)
                trace.write(
                    "user_input_requested",
                    subagent=obs.get("subagent", ""),
                    question=obs.get("question", ""),
                    options=obs.get("options", []),
                    reason=obs.get("reason", ""),
                    subagent_trace_id=obs.get("subagent_trace_id", ""),
                    subagent_trace_path=obs.get("subagent_trace_path", ""),
                )
                trace.write("session_end", result=result)
                emit_hook(
                    hooks,
                    "session_end",
                    kind=kind,
                    data={
                        "status": "needs_user_input",
                        "turns": turn + 1,
                        "result_preview": result[:1000],
                    },
                )
                return {
                    "status": "needs_user_input",
                    "result": result,
                    "observation": obs,
                    "turns": turn + 1,
                }
    result = "Agent stopped: max_turns exceeded."
    trace.write("session_end", result=result)
    emit_hook(
        hooks,
        "session_end",
        kind=kind,
        data={"status": "max_turns", "turns": max_turns, "result_preview": result},
    )
    return {"status": "max_turns", "result": result, "turns": max_turns}


def emit_hook(
    hooks: HookManager | None,
    name: str,
    *,
    kind: str,
    data: dict[str, Any],
) -> HookDecision:
    """Вызываем hooks, если manager подключён к этому запуску."""

    if hooks is None:
        return HookDecision()
    return hooks.emit(name, kind=kind, data=data)


def emit_session_error(
    hooks: HookManager | None,
    kind: str,
    exc: Exception,
    *,
    turn: int | None = None,
) -> None:
    """Сообщаем пользовательским hooks об ошибке сессии."""

    emit_hook(
        hooks,
        "session_error",
        kind=kind,
        data={
            "turn": turn,
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    )


def model_message_summary(message: dict[str, Any]) -> dict[str, Any]:
    """Передаём hooks краткую сводку ответа модели без полного payload."""

    calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    tools = []
    for call in calls:
        fn = call.get("function") or {}
        tools.append(
            {
                "id": str(call.get("id") or ""),
                "name": str(fn.get("name") or ""),
            }
        )
    return {
        "content_preview": str(content)[:1000],
        "tool_calls_count": len(calls),
        "tools": tools,
    }


def model_response_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Сохраняем компактную телеметрию ответа модели для анализа trace."""

    usage = raw.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    # finish_reason показывал бы, почему модель остановилась ('stop', 'length',
    # 'tool_calls'). Нас особенно интересует 'length' — признак обрезанной
    # генерации, которая часто приводит к битому arguments. OpenRouter не всегда
    # прокидывает поле, поэтому None — нормальное значение, не ошибка.
    choices = raw.get("choices")
    finish_reason = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
    return {
        "id": str(raw.get("id") or ""),
        "model": str(raw.get("model") or ""),
        "provider": str(raw.get("provider") or ""),
        "prompt_tokens": _optional_int(usage.get("prompt_tokens")),
        "completion_tokens": _optional_int(usage.get("completion_tokens")),
        "total_tokens": _optional_int(usage.get("total_tokens")),
        "finish_reason": finish_reason,
    }


def _optional_int(value: Any) -> int | None:
    """Аккуратно приводим usage поля провайдера к int, если это возможно."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def apply_hidden_observation_effects(
    context: ContextManager,
    trace: Trace,
    observation: dict[str, Any],
) -> None:
    """Применяем служебные эффекты tool observation, не отправляя их модели."""

    fragments = observation.pop("_context_fragments", [])
    for fragment in fragments:
        context.add_fragment(fragment)
    skill_event = observation.pop("_skill_event", None)
    if skill_event:
        trace.write("skill_activated", **skill_event)


def _maybe_apply_patch_retry_hint(
    tool_name: str,
    args: dict[str, Any],
    observation: dict[str, Any],
    failed_patches_by_path: dict[str, int],
) -> None:
    """Следим за неудачными apply_patch и подсказываем модели прервать цикл.

    После PATCH_RETRY_HINT_THRESHOLD неудач по одному пути добавляем в observation
    поле retry_hint: модель видит, что её память о файле рассинхронизирована, и
    быстрее переходит к read_file или write_file целиком. Успешная правка путь
    сбрасывает счётчик. Мутирует observation in-place до записи в trace/историю.
    """

    if tool_name != "apply_patch":
        return
    patch = args.get("patch")
    if not isinstance(patch, str):
        return
    paths = paths_from_patch(patch)
    if not paths:
        return
    observation_ok = bool(observation.get("ok"))
    for path in paths:
        if observation_ok:
            # Модель исправилась — обнуляем счётчик неудач по этому пути.
            failed_patches_by_path.pop(path, None)
            continue
        count = failed_patches_by_path.get(path, 0) + 1
        failed_patches_by_path[path] = count
        if count >= PATCH_RETRY_HINT_THRESHOLD:
            observation["retry_hint"] = (
                f"apply_patch failed {count} times for {path}; the file likely "
                "differs from your memory. Use read_file for the current state "
                "or write_file for the full content."
            )


def is_parent_user_input_request(observation: dict[str, Any]) -> bool:
    """Понимаем, что делегация просит остановить `run` и спросить пользователя."""

    return (
        observation.get("tool") == "delegate_task"
        and observation.get("status") == "needs_user_input"
        and bool(str(observation.get("question") or "").strip())
    )


def render_user_input_request(observation: dict[str, Any]) -> str:
    """Печатаем вопрос субагента напрямую пользователю без ещё одного model call."""

    subagent = str(observation.get("subagent") or "subagent").strip()
    question = str(observation.get("question") or "").strip()
    reason = str(observation.get("reason") or "").strip()
    options = observation.get("options") or []
    lines = [f"{subagent} просит уточнение:", "", question]
    cleaned_options = [str(item).strip() for item in options if str(item).strip()]
    if cleaned_options:
        lines.extend(["", "Варианты:"])
        lines.extend(f"{index}. {item}" for index, item in enumerate(cleaned_options, 1))
    if reason:
        lines.extend(["", f"Причина: {reason}"])
    lines.extend(
        [
            "",
            "Ответьте на вопрос и повторите команду `run` с выбранным решением в задаче.",
        ]
    )
    return "\n".join(lines)


def safe_context_report(context: ContextManager) -> dict[str, Any]:
    """Пишем context error в trace, даже если повторная сборка отчёта тоже падает."""

    try:
        return context.report()
    except Exception as exc:
        return {"error": str(exc)}


def _content_hash(text: str | None) -> str | None:
    """Короткий хэш содержимого для файлового реестра; None для пустого ввода."""

    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _file_refs_from_call(
    name: str,
    args: dict[str, Any],
    observation: dict[str, Any],
) -> list[FileRef]:
    """Собираем файловые эффекты tool call для контекстного слоя.

    Рассматриваем только успешные read/write/patch: неудачные вызовы не меняют
    состояние файлов и не должны попадать в реестр. Для read берём хэш из того,
    что увидела модель в observation (clipped excerpt); для write — из args,
    где лежит полный записанный текст. apply_patch мультифайловый: пути достаём
    общим парсером patch-формата.
    """

    if not observation.get("ok"):
        return []
    if name == "read_file":
        path = args.get("path")
        if not isinstance(path, str):
            return []
        content = observation.get("content")
        return [FileRef(path=path, kind="read", content_hash=_content_hash(content))]
    if name == "write_file":
        path = args.get("path")
        if not isinstance(path, str):
            return []
        content = args.get("content")
        return [FileRef(path=path, kind="write", content_hash=_content_hash(content))]
    if name == "apply_patch":
        patch = args.get("patch")
        if not isinstance(patch, str):
            return []
        # У apply_patch нет финального содержимого в observation, поэтому хэш
        # неизвестен: для реестра важен сам факт правки пути.
        return [FileRef(path=p, kind="patch") for p in paths_from_patch(patch)]
    return []
