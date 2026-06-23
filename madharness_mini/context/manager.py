"""Собираем сообщения, которые будут отправлены модели."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from .budget import (
    TOKEN_ESTIMATE_BYTES_PER_TOKEN,
    clip_tool_content,
    clip_tool_messages,
    clip_text,
    dedup_tool_messages,
    digest_read_file,
    digest_write_args,
    estimate_request_tokens,
    estimate_tokens,
)
from .fragments import ContextFragment, ContextProvider, ContextState
from .history import FileRef, HistoryEntry
from .render import render_messages
from .summary import ReasoningSummarizer

# Полный assistant-текст полезен только до разумного предела: модель уже
# получила свои рассуждения в прошлом ходе, а следующий запрос платит за них снова.
ASSISTANT_CONTENT_LIMIT = 8000

# Если assistant сразу вызывает tool, его текст обычно служебный; сохраняем кратко.
ASSISTANT_TOOL_CONTENT_LIMIT = 2000

# Пределы возрастной эвикции (гипотеза B): старые assistant-рассуждения и
# tool-наблюдения усекаем до этих значений, оставляя «скелет» хода для модели.
SUMMARY_ASSISTANT_LIMIT = 500
SUMMARY_TOOL_LIMIT = 200

# Аргументы write_file/apply_patch старого хода сворачиваем в указатель, если они
# крупнее этого порога: тело файла (код) доминирует в стоимости assistant-ходов.
SUMMARY_TOOLCALL_LIMIT = 200

# Маркер заменяет вложенный блок только в диагностическом goal-anchor hash.
ATTACHED_DATA_MARKER = "[attached data tracked as a separate context unit]"
# Fenced-блоки часто несут код, логи или вставленные документы из запроса.
FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# Явные заголовки помогают отделить внешний материал от самой цели задачи.
ATTACHED_HEADING_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?"
    r"(?:attached data|attachment|external document|external text|document|log|"
    r"tool output|trace|web text|pasted text|вложенные данные|внешний документ|"
    r"документ|лог|вывод)\s*:?\s*$"
)

# Идентификатор transient-фрагмента с напоминанием о «грязных» файлах.
FILE_STATE_REMINDER_ID = "file-state:reminder"

# Идентификатор транзиентного фрагмента-скелета свёрнутой/выброшенной истории.
COMPACTED_HISTORY_ID = "history:compacted"
# Скелет не должен расти безгранично: показываем последние ходы и общий лимит.
COMPACTED_HISTORY_MAX_LINES = 20
# Чуть увеличили: помимо скелета ходов, теперь помещаем списки прочитанных и
# изменённых в свёрнутых ходах файлов (п.3) — модель сохраняет знание о них.
COMPACTED_HISTORY_CAP_CHARS = 2500
# Сколько путей показываем в каждой секции файлов свёрнутой истории.
COMPACTED_HISTORY_MAX_FILES = 15

# Идентификатор закреплённого фрагмента накопительной LLM-сводки старых ходов.
ROLLING_SUMMARY_ID = "summary:rolling"

# Маркер emergency-клиппинга инструкций: последний эшелон перед fatal помечает
# усечённый текст, чтобы модель видела — системные правила были пожертвованы ради
# продолжения сессии, и относилась к остатку осторожно.
EMERGENCY_CLIP_MARKER = "\n...[emergency clipped to fit context budget]"


@dataclass
class _FileState:
    """Запись о последнем чтении и правке одного пути.

    Хранит номера ходов последнего read и write/patch. Файл считается «грязным»,
    если правка случилась позже чтения (или файл не перечитывали вовсе). turn —
    это индекс элемента истории, который оставил событие.
    """

    last_read_turn: int | None = None
    last_write_turn: int | None = None


class ContextManager:
    """Хранит контекст одного ask/run и собирает сообщения для Chat Completions.

    Менеджер ничего не знает о Config, Policy, ModelClient и реальных handlers.
    Loop сообщает ему факты: стартовые фрагменты, ответ модели и результат
    инструмента. На выходе получается обычный список сообщений для API.
    """

    def __init__(
        self,
        user_task: str,
        *,
        max_tokens: int = 60000,
        keep_recent_turns: int = 3,
        summarize_after_turns: int = 0,
        providers: Iterable[ContextProvider] | None = None,
        summarizer: ReasoningSummarizer | None = None,
        summary_trigger_tokens: int = 0,
        contract_protection_turns: int = 12,
        contract_protection_writes: int = 3,
        reserve_tokens: int = 0,
    ):
        self.user_task = user_task
        self.max_tokens = max(int(max_tokens), 0)
        self.keep_recent_turns = max(int(keep_recent_turns), 0)
        # Headroom для проактивного усечения: дроп истории срабатывает, когда
        # оценка превышает (max_tokens - reserve_tokens), а не сам max_tokens.
        # Так между порогом дропа и жёстким пределом остаётся запас под рост
        # следующих ходов. 0 = выкл (поведение прежнее, порог совпадает с max).
        self.reserve_tokens = max(int(reserve_tokens), 0)
        # Граница возрастной эвикции: assistant-текст старше этого числа ходов
        # усекается, а его tool-наблюдения сворачиваются. 0 — выкл (поведение по
        # умолчанию), чтобы не менять существующие прогоны учебного харнесса.
        self.summarize_after_turns = max(int(summarize_after_turns), 0)
        # Fan-out защита контрактных чтений от эвикции: если после read_file за
        # contract_protection_turns ходов последовало ≥ contract_protection_writes
        # правок других путей, чтение сворачивать нельзя — модель использует его
        # как спецификацию для зависимых правок (cold gap из трассы flappy2).
        self.contract_protection_turns = max(int(contract_protection_turns), 0)
        self.contract_protection_writes = max(int(contract_protection_writes), 0)
        self.providers = list(providers or [])
        # Внешний суммаризатор рассуждений (DIP): менеджер его не создаёт, а
        # получает извне. Триггер по токеновому порогу истории; 0 — выключено,
        # чтобы по умолчанию не было платных вызовов и сохранялся детерминизм.
        self.summarizer = summarizer
        self.summary_trigger_tokens = max(int(summary_trigger_tokens), 0)
        # Накопительная LLM-сводка свёрнутых ходов и граница уже свёрнутого:
        # ходы с original_index < _summarized_upto заменяются фрагментом сводки.
        self._rolling_summary: str = ""
        self._summarized_upto: int = 0
        self._fragments: list[ContextFragment] = []
        self._history: list[HistoryEntry] = []
        # Реестр файлового состояния: путь -> последняя read/write правка.
        # Кормит напоминание о «грязных» файлах (гипотеза C) и не пишется в trace.
        self._file_state: dict[str, _FileState] = {}
        # История правок по ходам: turn -> множество правленных путей. Нужна для
        # fan-out защиты _is_protected_read (сколько разных путей правилось после
        # чтения). _file_state хранит только максимум, поэтому трассу правок ведём
        # отдельно. Не пишется в trace — производный индекс.
        self._writes_by_turn: dict[int, set[str]] = {}
        # Скелет свёрнутых/выброшенных ходов: original_index -> компактная строка.
        # Ключ по индексу делает накопление идемпотентным при повторных messages().
        self._dropped_summary: dict[int, str] = {}
        # Калибровка эвристики bytes/3 к реальному tokenizer провайдера: loop
        # сообщает prompt_tokens из ответа модели, мы сглаживаем отношение
        # «реальный/наш estimate» и применяем ко всем порогам бюджета. Стартуем с
        # 1.0 — до первого record_usage поведение идентично прежней эвристике.
        self._token_ratio = 1.0
        # Сырая оценка последнего отправленного запроса (без ratio): нужна как
        # база для расчёта ratio в record_usage. Сохряняется в конце messages().
        self._last_request_estimate = 0
        self._last_stats: dict[str, int | bool] | None = None
        self._last_report: dict[str, Any] | None = None

    def add_fragment(self, fragment: ContextFragment) -> None:
        """Добавляем или заменяем фрагмент по id."""

        self._fragments = [item for item in self._fragments if item.id != fragment.id]
        self._fragments.append(fragment)
        self._last_stats = None
        self._last_report = None

    def record_usage(self, prompt_tokens: int | None) -> None:
        """Калибруем эвристику оценки токенов по реальному usage провайдера.

        Loop вызывает метод после каждого model_call, передавая prompt_tokens из
        ответа модели. Берём отношение реального значения к нашей оценке того же
        запроса (_last_request_estimate) и сглаживаем _token_ratio: EMA с весом
        0.5. None/ноль/отсутствие базы — no-op (детерминированно без изменений).
        Ratio ограничен [0.3, 3.0], чтобы разовый выброс провайдера не исказил
        пороги бюджета. Инвалидация отчёта/статистики запускает пересчёт в下次
        messages().
        """

        if not prompt_tokens or not self._last_request_estimate:
            return
        try:
            measured = float(prompt_tokens) / float(self._last_request_estimate)
        except (TypeError, ValueError, ZeroDivisionError):
            return
        measured = max(0.3, min(3.0, measured))
        self._token_ratio = 0.5 * self._token_ratio + 0.5 * measured
        self._last_stats = None
        self._last_report = None

    def _over_budget(self, raw_estimate: int, *, include_reserve: bool = True) -> bool:
        """Скорректированная оценка токенов превышает бюджет?

        Применяем _token_ratio (калибровка bytes/3 к реальному tokenizer) к сырой
        оценке и сравниваем с порогом. При include_reserve=True (по умолчанию) —
        это проактивный порог (max_tokens - reserve_tokens), на котором работают
        усечение и дроп: между ним и жёстким пределом остаётся запас под рост
        следующих ходов. При include_reserve=False порог = сам max_tokens: так
        считается hard_limit_exceeded (fatal остаётся на настоящем пределе окна).
        """

        if not self.max_tokens:
            return False
        threshold = self.max_tokens
        if include_reserve and self.reserve_tokens > 0:
            threshold = max(self.max_tokens - self.reserve_tokens, 1)
        return int(raw_estimate * self._token_ratio) > threshold

    def record_assistant(self, message: dict[str, Any]) -> None:
        """Запоминаем ответ модели как следующий атомарный элемент истории."""

        stored = _sanitize_assistant_message(message)
        expected = {
            str(call.get("id"))
            for call in stored.get("tool_calls") or []
            if call.get("id")
        }
        kind = "tool_turn" if expected else "assistant"
        self._history.append(
            HistoryEntry(
                kind=kind,
                messages=[stored],
                expected_tool_call_ids=expected,
            )
        )
        self._last_stats = None
        self._last_report = None

    def record_tool_result(
        self,
        call: dict[str, Any],
        observation: dict[str, Any],
        followup_messages: Iterable[dict[str, Any]] = (),
        *,
        file_refs: Iterable[FileRef] = (),
    ) -> None:
        """Добавляем role=tool и отложенные follow-up сообщения.

        file_refs — опциональные файловые эффекты этого tool call (путь, тип,
        хэш). Loop передаёт их для read_file/write_file/apply_patch, чтобы слой
        контекста мог предупреждать о правках без свежего чтения. Старый вызов
        без file_refs остаётся полностью совместимым: реестр просто не растёт.
        """

        entry = self._last_tool_entry()
        call_id = str(call.get("id") or _tool_name(call, observation))
        entry.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(observation, ensure_ascii=False),
            }
        )
        entry.seen_tool_call_ids.add(call_id)
        entry.pending_followups.extend(copy.deepcopy(list(followup_messages)))
        refs = list(file_refs)
        if refs:
            # Индекс хода — позиция записи в истории до её возможного роста.
            turn = len(self._history) - 1
            for ref in refs:
                self._update_file_state(ref, turn)
                # Копим правки других путей по ходам — для fan-out защиты
                # _is_protected_read. read сюда не кладём: нужны только write/patch.
                if ref.kind in ("write", "patch"):
                    self._writes_by_turn.setdefault(turn, set()).add(ref.path)
            entry.file_refs = refs
        self._last_stats = None
        self._last_report = None

    def messages(self, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Возвращаем сообщения для модели с учётом бюджета контекста."""

        fragments = self._collect_fragments()
        entries = copy.deepcopy(self._history)
        entry_indexes = list(range(len(entries)))
        # Дедуп сворачивает избыточные tool-наблюдения (read_file, дублирующий
        # постоянный фрагмент; повторы внутри истории) до оценки бюджета.
        deduped_tool_messages = dedup_tool_messages(
            entries, fragments, is_protected_read=self._is_protected_read
        )
        summarized_old_entries = self._summarize_old_entries(entries, entry_indexes)
        # LLM-свёртка по токеновому порогу: обновляет накопительную сводку, а
        # _apply_summary_fold убирает уже свёрнутые ходы из рендера.
        self._maybe_summarize(entries, entry_indexes)
        self._apply_summary_fold(entries, entry_indexes)
        messages = render_messages(self.user_task, fragments, entries)
        initial_estimate = estimate_request_tokens(messages, tools)
        initial_tokens = initial_estimate["request_tokens_estimate"]
        truncated = False
        dropped_entries: list[dict[str, Any]] = []
        clipped_tool_messages: list[dict[str, Any]] = []
        clip_limit_chars = 0

        if self.max_tokens and self._over_budget(initial_tokens):
            clip_limit_chars = max(
                80,
                min(4000, self.max_tokens * TOKEN_ESTIMATE_BYTES_PER_TOKEN // 8),
            )
            clipped_tool_messages = clip_tool_messages(entries, clip_limit_chars)
            if clipped_tool_messages:
                truncated = True
                messages = render_messages(self.user_task, fragments, entries)

        current_estimate = estimate_request_tokens(messages, tools)
        if self._over_budget(current_estimate["request_tokens_estimate"]):
            dropped_entries = self._drop_old_entries_until_budget(
                fragments,
                entries,
                entry_indexes,
                tools,
            )
            messages = render_messages(self.user_task, fragments, entries)
            truncated = truncated or bool(dropped_entries)
            current_estimate = estimate_request_tokens(messages, tools)

        if self._over_budget(current_estimate["request_tokens_estimate"]):
            forced_dropped = self._drop_old_entries_until_budget(
                fragments,
                entries,
                entry_indexes,
                tools,
                keep_recent_turns=0,
                forced=True,
            )
            dropped_entries.extend(forced_dropped)
            messages = render_messages(self.user_task, fragments, entries)
            truncated = truncated or bool(forced_dropped)
            current_estimate = estimate_request_tokens(messages, tools)

        # Эшелон emergency (п.1 recovery): когда обычное усечение и forced-drop не
        # справились, жертвуем рабочими фрагментами и клипаем инструкции, чтобы
        # избежать fatal RuntimeError и дать сессии продолжиться.
        emergency_truncated = False
        emergency_dropped_fragments: list[str] = []
        if self._over_budget(
            current_estimate["request_tokens_estimate"], include_reserve=False
        ):
            success, emergency_dropped_fragments, fragments, messages = (
                self._emergency_truncate(fragments, entries, entry_indexes, tools)
            )
            if success:
                emergency_truncated = True
                truncated = True
                current_estimate = estimate_request_tokens(messages, tools)

        hard_limit_exceeded = self._over_budget(
            current_estimate["request_tokens_estimate"], include_reserve=False
        )
        context_packet = _context_packet_report(
            self.user_task,
            fragments,
            entries,
            entry_indexes,
            tools,
            current_estimate,
        )
        self._last_stats = {
            "context_tokens_estimate": current_estimate["request_tokens_estimate"],
            "messages_tokens_estimate": current_estimate["messages_tokens_estimate"],
            "tools_tokens_estimate": current_estimate["tools_tokens_estimate"],
            "fragments": len(fragments),
            "history_entries": len(self._history),
            "dropped_entries": len(dropped_entries),
            "truncated": truncated,
            "hard_limit_exceeded": hard_limit_exceeded,
            "token_ratio": self._token_ratio,
            "emergency_truncated": emergency_truncated,
        }
        self._last_report = {
            "max_tokens": self.max_tokens,
            "initial_request_tokens_estimate": initial_tokens,
            **current_estimate,
            "over_budget": self._over_budget(initial_tokens),
            "truncated": truncated,
            "hard_limit_exceeded": hard_limit_exceeded,
            "token_ratio": self._token_ratio,
            "reserve_tokens": self.reserve_tokens,
            "emergency_truncated": emergency_truncated,
            "emergency_dropped_fragments": emergency_dropped_fragments,
            "fragments": [_fragment_report(fragment) for fragment in fragments],
            "context_packet": context_packet,
            "history": {
                "total_entries": len(self._history),
                "rendered_entries": len(entries),
                "keep_recent_turns": self.keep_recent_turns,
                "summarize_after_turns": self.summarize_after_turns,
                "clip_limit_chars": clip_limit_chars,
                "clipped_tool_messages": clipped_tool_messages,
                "deduped_tool_messages": deduped_tool_messages,
                "summarized_old_entries": summarized_old_entries,
                "dropped_entries": dropped_entries,
                "included_entries": [
                    _history_entry_report(entry, index)
                    for index, entry in zip(entry_indexes, entries)
                ],
            },
        }
        # Сырая оценка (без ratio) — база для следующего record_usage. Берём её
        # только при успехе: при RuntimeError запрос не уходит, калибровка по нему
        # некорректна.
        self._last_request_estimate = current_estimate["request_tokens_estimate"]
        if hard_limit_exceeded:
            raise RuntimeError(
                "context budget exceeded after truncation: "
                f"{current_estimate['request_tokens_estimate']}/{self.max_tokens} "
                "estimated tokens"
            )
        return messages

    def stats(self) -> dict[str, int | bool]:
        """Короткая диагностика последней сборки контекста."""

        if self._last_stats is None:
            self.messages()
        return dict(self._last_stats or {})

    def report(self) -> dict[str, Any]:
        """Подробно описываем последнюю сборку контекста без текстов сообщений.

        Отчёт нужен для трасс и отладки бюджета: он показывает размеры,
        фрагменты, оставшуюся историю и действия обрезки, но не дублирует
        содержимое prompt/tool output.
        """

        if self._last_report is None:
            self.messages()
        return copy.deepcopy(self._last_report or {})

    def _last_tool_entry(self) -> HistoryEntry:
        """Находим последний tool turn или создаём защитный entry для сбоя."""

        if self._history and self._history[-1].kind == "tool_turn":
            return self._history[-1]
        entry = HistoryEntry(kind="tool_turn")
        self._history.append(entry)
        return entry

    def _update_file_state(self, ref: FileRef, turn: int) -> None:
        """Обновляем реестр файлового состояния по одной ссылке от tool call.

        read обновляет last_read_turn, write/patch — last_write_turn. Берём
        максимум по turn, чтобы несколько событий по одному файлу в одном ходе
        не затирали друг друга и сохраняли самую свежую правку.
        """

        state = self._file_state.setdefault(ref.path, _FileState())
        if ref.kind == "read":
            state.last_read_turn = (
                turn if state.last_read_turn is None else max(state.last_read_turn, turn)
            )
        else:  # write или patch
            state.last_write_turn = (
                turn
                if state.last_write_turn is None
                else max(state.last_write_turn, turn)
            )

    def _dirty_files(self) -> list[tuple[str, int]]:
        """Пути, изменённые после последнего чтения (или не прочитанные вовсе).

        Возвращаем пары (путь, turn последней правки), отсортированные по убыванию
        turn: самые свежие «грязные» файлы оказываются первыми в напоминании.
        """

        dirty: list[tuple[str, int]] = []
        for path, state in self._file_state.items():
            if state.last_write_turn is None:
                continue
            if state.last_read_turn is None or state.last_read_turn < state.last_write_turn:
                dirty.append((path, state.last_write_turn))
        dirty.sort(key=lambda item: item[1], reverse=True)
        return dirty

    def _dirty_files_by_category(
        self,
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Делим грязные файлы на «ни разу не прочитан» и «устарел после правки».

        never_read: была правка, но чтения по пути не было вовсе. stale_after_write:
        чтение было, но раньше последней правки. Критерий грязного файла совпадает
        с `_dirty_files`; внутри каждой группы сортируем по убыванию turn правки.
        """

        never_read: list[tuple[str, int]] = []
        stale_after_write: list[tuple[str, int]] = []
        for path, state in self._file_state.items():
            if state.last_write_turn is None:
                continue
            if state.last_read_turn is None:
                never_read.append((path, state.last_write_turn))
            elif state.last_read_turn < state.last_write_turn:
                stale_after_write.append((path, state.last_write_turn))
        never_read.sort(key=lambda item: item[1], reverse=True)
        stale_after_write.sort(key=lambda item: item[1], reverse=True)
        return never_read, stale_after_write

    def _entry_skeleton(self, entry: HistoryEntry) -> str:
        """Компактный «скелет» одного хода для свёрнутой истории.

        Строка вида `{kind}; tools=[{name} {path?} {ok|fail}, ...]`. Имена берём
        из assistant.tool_calls (function.name), путь — из file_refs, признак
        ok|fail — из JSON role=tool наблюдения (поле "ok"). Любая из частей может
        отсутствовать: тогда подставляем пустую часть, не роняя сборку.
        """

        names: list[str] = []
        for message in entry.messages:
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if isinstance(name, str) and name:
                    names.append(name)
        oks: list[bool | None] = []
        for message in entry.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            ok_value: bool | None = None
            if isinstance(content, str):
                try:
                    payload = json.loads(content)
                except (ValueError, TypeError):
                    payload = None
                if isinstance(payload, dict) and "ok" in payload:
                    ok_value = bool(payload.get("ok"))
            oks.append(ok_value)
        refs = list(entry.file_refs)
        parts: list[str] = []
        count = max(len(names), len(oks), len(refs))
        for index in range(count):
            name = names[index] if index < len(names) else ""
            path = refs[index].path if index < len(refs) else ""
            ok_value = oks[index] if index < len(oks) else None
            status = "" if ok_value is None else ("ok" if ok_value else "fail")
            tool_part = " ".join(piece for piece in (name, path, status) if piece)
            if tool_part:
                parts.append(tool_part)
        return f"{entry.kind}; tools=[{', '.join(parts)}]"

    def _compacted_history_fragment(self) -> ContextFragment | None:
        """Транзиентный фрагмент-скелет уже свёрнутых/выброшенных ходов.

        Возвращаем None, пока ничего не выброшено. Иначе показываем последние
        COMPACTED_HISTORY_MAX_LINES строк (по возрастанию индекса хода) и
        ограничиваем общий размер COMPACTED_HISTORY_CAP_CHARS через clip_text.
        Фрагмент отдельный, а не замена записи: это исключает рекурсию бюджета и
        двойной учёт, а ключ-индекс гарантирует идемпотентность накопления.

        Помимо скелета ходов, показываем списки прочитанных и изменённых в
        свёрнутом диапазоне файлов (п.3): модель сохраняет знание, с какими путями
        работала, даже когда детали хода ушли в skeleton. Источник — _file_state.
        """

        if not self._dropped_summary:
            return None
        ordered = sorted(self._dropped_summary.items())[-COMPACTED_HISTORY_MAX_LINES:]
        lines = ["# Свёрнутые ходы"]
        for index, skeleton in ordered:
            lines.append(f"turn {index}: {skeleton}")
        # Граница свёрнутого диапазона — самый большой выброшенный индекс. Чтения
        # и правки с меньшим turn точно ушли из активной истории и заслуживают
        # упоминания; более свежие — ещё в окне, их не дублируем.
        compacted_boundary = max(self._dropped_summary)
        read_files, modified_files = self._compacted_file_lists(compacted_boundary)
        if read_files:
            lines.append("")
            lines.append("## Прочитанные ранее файлы")
            for path in read_files:
                lines.append(f"- {path}")
        if modified_files:
            lines.append("")
            lines.append("## Изменённые ранее файлы")
            for path in modified_files:
                lines.append(f"- {path}")
        text = clip_text("\n".join(lines), COMPACTED_HISTORY_CAP_CHARS)
        return ContextFragment(
            id=COMPACTED_HISTORY_ID,
            source="madharness-mini compacted history",
            text=text,
            priority=25,
            placement="system",
            transient=True,
            authority_level="harness",
            context_layer="working",
            evictability="preferred",
            stability="turn",
            applicability="current_task",
        )

    def _compacted_file_lists(
        self, compacted_boundary: int
    ) -> tuple[list[str], list[str]]:
        """Списки путей, задействованных в свёрнутом диапазоне (turn <= границы).

        read_files — прочитанные, но не правленные после чтения (только чтение в
        свёрнутом диапазоне). modified_files — правленные (write/patch). Путь
        может попасть в обе группы, если его читали и правили в разных ходах: это
        корректно, обе операции важны для модели. Ограничиваем каждую секцию
        COMPACTED_HISTORY_MAX_FILES путями, сортируем по последнему turn операции.
        """

        read_files: list[tuple[str, int]] = []
        modified_files: list[tuple[str, int]] = []
        for path, state in self._file_state.items():
            read_in_range = (
                state.last_read_turn is not None
                and state.last_read_turn <= compacted_boundary
            )
            write_in_range = (
                state.last_write_turn is not None
                and state.last_write_turn <= compacted_boundary
            )
            if read_in_range:
                read_files.append((path, state.last_read_turn or 0))
            if write_in_range:
                modified_files.append((path, state.last_write_turn or 0))
        read_files.sort(key=lambda item: item[1], reverse=True)
        modified_files.sort(key=lambda item: item[1], reverse=True)
        return (
            [path for path, _ in read_files[:COMPACTED_HISTORY_MAX_FILES]],
            [path for path, _ in modified_files[:COMPACTED_HISTORY_MAX_FILES]],
        )

    def _is_protected_read(self, path: str | None, read_turn: int) -> bool:
        """Защищено ли read_file-наблюдение от любой эвикции контекста.

        Два случая защиты:

        1. Тот же путь правился в этом же ходе или позже. Без такой защиты
        эвикция заменяет чтение дайджестом или роняет его целиком, модель
        генерирует патч по устаревшему воспоминанию, а harness применяет его к
        актуальному файлу — цикл неудачных apply_patch (см. apply_patch_storm).

        2. Fan-out контрактного чтения: после чтения этого пути было
        ≥ contract_protection_writes правок других путей за contract_protection_turns
        ходов. Модель читает спецификацию (контракт, интерфейс, ТЗ) и потом правит
        много зависимых файлов — сворачивание чтения оставляет её без актуального
        источника правды (cold gap из трассы flappy2: LeaderboardPorts.js → 8 правок
        соседних файлов). Защищаем только чтение, пока fan-out активен; если правки
        закончились раньше порога или вышли за окно — чтение сворачивается как обычно.

        Единый предикат переиспользуется в возрастной компактизации, дедупе
        tool-наблюдений и дропе истории по бюджету.
        """

        if not path:
            return False
        state = self._file_state.get(path)
        if state is not None and state.last_write_turn is not None:
            if state.last_write_turn >= read_turn:
                return True
        # Fan-out: считаем различные пути (кроме читаемого), правленные после
        # чтения в пределах окна. O(K) по числу ходов в окне — пренебрежимо для
        # учебного харнесса (≤300 ходов, окно по умолчанию 12).
        if self.contract_protection_writes <= 0 or self.contract_protection_turns <= 0:
            return False
        window_end = min(read_turn + 1 + self.contract_protection_turns, len(self._history))
        distinct_other: set[str] = set()
        for turn in range(read_turn + 1, window_end):
            for written_path in self._writes_by_turn.get(turn, ()):
                if written_path != path:
                    distinct_other.add(written_path)
        return len(distinct_other) >= self.contract_protection_writes

    def _entry_has_protected_read(self, entry: HistoryEntry, read_turn: int) -> bool:
        """Содержит ли ход защищённое read_file-наблюдение (для дропа по бюджету).

        Ход защищён от удаления в нефорсированном проходе, если читал файл,
        который позже правился: иначе мы выбросим актуальное содержимое и вернём
        модель к слепой правке (cold_gap).
        """

        return any(
            self._is_protected_read(ref.path, read_turn)
            for ref in entry.file_refs
            if ref.kind == "read"
        )

    def _file_state_reminder(self) -> ContextFragment | None:
        """Собираем transient-напоминание о файлах, которые правились без read.

        Различаем две категории: устаревшие после правки (читали раньше правки) и
        ни разу не прочитанные (правка без единого чтения). Для вторых указываем
        явно вызвать read_file перед правкой — иначе модель правит вслепую.
        """

        never_read, stale_after_write = self._dirty_files_by_category()
        if not never_read and not stale_after_write:
            return None
        lines = ["# Напоминание о файловом состоянии"]
        if stale_after_write:
            lines.append(
                "Эти файлы изменены после последнего чтения. Перед правкой "
                "убедитесь, что текущее содержимое известно, иначе вызовите read_file:"
            )
            for path, turn in stale_after_write:
                lines.append(
                    f"- {path} (изменён на ходу {turn}, не перечитан после правки)"
                )
        if never_read:
            lines.append(
                "Эти файлы записаны без единого чтения — текущее содержимое "
                "неизвестно:"
            )
            for path, turn in never_read:
                lines.append(
                    f"- {path} (записан на ходу {turn}, ни разу не прочитан — "
                    "вызовите read_file перед правкой)"
                )
        return ContextFragment(
            id=FILE_STATE_REMINDER_ID,
            source="madharness-mini file-state reminder",
            text="\n".join(lines),
            priority=20,
            placement="system",
            transient=True,
            authority_level="harness",
            context_layer="evidence",
            evictability="normal",
            stability="turn",
            applicability="current_task",
        )

    def _collect_fragments(self) -> list[ContextFragment]:
        """Собираем закреплённые и provider-фрагменты в стабильном порядке."""

        state = ContextState(
            user_task=self.user_task,
            fragments_count=len(self._fragments),
            history_entries=len(self._history),
            max_tokens=self.max_tokens,
            keep_recent_turns=self.keep_recent_turns,
        )
        fragments = list(self._fragments)
        for provider in self.providers:
            fragments.extend(provider.collect(state))
        reminder = self._file_state_reminder()
        if reminder is not None:
            fragments.append(reminder)
        compacted = self._compacted_history_fragment()
        if compacted is not None:
            fragments.append(compacted)
        rolling = self._rolling_summary_fragment()
        if rolling is not None:
            fragments.append(rolling)
        return sorted(
            fragments,
            key=lambda item: (item.placement, item.priority, item.id),
        )

    def _summarize_old_entries(
        self,
        entries: list[HistoryEntry],
        entry_indexes: list[int],
    ) -> list[dict[str, Any]]:
        """Сворачиваем старые entries по возрасту, не трогая свежие.

        Работает только когда задан summarize_after_turns > 0. Защищаем окно из
        keep_recent_turns и summarize_after_turns записей, а всё, что старше,
        усекаем: assistant-текст — до SUMMARY_ASSISTANT_LIMIT, role=tool — через
        digest_read_file для чтений файлов (указатель вместо обрезка) и
        clip_tool_content для прочего вывода. Возвращает описания свёрнутых
        записей для отчёта.
        """

        if self.summarize_after_turns <= 0:
            return []
        protected_count = self.keep_recent_turns + self.summarize_after_turns
        protected_start = max(len(entries) - protected_count, 0)
        summarized: list[dict[str, Any]] = []
        for position in range(protected_start):
            entry = entries[position]
            original_index = entry_indexes[position]
            read_paths = {
                ref.path for ref in entry.file_refs if ref.kind == "read"
            }
            changed = False
            for message in entry.messages:
                role = message.get("role")
                content = message.get("content")
                if role == "assistant":
                    if (
                        isinstance(content, str)
                        and len(content) > SUMMARY_ASSISTANT_LIMIT
                    ):
                        message["content"] = clip_text(content, SUMMARY_ASSISTANT_LIMIT)
                        changed = True
                    # Сворачиваем тяжёлые аргументы write_file/apply_patch: тело
                    # файла переотправляется каждый ход и доминирует в стоимости
                    # старых assistant-ходов. Полный текст лежит на диске —
                    # дайджест подсказывает перечитать его при необходимости.
                    if _digest_old_write_tool_calls(message):
                        changed = True
                elif role == "tool" and isinstance(content, str):
                    if len(content) <= SUMMARY_TOOL_LIMIT:
                        continue
                    # read_file сворачиваем в указатель: модель сохраняет знание
                    # о прочитанном, а не теряет его в обрезке середины текста.
                    tool_name, payload_path = _tool_kind_and_path(content)
                    path = payload_path or (
                        next(iter(read_paths), None) if tool_name == "read_file" else None
                    )
                    if tool_name == "read_file":
                        # Защита от рассинхронизации: если путь позже правился
                        # (write/patch в этом же или более свежем ходе), сворачивать
                        # чтение нельзя — модель будет генерировать патч по старому
                        # содержимому и получать "expected 1 hunk match, found 0".
                        # Оставляем полное наблюдение, оплачивая это токенами.
                        if not self._is_protected_read(path, original_index):
                            message["content"] = digest_read_file(content, path)
                            changed = True
                    else:
                        message["content"] = clip_tool_content(content, SUMMARY_TOOL_LIMIT)
                        changed = True
            if changed:
                summarized.append({"index": original_index, "kind": entry.kind})
        return summarized

    def _maybe_summarize(
        self,
        entries: list[HistoryEntry],
        entry_indexes: list[int],
    ) -> None:
        """Сворачиваем старые ходы LLM-сводкой при превышении токенового порога.

        Реализует FL3: работает только при заданном суммаризаторе и положительном
        пороге. Свёртке подлежит префикс истории за вычетом keep_recent_turns; из
        него исключаем ходы с защищённым чтением (FL1), чтобы не потерять актуальное
        состояние файлов. Любое исключение суммаризатора или пустой результат —
        детерминированный fallback: состояние не меняется. Само удаление свёрнутых
        ходов из рендера делает _apply_summary_fold.
        """

        if self.summarizer is None or self.summary_trigger_tokens <= 0:
            return
        rendered: list[dict[str, Any]] = []
        for entry in entries:
            rendered.extend(entry.rendered_messages())
        if estimate_tokens(rendered) <= self.summary_trigger_tokens:
            return
        foldable_limit = max(len(entries) - self.keep_recent_turns, 0)
        foldable: list[tuple[HistoryEntry, int]] = []
        for position in range(foldable_limit):
            entry = entries[position]
            index = entry_indexes[position]
            # Защищённое чтение не сворачиваем: его полный текст ещё нужен модели.
            if self._entry_has_protected_read(entry, index):
                continue
            foldable.append((entry, index))
        if not foldable:
            return
        try:
            new_summary = self.summarizer.summarize(
                [entry for entry, _ in foldable], self._rolling_summary
            )
        except Exception:
            # Fallback: при сбое суммаризатора ничего не меняем (A7).
            return
        if not new_summary:
            return
        self._rolling_summary = new_summary
        self._summarized_upto = max(index for _, index in foldable) + 1

    def _apply_summary_fold(
        self,
        entries: list[HistoryEntry],
        entry_indexes: list[int],
    ) -> None:
        """Убираем из рендера ходы, уже свёрнутые в накопительную сводку.

        Ход исключается, если его original_index < _summarized_upto. Защищённые
        чтения сохраняем, даже если попали в этот диапазон: их актуальное
        содержимое нельзя терять (FL1).
        """

        if self._summarized_upto <= 0:
            return
        kept = [
            (entry, index)
            for entry, index in zip(entries, entry_indexes)
            if index >= self._summarized_upto
            or self._entry_has_protected_read(entry, index)
        ]
        entries[:] = [entry for entry, _ in kept]
        entry_indexes[:] = [index for _, index in kept]

    def _rolling_summary_fragment(self) -> ContextFragment | None:
        """Закреплённый фрагмент с накопительной LLM-сводкой старых ходов."""

        if not self._rolling_summary:
            return None
        return ContextFragment(
            id=ROLLING_SUMMARY_ID,
            source="madharness-mini rolling summary",
            text="# Сводка предыдущих ходов\n\n" + self._rolling_summary,
            priority=15,
            placement="system",
            transient=False,
            authority_level="harness",
            context_layer="working",
            evictability="only_after_validation",
            stability="session",
            applicability="current_task",
        )

    def _emergency_truncate(
        self,
        fragments: list[ContextFragment],
        entries: list[HistoryEntry],
        entry_indexes: list[int],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[bool, list[str], list[ContextFragment], list[dict[str, Any]]]:
        """Последний эшелон перед fatal RuntimeError (п.1 recovery).

        Жертвуем рабочими фрагментами и клипаем инструкции, чтобы вписаться в
        max_tokens, когда обычное усечение и forced-drop не справились. Работает
        по существующей evictability-классификации фрагментов, без хардкода id:

        1. Убираем фрагменты с evictability in ('normal','preferred') — это карта
           проекта, напоминание о файлах, скелет свёрнутой истории. Системный
           промпт, project-instructions и rolling summary (never /
           only_after_validation / goal_update_only) остаются.
        2. Если не вписались — клипаем project-instructions до половины длины.
        3. Если не вписались — клипаем системный промпт аналогично.

        Возвращает (успех, id затронутых фрагментов, итоговые фрагменты, messages).
        При неудаче (даже system+task не лезут) — (False, ..., фрагменты, messages),
        что вызывающая сторона трактует как fatal RuntimeError. Сравнение везде по
        жёсткому max_tokens (include_reserve=False): reserve — запас для
        проактивного дропа, emergency работает на самом пределе окна.
        """

        dropped_ids: list[str] = []
        # 1. Убираем эвиктируемые рабочие фрагменты.
        emergency_fragments = [
            fragment
            for fragment in fragments
            if fragment.evictability not in ("normal", "preferred")
        ]
        removed = [
            fragment.id
            for fragment in fragments
            if fragment.evictability in ("normal", "preferred")
        ]
        dropped_ids.extend(removed)
        messages = render_messages(self.user_task, emergency_fragments, entries)
        estimate = estimate_request_tokens(messages, tools)["request_tokens_estimate"]
        if not self._over_budget(estimate, include_reserve=False):
            return True, dropped_ids, emergency_fragments, messages

        # 2. Клипаем project-instructions (only_after_validation) до половины.
        clipped_project = self._emergency_clip_stage(
            emergency_fragments, "only_after_validation"
        )
        if clipped_project is not None:
            dropped_ids.extend(
                fragment.id
                for fragment in clipped_project
                if EMERGENCY_CLIP_MARKER in fragment.text
            )
            emergency_fragments = clipped_project
            messages = render_messages(self.user_task, emergency_fragments, entries)
            estimate = estimate_request_tokens(messages, tools)[
                "request_tokens_estimate"
            ]
            if not self._over_budget(estimate, include_reserve=False):
                return True, dropped_ids, emergency_fragments, messages

        # 3. Клипаем системный промпт (never) до половины.
        clipped_system = self._emergency_clip_stage(emergency_fragments, "never")
        if clipped_system is not None:
            dropped_ids.extend(
                fragment.id
                for fragment in clipped_system
                if EMERGENCY_CLIP_MARKER in fragment.text
                and fragment.id not in dropped_ids
            )
            emergency_fragments = clipped_system
            messages = render_messages(self.user_task, emergency_fragments, entries)
            estimate = estimate_request_tokens(messages, tools)[
                "request_tokens_estimate"
            ]
            if not self._over_budget(estimate, include_reserve=False):
                return True, dropped_ids, emergency_fragments, messages

        return False, dropped_ids, emergency_fragments, messages

    def _emergency_clip_stage(
        self, fragments: list[ContextFragment], evictability: str
    ) -> list[ContextFragment] | None:
        """Клипаем фрагменты указанной evictability-стадии до половины длины.

        Возвращает новый список с клипнутым text (маркер emergency добавлен) или
        None, если ни один фрагмент этой стадии не изменился. Маркер показывает
        модели, что правила пожертвованы частью длины ради продолжения сессии.
        """

        result = list(fragments)
        changed = False
        for index, fragment in enumerate(result):
            if fragment.evictability != evictability:
                continue
            if not fragment.text.strip():
                continue
            limit = max(len(fragment.text) // 2, 1)
            if len(fragment.text) <= limit:
                continue
            clipped_text = clip_text(fragment.text, limit) + EMERGENCY_CLIP_MARKER
            result[index] = replace(fragment, text=clipped_text)
            changed = True
        return result if changed else None

    def _drop_old_entries_until_budget(
        self,
        fragments: list[ContextFragment],
        entries: list[HistoryEntry],
        entry_indexes: list[int],
        tools: list[dict[str, Any]] | None,
        keep_recent_turns: int | None = None,
        forced: bool = False,
    ) -> list[dict[str, Any]]:
        """Удаляем старые неприкреплённые элементы, сохраняя недавнюю историю."""

        dropped: list[dict[str, Any]] = []
        messages = render_messages(self.user_task, fragments, entries)
        keep_recent = (
            self.keep_recent_turns if keep_recent_turns is None else keep_recent_turns
        )
        protected_start = max(len(entries) - keep_recent, 0)
        while (
            entries
            and self._over_budget(
                estimate_request_tokens(messages, tools)["request_tokens_estimate"]
            )
        ):
            removable = next(
                (
                    index
                    for index in range(protected_start)
                    if entries[index]
                    and (forced or not self._entry_has_protected_read(
                        entries[index], entry_indexes[index]
                    ))
                ),
                None,
            )
            if removable is None:
                break
            report = _history_entry_report(entries[removable], entry_indexes[removable])
            if forced:
                report["forced"] = True
            dropped.append(report)
            # Копим скелет выброшенного хода по его исходному индексу: модель
            # увидит, что было свёрнуто, а ключ-индекс исключает двойной учёт.
            self._dropped_summary[entry_indexes[removable]] = self._entry_skeleton(
                entries[removable]
            )
            del entries[removable]
            del entry_indexes[removable]
            protected_start = max(len(entries) - keep_recent, 0)
            messages = render_messages(self.user_task, fragments, entries)
        return dropped


def _digest_old_write_tool_calls(message: dict[str, Any]) -> bool:
    """Сворачиваем аргументы write_file/apply_patch у старого assistant-хода.

    Заменяем тело файла (content/patch) дайджестом-указателем, оставляя валидный
    JSON: иначе провайдер отвергнет следующий запрос. Возвращаем True, если хотя
    бы один аргумент действительно свернулся — это помечает ход как изменённый
    для отчёта возрастной свёртки.
    """

    changed = False
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if name not in ("write_file", "apply_patch"):
            continue
        if not isinstance(arguments, str) or len(arguments) <= SUMMARY_TOOLCALL_LIMIT:
            continue
        digested = digest_write_args(arguments, name)
        if digested != arguments:
            function["arguments"] = digested
            changed = True
    return changed


def _sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Сохраняем в историю только поля, которые нужны следующему Chat request."""

    tool_calls = _sanitize_tool_calls(message.get("tool_calls") or [])
    content = message.get("content")
    limit = ASSISTANT_TOOL_CONTENT_LIMIT if tool_calls else ASSISTANT_CONTENT_LIMIT
    stored: dict[str, Any] = {"role": "assistant"}
    if isinstance(content, str):
        stored["content"] = clip_text(content, limit)
    elif content is None:
        stored["content"] = None if tool_calls else ""
    else:
        stored["content"] = copy.deepcopy(content)
    if tool_calls:
        stored["tool_calls"] = tool_calls
    return stored


def _sanitize_tool_calls(calls: list[Any]) -> list[dict[str, Any]]:
    """Оставляем у tool call только OpenAI-compatible id, type и function."""

    sanitized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        # Гарантируем, что в историю не попадёт обрезанный/битый JSON: если модель
        # упёрлась в лимит токенов внутри строкового литерала, arguments останется
        # незакрытым, и провайдер отвергнет следующий запрос с HTTP 400. Невалидную
        # строку заменяем на '{}' и сохраняем обрезок для диагностики в trace.
        sanitized_call: dict[str, Any] = {
            "id": str(call.get("id") or name),
            "type": str(call.get("type") or "function"),
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
        try:
            json.loads(arguments)
        except (ValueError, TypeError):
            sanitized_call["function"]["arguments"] = "{}"
            sanitized_call["_malformed_arguments"] = arguments[:500]
        sanitized.append(sanitized_call)
    return sanitized


def _fragment_report(fragment: ContextFragment) -> dict[str, Any]:
    """Описываем фрагмент контекста без самого текста."""

    return {
        "id": fragment.id,
        "source": fragment.source,
        "placement": fragment.placement,
        "priority": fragment.priority,
        "chars": len(fragment.text),
        "transient": fragment.transient,
        "empty": not bool(fragment.text.strip()),
        "authority_level": fragment.authority_level,
        "context_layer": fragment.context_layer,
        "evictability": fragment.evictability,
        "stability": fragment.stability,
        "applicability": fragment.applicability,
        "normative_role": fragment.normative_role,
        "goal_role": fragment.goal_role,
    }


def _history_entry_report(entry: HistoryEntry, index: int) -> dict[str, Any]:
    """Описываем элемент истории так, чтобы трасса не раздувалась контентом."""

    rendered = entry.rendered_messages()
    return {
        "index": index,
        "kind": entry.kind,
        "messages": len(entry.messages),
        "rendered_messages": len(rendered),
        "tokens_estimate": estimate_tokens(rendered),
        "roles": [str(message.get("role") or "") for message in rendered],
        "tool_call_ids": sorted(entry.expected_tool_call_ids | entry.seen_tool_call_ids),
        "pending_followups": len(entry.pending_followups),
        # Файловые эффекты без хэшей содержимого: для диагностики дедупа и
        # напоминания о состоянии достаточно путей и типов воздействия.
        "file_refs": [{"path": ref.path, "kind": ref.kind} for ref in entry.file_refs],
    }


def _tool_kind_and_path(content: str) -> tuple[str, str | None]:
    """Достаём имя инструмента и путь из JSON-сериализованного tool-наблюдения.

    Нужно, чтобы возрастная эвикция различала read_file (сворачиваем в указатель)
    и прочие tool outputs (обрезаем). Путь может отсутствовать в observation —
    тогда вызывающая сторона подставит его из file_refs.
    """

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "", None
    if not isinstance(payload, dict):
        return "", None
    tool_name = str(payload.get("tool") or "")
    path_value = str(payload.get("path") or "") or None
    return tool_name, path_value


def _tool_observation_meta(content: str) -> dict[str, str]:
    """Метаданные tool-наблюдения для context_packet: имя инструмента и путь.

    Heat-анализатор читает metadata.tool_name и metadata.path, чтобы отличать
    read_file от прочих tool_output и сверять свежесть чтения перед правкой.
    Без этих полей все tool-наблюдения выглядят безымянными — анализатор ставит
    ложный cold gap. Безопасный пустой возврат, если JSON не разобрался или полей
    нет: метаданные просто не дополняются, поведение остаётся прежним.
    """

    tool_name, path_value = _tool_kind_and_path(content)
    meta: dict[str, str] = {}
    if tool_name:
        meta["tool_name"] = tool_name
    if path_value:
        meta["path"] = path_value
    return meta


def _context_packet_report(
    user_task: str,
    fragments: list[ContextFragment],
    entries: list[HistoryEntry],
    entry_indexes: list[int],
    tools: list[dict[str, Any]] | None,
    estimate: dict[str, int],
) -> dict[str, Any]:
    """Пишем индекс prompt-сборки для будущей тепловой карты без текста prompt."""

    units: list[dict[str, Any]] = []
    cursor = 0

    for fragment in fragments:
        cursor = _append_context_unit(
            units,
            cursor,
            unit_id=f"fragment:{fragment.id}",
            source_type=_fragment_source_type(fragment),
            source_name=fragment.source,
            source_ref=fragment.id,
            payload=fragment.text,
            included_because=f"{fragment.placement}_fragment",
            metadata={
                "placement": fragment.placement,
                "priority": fragment.priority,
                "transient": fragment.transient,
                "chars": len(fragment.text),
            },
            classification=_classification_from_fragment(fragment),
            confidence=0.95,
        )

    for unit in _user_task_units(user_task):
        cursor = _append_context_unit(units, cursor, **unit)

    for entry, original_index in zip(entries, entry_indexes):
        for message_index, message in enumerate(entry.rendered_messages()):
            source_type = _message_source_type(message)
            role = str(message.get("role") or "")
            metadata = {
                "history_index": original_index,
                "history_kind": entry.kind,
                "role": role,
                "tool_call_id": str(message.get("tool_call_id") or ""),
                "tool_call_ids": sorted(
                    entry.expected_tool_call_ids | entry.seen_tool_call_ids
                ),
            }
            # Для tool-наблюдений поднимаем имя инструмента и путь из JSON-контента:
            # иначе heat-анализатор видит все чтения/выводы как безымянный
            # "tool_output" и не может отличить read_file, чтобы проверить свежесть
            # чтения перед правкой (ложный cold gap). Поля работают и после
            # возрастной эвикции — digest_read_file сохраняет tool и path.
            if role == "tool":
                content = message.get("content")
                if isinstance(content, str):
                    metadata.update(_tool_observation_meta(content))
            cursor = _append_context_unit(
                units,
                cursor,
                unit_id=f"history:{original_index}:{message_index}",
                source_type=source_type,
                source_name=role or entry.kind,
                source_ref=f"history[{original_index}].messages[{message_index}]",
                payload=message,
                included_because="rendered_history",
                metadata=metadata,
                classification=_classification_for_source_type(source_type),
                confidence=0.9,
            )

    for tool_index, tool in enumerate(tools or []):
        function = tool.get("function") if isinstance(tool, dict) else {}
        name = ""
        if isinstance(function, dict):
            name = str(function.get("name") or "")
        cursor = _append_context_unit(
            units,
            cursor,
            unit_id=f"tool_schema:{name or tool_index}",
            source_type="tool_schema",
            source_name=name or "tool_schema",
            source_ref=f"tools[{tool_index}]",
            payload=tool,
            included_because="available_tool_schema",
            metadata={"tool_index": tool_index},
            classification=_classification_for_source_type("tool_schema"),
            confidence=0.85,
        )

    warnings: list[str] = []
    if units and cursor != estimate["request_tokens_estimate"]:
        warnings.append("token_positions_are_estimates")
    return {
        "version": 1,
        "token_count_method": "char_estimate",
        "position_method": "estimated_sequential_units",
        "messages_tokens_estimate": estimate["messages_tokens_estimate"],
        "tools_tokens_estimate": estimate["tools_tokens_estimate"],
        "request_tokens_estimate": estimate["request_tokens_estimate"],
        "units": units,
        "warnings": warnings,
    }


def _classification_from_fragment(fragment: ContextFragment) -> dict[str, str]:
    """Берём явную классификацию фрагмента или fallback по source_type."""

    fallback = _classification_for_source_type(_fragment_source_type(fragment))
    explicit = {
        "authority_level": fragment.authority_level,
        "context_layer": fragment.context_layer,
        "evictability": fragment.evictability,
        "stability": fragment.stability,
        "applicability": fragment.applicability,
        "normative_role": fragment.normative_role,
        "goal_role": fragment.goal_role,
    }
    return {
        key: value if value != "unknown" and value != "none" else fallback[key]
        for key, value in explicit.items()
    }


def _classification_for_source_type(source_type: str) -> dict[str, str]:
    """Назначаем базовый слой для старых фрагментов без явной разметки."""

    if source_type == "system_instruction":
        return {
            "authority_level": "system",
            "context_layer": "normative",
            "evictability": "never",
            "stability": "stable",
            "applicability": "active",
            "normative_role": "safety",
            "goal_role": "none",
        }
    if source_type == "developer_instruction":
        return {
            "authority_level": "developer",
            "context_layer": "normative",
            "evictability": "never",
            "stability": "stable",
            "applicability": "active",
            "normative_role": "workflow",
            "goal_role": "none",
        }
    if source_type == "user_message":
        return {
            "authority_level": "user",
            "context_layer": "goal",
            "evictability": "goal_update_only",
            "stability": "task",
            "applicability": "active",
            "normative_role": "none",
            "goal_role": "primary_goal",
        }
    if source_type in {"file_snippet", "test_result"}:
        return {
            "authority_level": "tool",
            "context_layer": "evidence",
            "evictability": "normal",
            "stability": "turn",
            "applicability": "current_task",
            "normative_role": "none",
            "goal_role": "none",
        }
    if source_type == "tool_schema":
        return {
            "authority_level": "tool",
            "context_layer": "tooling",
            "evictability": "normal",
            "stability": "session",
            "applicability": "active",
            "normative_role": "none",
            "goal_role": "none",
        }
    if source_type == "assistant_message":
        return {
            "authority_level": "assistant",
            "context_layer": "working",
            "evictability": "preferred",
            "stability": "turn",
            "applicability": "current_task",
            "normative_role": "none",
            "goal_role": "none",
        }
    if source_type == "tool_output":
        return {
            "authority_level": "tool",
            "context_layer": "working",
            "evictability": "preferred",
            "stability": "turn",
            "applicability": "current_task",
            "normative_role": "none",
            "goal_role": "none",
        }
    return {
        "authority_level": "unknown",
        "context_layer": "unknown",
        "evictability": "normal",
        "stability": "unknown",
        "applicability": "unknown",
        "normative_role": "none",
        "goal_role": "none",
    }


def _user_task_units(user_task: str) -> list[dict[str, Any]]:
    """Разделяем активную цель и вложенные пользовательские данные в telemetry."""

    spans = _attached_data_spans(user_task)
    primary_parts: list[str] = []
    previous = 0
    for start, end in spans:
        primary_parts.append(user_task[previous:start])
        primary_parts.append(ATTACHED_DATA_MARKER)
        previous = end
    primary_parts.append(user_task[previous:])
    primary_payload = "".join(primary_parts).strip() or "user provided attached data"
    attached_chars = sum(end - start for start, end in spans)
    units: list[dict[str, Any]] = [
        {
            "unit_id": "user_task",
            "source_type": "user_message",
            "source_name": "task",
            "source_ref": "user_task",
            "payload": primary_payload,
            "included_because": "current_task",
            "metadata": {
                "chars": len(user_task),
                "goal_anchor_chars": len(primary_payload),
                "attached_data_units": len(spans),
                "attached_data_chars": attached_chars,
            },
            "classification": {
                "authority_level": "user",
                "context_layer": "goal",
                "evictability": "goal_update_only",
                "stability": "task",
                "applicability": "active",
                "normative_role": "none",
                "goal_role": "primary_goal",
            },
            "confidence": 1.0,
        }
    ]
    for index, (start, end) in enumerate(spans):
        payload = user_task[start:end]
        taint_score = _attached_data_taint_score(payload)
        units.append(
            {
                "unit_id": f"user_task_attached:{index}",
                "source_type": "user_message",
                "source_name": "attached_data",
                "source_ref": f"user_task.attached[{index}]",
                "payload": payload,
                "included_because": "user_attached_data",
                "metadata": {
                    "chars": len(payload),
                    "attached_data_index": index,
                    "attached_data_taint_score": round(taint_score, 4),
                    "attachment_kind": _attached_data_kind(payload),
                },
                "classification": {
                    "authority_level": "external" if taint_score >= 0.50 else "user",
                    "context_layer": _attached_data_context_layer(payload),
                    "evictability": "preferred",
                    "stability": "temporary",
                    "applicability": "current_task",
                    "normative_role": "none",
                    "goal_role": "attached_data",
                },
                "confidence": 0.85,
            }
        )
    return units


def _attached_data_spans(text: str) -> list[tuple[int, int]]:
    """Находим вложенные блоки, которые не должны становиться goal anchor."""

    spans = [(match.start(), match.end()) for match in FENCED_BLOCK_RE.finditer(text)]
    heading_matches = list(ATTACHED_HEADING_RE.finditer(text))
    section_heading = re.compile(r"(?m)^#{1,6}\s+\S")
    for match in heading_matches:
        start = match.start()
        if any(span_start <= start < span_end for span_start, span_end in spans):
            continue
        next_match = section_heading.search(text, match.end())
        end = next_match.start() if next_match else len(text)
        spans.append((start, end))
    return _merge_spans(spans)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Склеиваем пересекающиеся диапазоны вложенных данных."""

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _attached_data_taint_score(text: str) -> float:
    """Оцениваем, похож ли вложенный материал на чужие инструкции."""

    lowered = text.lower()
    instruction_markers = (
        "ignore previous instructions",
        "system prompt",
        "developer message",
        "you are chatgpt",
        "follow these instructions",
        "do not obey",
        "не выполняй предыдущие инструкции",
        "системный промпт",
        "инструкция разработчика",
    )
    if any(marker in lowered for marker in instruction_markers):
        return 0.85
    if any(marker in lowered for marker in ("instruction", "инструкция", "prompt")):
        return 0.45
    return 0.0


def _attached_data_kind(text: str) -> str:
    """Помечаем тип вложения для отчета без сохранения самого содержимого."""

    lowered = text.lower()
    if any(marker in lowered for marker in ("trace", "stderr", "stdout", "log", "error")):
        return "log"
    if any(marker in lowered for marker in ("test", "pytest", "unittest", "failed")):
        return "test_output"
    return "external_text"


def _attached_data_context_layer(text: str) -> str:
    """Логи и тесты считаем evidence, остальные вложения — working."""

    if _attached_data_kind(text) in {"log", "test_output"}:
        return "evidence"
    return "working"


def _append_context_unit(
    units: list[dict[str, Any]],
    cursor: int,
    *,
    unit_id: str,
    source_type: str,
    source_name: str,
    source_ref: str,
    payload: Any,
    included_because: str,
    metadata: dict[str, Any],
    classification: dict[str, str],
    confidence: float,
) -> int:
    """Добавляем один элемент prompt-индекса и возвращаем следующую позицию."""

    tokens = estimate_tokens(payload)
    end = cursor + tokens
    units.append(
        {
            "unit_id": unit_id,
            "source_type": source_type,
            "source_name": source_name,
            "source_ref": source_ref,
            "tokens_estimate": tokens,
            "position_start": cursor,
            "position_end": end,
            "included_because": included_because,
            "content_hash": _content_hash(payload),
            "confidence": confidence,
            **classification,
            "metadata": metadata,
        }
    )
    return end


def _content_hash(payload: Any) -> str:
    """Хэшируем содержимое, не записывая сам текст в trace."""

    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def _fragment_source_type(fragment: ContextFragment) -> str:
    """Грубо классифицируем закреплённые фрагменты для аналитики контекста."""

    value = f"{fragment.id} {fragment.source}".lower()
    if "system" in value:
        return "system_instruction"
    if "agents" in value or "project" in value or "instruction" in value:
        return "developer_instruction"
    if "skill" in value:
        return "developer_instruction"
    return "context_fragment"


def _message_source_type(message: dict[str, Any]) -> str:
    """Преобразуем Chat role в тип фрагмента тепловой карты."""

    role = str(message.get("role") or "")
    if role == "assistant":
        return "assistant_message"
    if role == "tool":
        return "tool_output"
    if role == "user":
        return "user_message"
    if role == "system":
        return "system_instruction"
    return "unknown"


def _tool_name(call: dict[str, Any], observation: dict[str, Any]) -> str:
    """Достаём имя инструмента для fallback tool_call_id."""

    function = call.get("function") or {}
    return str(function.get("name") or observation.get("tool") or "tool_call")
