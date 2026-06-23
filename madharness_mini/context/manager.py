"""Собираем сообщения, которые будут отправлены модели."""

from __future__ import annotations

import copy
import json
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
    estimate_request_tokens,
    estimate_tokens,
)
from .fragments import ContextFragment, ContextProvider, ContextState
from .history import FileRef, HistoryEntry
from .render import render_messages
from .reports import _fragment_report, _history_entry_report
from .sanitize import (
    _digest_old_write_tool_calls,
    _sanitize_assistant_message,
    _tool_kind_and_path,
    _tool_name,
)
from .summary import ReasoningSummarizer
from .telemetry import build_context_packet

# Пределы возрастной эвикции (гипотеза B): старые assistant-рассуждения и
# tool-наблюдения усекаем до этих значений, оставляя «скелет» хода для модели.
SUMMARY_ASSISTANT_LIMIT = 500
SUMMARY_TOOL_LIMIT = 200

# Идентификатор transient-фрагмента с напоминанием о «грязных» файлах.
FILE_STATE_REMINDER_ID = "file-state:reminder"

# Сколько путей показываем в каждой категории file-state-reminder'а. Напоминание
# живёт в transient-фрагменте evictability=normal: без лимита оно разрастается в
# длинных сессиях с массовой эвикцией и само себя убивает через emergency-drop.
# Ограничиваем так же, как COMPACTED_HISTORY_MAX_FILES.
FILE_STATE_REMINDER_MAX_FILES = 15

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
    если правка случалась позже чтения (или файл не перечитывали вовсе). turn —
    это индекс элемента истории, который оставил событие.

    last_*_collapsed_turn помечает ход, на котором текст последнего read/write
    покинул промпт в результате эвикции (digest-fold в dedup/summarize или
    удаление хода целиком в summary-fold/drop). None = авторитетное событие
    этого типа всё ещё видимо модели полным содержимым. Кормит предикат
    _model_knows_current_state и третью категорию file-state-reminder'а.
    """

    last_read_turn: int | None = None
    last_write_turn: int | None = None
    last_read_collapsed_turn: int | None = None
    last_write_collapsed_turn: int | None = None


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
            entries,
            fragments,
            is_protected_read=self._is_protected_read,
            entry_indexes=entry_indexes,
        )
        # read_file, свёрнутые дедупом в дайджест-указатель, теряют полный текст:
        # фиксируем потерю видимости для последующего предиката _model_knows_current_state
        # и reminder'а. Инвариант в _mark_read_collapsed отсечёт свёрнутые
        # исторические чтения при свежем чтении того же пути в окне.
        for folded in deduped_tool_messages:
            if folded.get("is_read_fold") and folded.get("path"):
                self._mark_read_collapsed(folded["path"], folded["original_index"])
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
        context_packet = build_context_packet(
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

        Новое событие восстанавливает видимость: обнуляем соответствующее
        collapsed-поле — актуальный текст снова в промпте, прежняя эвикция уже
        неактуальна. Второе collapsed-поле не трогаем: оно относится к другому
        типу события и могло сработать по старому ходу.
        """

        state = self._file_state.setdefault(ref.path, _FileState())
        if ref.kind == "read":
            state.last_read_turn = (
                turn if state.last_read_turn is None else max(state.last_read_turn, turn)
            )
            state.last_read_collapsed_turn = None
        else:  # write или patch
            state.last_write_turn = (
                turn
                if state.last_write_turn is None
                else max(state.last_write_turn, turn)
            )
            state.last_write_collapsed_turn = None

    def _mark_read_collapsed(self, path: str, entry_index: int) -> None:
        """Фиксируем, что текст последнего чтения пути покинул промпт.

        Помечаем только если свёрнутый/удалённый ход совпадает с last_read_turn:
        если в окне есть более свежее чтение того же пути, модель не слепая и
        помечать нельзя. Сравнение строгим равенством (а не «<=») принципиально:
        свернуть можно лишь авторитетное событие, историческое трогать не нужно.
        """

        state = self._file_state.get(path)
        if state is None or state.last_read_turn != entry_index:
            return
        state.last_read_collapsed_turn = entry_index

    def _mark_write_collapsed(self, path: str, entry_index: int) -> None:
        """Фиксируем, что текст последней правки пути покинул промпт.

        Симметрично _mark_read_collapsed, но для write/patch. Помечаем только при
        совпадении свёрнутого хода с last_write_turn.
        """

        state = self._file_state.get(path)
        if state is None or state.last_write_turn != entry_index:
            return
        state.last_write_collapsed_turn = entry_index

    def _model_knows_current_state(self, path: str) -> bool:
        """Знает ли модель текущее состояние файла.

        Авторитет — последнее по времени событие (read или write). Знание есть,
        если авторитетное событие всё ещё видно в промпте полным содержимым:
        observation для read, args для write/patch. Более старые события не
        помогают — они показывают состояние до авторитетного.

        Tie-break при равенстве ходов за write: в одном ходе write новее read и
        отражает post-write состояние, а read показывал pre-write (уже устарел).

        clip_tool_messages намеренно не учитывается: частичная видимость лучше
        ничего, и свернуть полный текст до excerpt'а — не та же потеря, что
        digest или удаление хода. Это сознательное MVP-ограничение, документируется
        тестом clipped-чтение → «знает».
        """

        state = self._file_state.get(path)
        if state is None:
            return False
        last_read = state.last_read_turn if state.last_read_turn is not None else -1
        last_write = state.last_write_turn if state.last_write_turn is not None else -1
        if last_read < 0 and last_write < 0:
            return False
        if last_read > last_write:
            return state.last_read_collapsed_turn is None
        return state.last_write_collapsed_turn is None

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

    def _collapsed_authority_files(self) -> list[tuple[str, int, str]]:
        """Пути, по которым модель работала, но потеряла видимость.

        Возвращаем тройки (путь, turn авторитетного события, тип события), где
        авторитетное событие свернулось (digest-fold) или удалилось вместе с
        ходом. Фильтруем через _model_knows_current_state == False и требуем,
        чтобы по пути было хотя бы одно событие: путь, которого не касались, сюда
        не относится (его и напоминать незачем). Сортируем по убыванию turn
        авторитетного события.

        Тип события ('read' | 'write') нужен формулировке reminder'а: модель
        должна понять, что именно она потеряла — чтение файла или свою правку.
        """

        collapsed: list[tuple[str, int, str]] = []
        for path, state in self._file_state.items():
            if self._model_knows_current_state(path):
                continue
            last_read = state.last_read_turn
            last_write = state.last_write_turn
            # Путь без событий не относится к этой категории — для него нет
            # «потерянной» видимости, его提醒ает never_read, если он правлен.
            if last_read is None and last_write is None:
                continue
            # Авторитет — последнее событие (tie-break за write, как в предикате).
            if last_read is not None and (
                last_write is None or last_read > last_write
            ):
                collapsed.append((path, last_read, "read"))
            elif last_write is not None:
                collapsed.append((path, last_write, "write"))
        collapsed.sort(key=lambda item: item[1], reverse=True)
        return collapsed

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
        """Собираем transient-напоминание о файлах, требующих перечитывания.

        Три категории, ортогональные по причине слепоты:

        1. never_read — файл записан без единого чтения. Модель никогда не видела
           содержимое, правит вслепую. Жёстко просим read_file перед правкой.
        2. stale_after_write — читали раньше правки и не перечитали. Модель видела
           pre-write состояние, но файл уже изменён.
        3. collapsed_authority — модель видела актуальное состояние (чтение или
           свою правку), но harness свернул это наблюдение в digest-указатель или
           выкинул ход. Это новая категория visibility-трекинга: без неё модель
           продолжает действовать по «вытесненной» памяти, не зная, что harness её
           ослепил. Пути из первых двух категорий сюда не дублируем.
        """

        never_read, stale_after_write = self._dirty_files_by_category()
        # Каждую категорию ограничиваем FILE_STATE_REMINDER_MAX_FILES: напоминание
        # живёт в transient-фрагменте evictability=normal, и без лимита массовая
        # эвикция разрастает его так, что emergency-drop убирает сам фрагмент.
        never_read = never_read[:FILE_STATE_REMINDER_MAX_FILES]
        stale_after_write = stale_after_write[:FILE_STATE_REMINDER_MAX_FILES]
        dirty_paths = {path for path, _ in never_read} | {
            path for path, _ in stale_after_write
        }
        collapsed = [
            item
            for item in self._collapsed_authority_files()
            if item[0] not in dirty_paths
        ][:FILE_STATE_REMINDER_MAX_FILES]
        if not never_read and not stale_after_write and not collapsed:
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
        if collapsed:
            # Компактный inline-формат: collapsed-категория активна именно в
            # длинных сессиях с массовой эвикцией, где бюджет уже под давлением.
            # Многословное описание каждого файла раздуло бы transient-фрагмент
            # так, что emergency-drop убрал бы его (и соседние) целиком. Поэтому
            # только пути через запятую — модель видит, какие файлы нужно
            # перечитать, без перегрузки бюджета.
            lines.append(
                "Эти файлы вы ранее читали или правили, но harness свернул "
                "соответствующее наблюдение — состояние больше не в контексте. "
                "Перед правкой вызовите read_file:"
            )
            paths_text = ", ".join(path for path, _turn, _kind in collapsed)
            lines.append(f"- {paths_text}")
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
                    digested_paths = _digest_old_write_tool_calls(message)
                    if digested_paths:
                        # Текст последней правки каждого свёрнутого пути покинул
                        # промпт: фиксируем потерю write-видимости. Инвариант в
                        # хелпере отсечёт «историческую» правку при более свежей.
                        for path in digested_paths:
                            self._mark_write_collapsed(path, original_index)
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
                            # Текст последнего чтения пути свернулся в указатель:
                            # фиксируем потерю видимости. Инвариант в хелпере
                            # отсечёт свёрнутое «историческое» чтение, если в окне
                            # есть более свежее чтение того же пути.
                            if path:
                                self._mark_read_collapsed(path, original_index)
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

        Для удаляемых ходов фиксируем потерю видимости их read/write событий:
        текст покинул промпт целиком, модель теряет знание о файле. Защищённые
        чтения остаются в рендере, поэтому их не помечаем.
        """

        if self._summarized_upto <= 0:
            return
        kept: list[tuple[HistoryEntry, int]] = []
        for entry, index in zip(entries, entry_indexes):
            if index >= self._summarized_upto or self._entry_has_protected_read(
                entry, index
            ):
                kept.append((entry, index))
                continue
            # Ход уходит из рендера: его read/write файловые эффекты больше не
            # видны модели. Помечаем только пути, чьё авторитетное событие лежит
            # в этом ходе — инвариант в хелперах отсечёт «исторические» события.
            for ref in entry.file_refs:
                if ref.kind == "read":
                    self._mark_read_collapsed(ref.path, index)
                elif ref.kind in ("write", "patch"):
                    self._mark_write_collapsed(ref.path, index)
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
            # Ход выбывает из рендера целиком: фиксируем потерю видимости его
            # read/write-событий. В forced-режиме сюда попадает и protected-read,
            # которого _entry_has_protected_read уже не защитил — модель
            # гарантированно слепая, и помечать обязательно. Инвариант в хелперах
            # отсечёт «исторические» события при более свежем в окне.
            for ref in entries[removable].file_refs:
                if ref.kind == "read":
                    self._mark_read_collapsed(ref.path, entry_indexes[removable])
                elif ref.kind in ("write", "patch"):
                    self._mark_write_collapsed(ref.path, entry_indexes[removable])
            del entries[removable]
            del entry_indexes[removable]
            protected_start = max(len(entries) - keep_recent, 0)
            messages = render_messages(self.user_task, fragments, entries)
        return dropped
