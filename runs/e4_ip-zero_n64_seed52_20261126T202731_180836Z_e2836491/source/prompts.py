"""Role-local instructions, independent of the selected model backend."""
from __future__ import annotations

import inspect
import json

from .config import RunConfig
from .core import ALICE, View, function_bit
from .protocols import (
    DISJ_ALPHA, DISJ_DENOMINATOR, IP_ZERO_PARTNER, IP_ZERO_TRIGGERS, LocalMove,
    baseline, capacity_width, eq_zero, replay_local_steps, run_subprotocol, transmit_number,
)

SELF_CHECK = ("Перед ходом самостоятельно восстанови нужные промежуточные значения "
              "из своего входа и локальной истории. Проверь, что выбранное действие "
              "соответствует правилу текущего раунда.")
EXPLAIN_PREVIOUS_MOVE = (
    "Коротко укажи, какие промежуточные значения ты использовал для предыдущего хода "
    "и какое правило применил. Если не помнишь или не уверен в значении, так и скажи. "
    "Оставь предыдущий ход как есть. На этот вопрос ответь обычным текстом."
)
TOKEN_REPLY = "Верни ровно один токен: SEND_0, SEND_1, LISTEN, ANSWER_0 или ANSWER_1."
ACTION_FIELD = "В action допустимы SEND_0, SEND_1, LISTEN, ANSWER_0 или ANSWER_1. Без Markdown."

DESCRIPTIONS = {
    "send_x_then_result": """Первые n раундов: Alice отправляет биты x слева направо, Bob слушает.
В раунде n+1 Bob вычисляет f по полученным битам и своему y, отправляет ответ,
Alice слушает. Затем оба выдают ANSWER с этим битом.""",
    "ip_silence": """Если n нечётно, допиши справа один нулевой бит. Разбей свой вход на пары.
Пусть k=ceil(n/2). Первые k раундов обрабатывают по одной паре.
Таблица Alice: 00 -> SEND_0; 01 -> SEND_1; 10 -> LISTEN; 11 -> LISTEN.
Таблица Bob:   00 -> SEND_0; 01 -> LISTEN; 10 -> SEND_1; 11 -> LISTEN.
Локальный аккумулятор изначально 0. После каждого из этих раундов:
для пары 00 ничего не меняй; для 01 или 10 инвертируй аккумулятор только
при наблюдении 1; для 11 инвертируй, если наблюдение НЕ равно 0, включая SILENCE.
В раунде k+1 Alice отправляет свой аккумулятор, Bob слушает.
В раунде k+2 Bob отправляет XOR полученного бита и своего аккумулятора,
Alice слушает. Затем оба отвечают этим результатом.""",
    "ip_zero": """Дополни нечётный вход нулём справа и разбей его на пары слева направо.
Alice выбирает пару, которая чаще остальных встречается в её входе. При равенстве
частот берёт первую в порядке 00, 01, 10, 11 и отправляет её два бита по очереди.
Bob в первые два раунда слушает. По этой паре оба выбирают правила из таблицы:

Пара Alice | проход 1 Alice / Bob | проход 2 Alice / Bob
00         | 00 / 00             | 01 / 10
01         | 01 / 10             | 00 / 00
10         | 10 / 01             | 11 / 11
11         | 11 / 11             | 10 / 01

В первом проходе обработай все позиции пар по порядку. Отправь SEND_1, если твоя
пара равна указанной для твоей роли, иначе LISTEN. Во втором проходе обработай
только позиции, где в первом проходе ты слушал и получил 0, в их исходном порядке.
В проходах отправляется только 1, поэтому эти позиции у обоих совпадут.
Локальный аккумулятор сначала равен чётности числа твоих пар 11. В каждом проходе
инвертируй его ровно тогда, когда получил 1 и пара другого игрока, указанная для
этого прохода, является партнёром твоей пары. Партнёры: 00 ↔ 11, 01 ↔ 01, 10 ↔ 10.
После двух проходов Alice отправляет свой аккумулятор, Bob слушает. Затем Bob
отправляет XOR обоих аккумуляторов, Alice слушает. Оба отвечают этим результатом.""",
    "eq_silence": """Прочитай свой битовый вход как неотрицательное целое число, закодируй в
основании 5 ровно k цифрами, с ведущими нулями, старшая цифра первой.
k — минимальное целое с 5**k >= 2**n. Первые k раундов обрабатывают цифры:
цифра       0        1        2        3        4
Alice     SEND_0   SEND_1   LISTEN   LISTEN   LISTEN
Bob       LISTEN   LISTEN   SEND_0   SEND_1   LISTEN
Каждый слушатель сравнивает наблюдение с тем, что было бы при равных цифрах:
Alice: для 2 ожидает 0, для 3 ожидает 1, для 4 ожидает SILENCE;
Bob: для 0 ожидает 0, для 1 ожидает 1, для 4 ожидает SILENCE.
Несовпадение запоминается; отправитель не получает информации о другом игроке.
В раунде k+1 отправь SEND_0, если заметил несовпадение, иначе LISTEN.
После него, если сам отправлял или услышал бит, сразу выдай ANSWER_0.
Если наблюдал SILENCE, продолжай. Пусть w — минимальное целое с 3**w >= k+1.
Alice считает, сколько раз в первых k раундах она СЛУШАЛА И ПОЛУЧИЛА БИТ 0/1.
Следующие w раундов она передаёт этот счётчик в основании 3, ровно w цифрами
с ведущими нулями, старшая первой: цифры 0/1 через SEND_0/SEND_1, цифру 2 через LISTEN.
Bob во всех w раундах слушает и декодирует SILENCE как цифру 2.
В следующем раунде Bob отправляет 1, если принятый счётчик равен количеству
ЕГО отправок в первых k раундах, иначе 0; Alice слушает. Затем оба выдают ответ.
Не завершайся раньше фазы флага, даже если заметил несовпадение.""",
    "eq_zero": """Прочитай свой битовый вход как неотрицательное целое число, запиши в
основании 3 ровно k цифрами, с ведущими нулями, старшая первой.
k — минимальное целое с 3**k >= 2**n. Первые k раундов обрабатывают цифры:
цифра       0        1        2
Alice     LISTEN   SEND_1   LISTEN
Bob       LISTEN   LISTEN   SEND_1
При равных цифрах Alice, слушая с цифрой 0, ожидает 0; с цифрой 2 ожидает 1.
Bob, слушая с цифрой 0, ожидает 0; с цифрой 1 ожидает 1.
Несовпадение запоминается. Отправители не знают, был ли бит принят.
Пусть w — минимальное целое с 2**w >= k+1. Каждый считает свои отправки SEND_1
в первых k раундах и количество ПОЛУЧЕННЫХ единиц в тех же раундах.
Следующие w раундов Alice передаёт число СВОИХ отправок в двоичной записи
с ведущими нулями, старший бит первым; Bob слушает.
Следующие w раундов Bob аналогично передаёт число СВОИХ отправок; Alice слушает.
Если число отправок другого игрока не равно числу твоих полученных единиц,
также запомни несовпадение. В раунде k+2*w+1 каждый отправляет SEND_1,
если заметил несовпадение, иначе LISTEN. Затем выдай ANSWER_0, если сам
отправлял в этом раунде или услышал 1; иначе ANSWER_1.
Не останавливайся до завершающего раунда флага.""",
    "disj_zero": """Работай со своей текущей битовой строкой. При рекурсии её длина меняется.
Если строка пуста, выдай ANSWER_1. Если в ней один бит, Alice отправляет его Bob,
Bob отправляет результат DISJ для двух битов, затем оба отвечают результатом.
При нечётной длине больше 1 Alice отправляет последний бит, затем Bob отправляет
свой последний бит. Если оба бита равны 1, выдай ANSWER_0. Иначе отбрось последний бит.

Разбей оставшуюся строку на k пар слева направо. Bob передаёт Alice свои счётчики
пар 00, 01, 10, затем Alice передаёт свои в том же порядке. Каждый счётчик передаётся
обычными двоичными битами от старшего к младшему, ровно w бит, где w — минимальное
целое с 2**w > k. При передаче один отправляет, другой слушает. Счётчик 11 равен k
минус сумма остальных трёх. Обозначь счётчики Alice через A, Bob через B.
Выбери первый выполненный порог и его пару:

1. 5000*(A[01]+B[01]) >= 2559*k → 01.
2. 5000*(A[00]+A[11]) >= 2441*k → 00.
3. 5000*(A[10]+B[10]) >= 2559*k → 10.
4. 5000*(B[00]+B[11]) >= 2441*k → 00.

Следующие k раундов идут по парам слева направо. На выбранной паре отправь SEND_1,
на остальных слушай. Сохрани в том же порядке только свои пары, на которых получил 0.
Если выбрано 01 или 10, отметь пересечение, когда получил 1 на своей паре 11
либо принял меньше единиц, чем другой игрок объявил выбранных пар.
Если выбрано 00, отметь пересечение, когда получил 0 на своей паре 11.
Затем Alice отправляет бит своего флага, Bob слушает, затем Bob отправляет свой флаг,
Alice слушает. Если хотя бы один флаг равен 1, оба выдают ANSWER_0. Если ни одной пары
не осталось, оба выдают ANSWER_1.

После 01 замени каждую оставшуюся пару её первым битом и повтори этот протокол DISJ
с уменьшенной строкой. После 10 используй второй бит и также повтори DISJ.
После 00 Alice берёт первый бит каждой оставшейся пары, Bob берёт второй бит.
Оба переходят к описанному ниже протоколу EQ для полученных строк. Его ответ и есть DISJ.""",
}


def format_input(bits: str, kind: str) -> tuple[str, str]:
    if kind == "bits":
        return bits, "Битовая строка; старший бит слева."
    if kind == "bit_list":
        return json.dumps([int(b) for b in bits]), "Список битов слева направо."
    positions = [i for i, b in enumerate(bits, 1) if b == "1"]
    return json.dumps(positions), "Список позиций единиц, индексы с 1 слева; остальные позиции нулевые."


def _bob_accumulator_instructions(config: RunConfig, view: View) -> str:
    if view.round_number == 3:
        transition = (
            "Первый проход начинается. Самостоятельно вычисли начальный аккумулятор "
            "как чётность числа своих пар 11. Запиши этот бит в accumulator_before "
            "и accumulator_after. Два раунда объявления пары в обновление не входят."
        )
    else:
        before = "Возьми свой accumulator_after из предыдущего ответа и запиши его в accumulator_before. "
        finish = ""
        if config.allow_bob_accumulator_correction:
            before += (
                "До отправки итогового бита, если заметил ошибку в своей записи, пересчитай аккумулятор "
                "из своего входа и истории проходов до предыдущего раунда, исключая этот раунд. "
                "Запиши исправленное значение в accumulator_before. В calculation укажи старый бит, "
                "исправленный бит и причину. Уже сделанные ходы и полученные наблюдения используй как факты. "
            )
            finish = " После отправки итогового бита заверши игру с тем битом, который уже отправил."
        transition = (
            f"{before}"
            "Если предыдущий раунд был частью прохода, примени правило обновления "
            "к наблюдению именно того раунда. Если предыдущий раунд был вне проходов, "
            "сохрани прежний бит. Запиши итог в accumulator_after. Для отправки результата "
            f"вычисли XOR accumulator_after и полученного от Alice бита.{finish}"
        )
    return (
        f'{transition} Верни один JSON-объект с полями в порядке "accumulator_before", '
        '"accumulator_after", "calculation", "action". Два поля аккумулятора содержат по одному '
        'целому биту. В calculation укажи причину обновления или сохранения аккумулятора, '
        'текущую фазу и значения для хода. В action запиши выбранный токен. '
        + ACTION_FIELD
    )


def alice_tracks_second_pass(config: RunConfig, view: View) -> bool:
    return (config.track_alice_second_pass and view.role == ALICE and config.alice.backend == "codex"
            and view.round_number >= 3 + (config.n + 1) // 2)


def alice_tracks_accumulator(config: RunConfig, view: View) -> bool:
    return (config.track_alice_accumulator and view.role == ALICE and config.alice.backend == "codex"
            and view.round_number >= 3)


def _alice_second_pass_update(config: RunConfig, view: View) -> str:
    if view.round_number == 3 + (config.n + 1) // 2:
        update = (
            "Первый проход завершён. По своей истории составь список исходных позиций пар, "
            "где ты слушала и получила 0 именно в первом проходе. Нумерация начинается с 1. "
            "Запиши позиции в исходном порядке в remaining_positions."
        )
    else:
        update = (
            "Возьми remaining_positions из своего предыдущего ответа. Если в предыдущем раунде "
            "ты обработала позицию второго прохода, убери первую позицию из списка. "
            "Запиши оставшиеся позиции перед текущим ходом в remaining_positions."
        )
    return (
        f"{update} Если в списке есть позиции, сейчас обработай только первую по правилу второго прохода. "
        "Если список пуст и ты ещё не передавала аккумулятор, передай его сейчас. "
        "После передачи аккумулятора слушай результат Bob, затем выдай итоговое решение. "
    )


def _alice_second_pass_instructions(config: RunConfig, view: View) -> str:
    return (
        _alice_second_pass_update(config, view)
        + 'Верни один JSON-объект с полями в порядке "remaining_positions", "calculation", "action". '
        "В remaining_positions укажи список целых номеров позиций, в calculation — текущую фазу "
        "и использованные локальные значения, в action — выбранный токен. "
        + ACTION_FIELD
    )


def _alice_accumulator_instructions(config: RunConfig, view: View) -> str:
    if view.round_number == 3:
        transition = (
            "Начинается первый проход. Посчитай свои пары 11 и запиши чётность их числа "
            "в оба поля аккумулятора."
        )
    else:
        transition = (
            "Возьми accumulator_after из своего предыдущего ответа как accumulator_before. "
            "Обнови бит по наблюдению в предыдущем раунде. Инвертируй его ровно тогда, когда "
            "предыдущий раунд был частью прохода, ты получила 1 и пара из правила Bob для "
            "этого прохода была партнёром твоей пары в той позиции. В остальных случаях "
            "сохрани бит. Запиши итог в accumulator_after."
        )
    second_pass = alice_tracks_second_pass(config, view)
    positions_update = _alice_second_pass_update(config, view) if second_pass else ""
    positions_field = ', "remaining_positions"' if second_pass else ""
    return (
        f"{transition} При отправке своего аккумулятора передай accumulator_after. "
        f"{positions_update}"
        f'Верни один JSON-объект с полями в порядке "accumulator_before", '
        f'"accumulator_after"{positions_field}, "calculation", "action". '
        "В обоих полях аккумулятора запиши целый бит. В calculation кратко укажи фазу, "
        "свои промежуточные значения и причину инверсии или сохранения бита. "
        "В action запиши выбранный токен. "
        + ACTION_FIELD
    )


def calculation_instructions(config: RunConfig, view: View) -> str | None:
    if not config.record_calculations:
        return None
    if view.role == ALICE:
        if alice_tracks_accumulator(config, view):
            return _alice_accumulator_instructions(config, view)
        if alice_tracks_second_pass(config, view):
            return _alice_second_pass_instructions(config, view)
        if config.alice.backend != "codex" or view.round_number > 2:
            return None
        details = "Дополни вход при необходимости. Запиши пары, их частоты и выбранную пару."
    else:
        if config.bob.backend != "codex":
            return None
        if config.track_bob_accumulator:
            return _bob_accumulator_instructions(config, view) if view.round_number > 2 else None
        if view.round_number <= 2 + (config.n + 1) // 2:
            return None
        details = (
            "Определи текущую фазу. Во втором проходе запиши оставшиеся позиции, текущую позицию, "
            "свою пару и пару из правила. Перед отправкой результата запиши свой аккумулятор, "
            "полученный от Alice бит и результат XOR. Для остальных фаз укажи локальные значения, "
            "которые использовал для хода."
        )
    return (
        f"{details} Верни один JSON-объект. Сначала запиши эти промежуточные значения коротким "
        'текстом в поле "calculation", затем запиши токен действия в поле "action". '
        + ACTION_FIELD
    )


def render_prompt(config: RunConfig, view: View, *, persistent: bool = False) -> str:
    encoded, meaning = format_input(view.own_input, config.input_format)
    task = {"ip": "IP: чётность числа позиций с двумя единицами.",
            "eq": "EQ: 1 тогда и только тогда, когда x и y равны.",
            "disj": "DISJ: 1 тогда и только тогда, когда общих единичных позиций нет."}[config.task]
    implementation = config.implementation
    description = DESCRIPTIONS[implementation.name]
    if implementation.name == "disj_zero":
        description += "\n\nПодпротокол EQ для ветви 00. Здесь n — длина полученной строки.\n" + DESCRIPTIONS["eq_zero"]
    if config.explanation == "pseudocode":
        rule = implementation.choose
        rules = [rule]
        constants = ""
        if implementation.local_steps is not None:
            rules.extend((LocalMove, replay_local_steps, transmit_number, implementation.local_steps))
            if implementation.name == "ip_zero":
                constants = f"IP_ZERO_TRIGGERS = {IP_ZERO_TRIGGERS!r}\nIP_ZERO_PARTNER = {IP_ZERO_PARTNER!r}\n"
            else:
                constants = f"DISJ_ALPHA = {DISJ_ALPHA}\nDISJ_DENOMINATOR = {DISJ_DENOMINATOR}\n"
                rules.extend((run_subprotocol, baseline, eq_zero, function_bit))
        description += ("\n\nФормальное локальное правило на Python:\n" + constants
                        + "\n".join(inspect.getsource(member) for member in rules)
                        + "\nВспомогательные операции: digits(v,b,w) — ровно w цифр v в основании b,"
                        " старшая первая; capacity_width(s,b) — минимальное w с b**w>=s;"
                        " received_bit(o) — int(o) для 0/1; decode(obs,b) — число по наблюдениям,"
                        " в основании 3 SILENCE обозначает 2. ALICE='Alice', BOB='Bob'."
                        " view.history содержит только свои action и observation; n — длина входа.")
    history = ""
    if not persistent:
        history = "\nЛокальная история:\n" + ("\n".join(
            f"{i}: действие {t.action}, наблюдение {t.observation if t.observation is not None else 'нет'}"
            for i, t in enumerate(view.history, 1)) or "Пусто.")
    silence = "специальный символ SILENCE" if config.channel == "silence" else "обычный бит 0"
    settings = config.alice if view.role == "Alice" else config.bob
    tools = ("Можно использовать инструмент python для локальных вычислений."
             if settings.allow_python else "Вычислительные инструменты отключены.")
    self_check = SELF_CHECK + "\n" if config.self_check else ""
    reply = calculation_instructions(config, view) or (TOKEN_REPLY + "\nБез пояснений, кавычек и Markdown.")
    return f"""Ты исполняешь роль {view.role} в полудуплексном протоколе. n={config.n}.
Alice знает только x, Bob знает только y. {task}
В раунде оба независимо выбирают SEND_0, SEND_1 или LISTEN.
Если отправляет один, бит получает слушатель; у отправителя наблюдение «нет».
Если отправляют оба, биты теряются; оба наблюдают «нет», столкновение им не сообщается.
Если слушают оба, каждый наблюдает {silence}.
ANSWER_0/ANSWER_1 — отдельное итоговое решение без передачи по каналу. Оба должны
завершить одновременно по правилам протокола. Любой SEND/LISTEN стоит один раунд.
Лимит: {config.max_rounds} раундов. Наблюдения 0, 1, SILENCE и «нет» различаются.

Протокол {config.protocol} ({implementation.name}):
{description}

Твой приватный вход: {encoded}
Формат: {meaning} Восстанови свою строку длины n локально.
{history}
Сейчас нужно решение для позиции {view.round_number}. Применяй заданные правила,
включая локальные условия остановки. При продолжении беседы сохраняй свою историю.
{tools} Интернет недоступен.
{self_check}{reply}"""


def render_followup(config: RunConfig, view: View) -> str:
    t = view.history[-1]
    observation = t.observation if t.observation is not None else "нет"
    self_check = SELF_CHECK + " " if config.self_check else ""
    reply = calculation_instructions(config, view) or TOKEN_REPLY
    return (f"Раунд {len(view.history)} завершён. Твоё действие: {t.action}. "
            f"Твоё наблюдение: {observation}.\n"
            f"Выбери решение для позиции {view.round_number} по заданному протоколу, "
            f"включая условие остановки. {self_check}{reply}")
