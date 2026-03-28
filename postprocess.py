"""
Скрипт постобработки для фильтрации шумных событий в аналитике детекции столов.
Удаляет слишком короткие интервалы "стол занят"
(approach → leave) и пересчитывает среднее время простоя.
Позволяет задать порог фильтрации через аргумент командной строки.
"""

import argparse
from typing import Any, Dict, List

import pandas as pd

# ANSI-коды для синего и жирного
C_BLUE_BOLD = "\033[1;34m"
C_END = "\033[0m"


def format_time(seconds: float) -> str:
    """
    Преобразует время в секундах в человекочитаемый формат.

    Args:
        seconds: Время в секундах

    Returns:
        Строка вида "X мин Y сек" или "Y сек"
    """
    seconds = int(seconds)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes} мин {secs} сек" if minutes else f"{secs} сек"


def load_events(
        csv_path: str = "output/analytics.csv"
) -> List[Dict[str, Any]]:
    """
    Загружает события из CSV-файла, созданного основной программой.

    Args:
        csv_path: Путь к файлу с аналитикой

    Returns:
        Список событий, каждое с полями "time" (float)
        и "event" ("approach" / "leave")
    """
    df = pd.read_csv(csv_path)

    events = []
    for _, row in df.iterrows():
        # Событие "освобождение стола" (уход гостей)
        events.append(
            {"time": row["время_освобождения_сек"], "event": "leave"})
        # Событие "занятие стола" (приход гостей)
        events.append({"time": row["время_занятия_сек"], "event": "approach"})

    events.sort(key=lambda x: x["time"])
    return events


def filter_events(
    events: List[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """
    Фильтрует слишком короткие интервалы между approach и leave.

    Алгоритм:
        1. Проходит по списку событий, ища пары approach → leave подряд
        2. Если разница между ними меньше порога — оба события удаляются
        3. Остальные события сохраняются в исходном порядке

    Args:
        events: Список событий
        threshold: Минимальная длительность посадки в секундах

    Returns:
        Отфильтрованный список событий
    """
    filtered = []
    i = 0

    while i < len(events) - 1:
        current = events[i]
        nxt = events[i + 1]

        # Проверяем, образуют ли два подряд идущих события
        # пару approach → leave
        if current["event"] == "approach" and nxt["event"] == "leave":
            duration = nxt["time"] - current["time"]

            if duration < threshold:
                # Слишком короткая посадка — пропускаем оба события
                i += 2
                continue

        filtered.append(current)
        i += 1

    # Добавляем последний элемент, если он не был обработан
    if i == len(events) - 1:
        filtered.append(events[-1])

    return filtered


def rebuild_intervals(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Восстанавливает интервалы из очищенного списка событий.

    Алгоритм:
        - При встрече "leave" запоминаем время освобождения
        - При встрече следующего "approach" формируем интервал

    Args:
        events: Список событий

    Returns:
        Список интервалов с полями:
            - время_освобождения_сек
            - время_занятия_сек
            - длительность_сек
    """
    intervals = []
    last_leave = None

    for e in events:
        if e["event"] == "leave":
            last_leave = e["time"]

        elif e["event"] == "approach" and last_leave is not None:
            intervals.append(
                {
                    "время_освобождения_сек": last_leave,
                    "время_занятия_сек": e["time"],
                    "длительность_сек": e["time"] - last_leave,
                }
            )
            last_leave = None

    return intervals


def save_clean_analytics(
    intervals: List[Dict[str, Any]],
    output_path: str = "output/analytics_clean.csv"
) -> None:
    """
    Сохраняет очищенные интервалы в CSV с человекочитаемыми полями.

    Args:
        intervals: Список интервалов
        output_path: Путь для сохранения
    """
    if not intervals:
        print("Нет данных для сохранения")
        return

    df = pd.DataFrame(intervals)

    # Добавляем человекочитаемые поля
    df["время_освобождения"] = df["время_освобождения_сек"].apply(format_time)
    df["время_занятия"] = df["время_занятия_сек"].apply(format_time)
    df["длительность"] = df["длительность_сек"].apply(format_time)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {output_path}")


def main() -> None:
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description=("Фильтрация коротких посадок и пересчёт "
                     "среднего времени простоя стола")
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Минимальная длительность посадки в секундах (по умолчанию: 10)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/analytics.csv",
        help="Путь к входному CSV-файлу (по умолчанию: output/analytics.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/analytics_clean.csv",
        help=("Путь для сохранения результата "
              "(по умолчанию: output/analytics_clean.csv)"),
    )

    args = parser.parse_args()

    print(f"Загрузка событий из {args.input}...")
    events = load_events(args.input)

    print(f"Фильтрация событий (порог: {args.threshold} сек)...")
    filtered_events = filter_events(events, args.threshold)

    print("Восстановление интервалов...")
    intervals = rebuild_intervals(filtered_events)

    if not intervals:
        print("Нет данных после фильтрации")
        return

    # Вычисляем среднее время
    avg = sum(i["длительность_сек"] for i in intervals) / len(intervals)
    print(
        f"\n{C_BLUE_BOLD}Среднее время простоя стола (очищенное): "
        f"{format_time(avg)}{C_END}\n"
    )

    save_clean_analytics(intervals, args.output)


if __name__ == "__main__":
    main()
