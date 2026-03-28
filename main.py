import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# ===== Цветной вывод в консоль =====
class C:
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def format_time(seconds: float) -> str:
    """
    Преобразует время в секундах в человекочитаемый формат:
        183 -> "3 мин 3 сек"
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    parts = []
    if h:
        parts.append(f"{h} ч")
    if m:
        parts.append(f"{m} мин")
    if s or not parts:
        parts.append(f"{s} сек")

    return " ".join(parts)


def compute_intervals(events: List[Dict], video_duration: float) -> List[Dict]:
    """
    Формирует интервалы между событиями:
    leave → следующий approach
    Округляет delta до 3 знаков после запятой.

    Возвращает список словарей:
    {
        "leave_time": float,
        "approach_time": float,
        "delta": float
    }
    """
    intervals = []

    if not events:
        return intervals

    # === CASE 1: видео начинается с EMPTY ===
    if events[0]["event"] == "approach" and events[0]["time"] > 0:
        intervals.append(
            {
                "leave_time": 0.0,
                "approach_time": events[0]["time"],
                "delta": events[0]["time"],
            }
        )

    last_leave: Optional[float] = None

    for e in events:
        if e["event"] == "leave":
            last_leave = e["time"]

        elif e["event"] == "approach" and last_leave is not None:
            intervals.append(
                {
                    "leave_time": last_leave,
                    "approach_time": e["time"],
                    "delta": round(e["time"] - last_leave, 3),
                }
            )
            last_leave = None

    # === CASE 2: видео заканчивается на EMPTY ===
    if last_leave is not None:
        intervals.append(
            {
                "leave_time": last_leave,
                "approach_time": video_duration,
                "delta": round(video_duration - last_leave, 3),
            }
        )

    return intervals


def save_to_csv(
    intervals: List[Dict], path: str = "output/analytics.csv"
) -> None:
    """
    Сохраняет интервалы в CSV-файл.
    Добавляет человекочитаемые колонки времени.

    Колонки:
    - время_освобождения_сек
    - время_занятия_сек
    - длительность_сек
    - время_освобождения
    - время_занятия
    - длительность
    """
    if not intervals:
        return

    df = pd.DataFrame(intervals)

    # Переименование колонок в русский язык
    df = df.rename(
        columns={
            "leave_time": "время_освобождения_сек",
            "approach_time": "время_занятия_сек",
            "delta": "длительность_сек",
        }
    )

    # Добавление человекочитаемых колонок
    df["время_освобождения"] = df["время_освобождения_сек"].apply(format_time)
    df["время_занятия"] = df["время_занятия_сек"].apply(format_time)
    df["длительность"] = df["длительность_сек"].apply(format_time)

    df.to_csv(path, index=False, encoding="utf-8-sig")


def main(video_path: str, mode: str) -> None:
    """
    Основная функция обработки видео.

    Алгоритм:
    1. Чтение видео
    2. Детекция людей через YOLO
    3. Определение нахождения человека в ROI (зона стола)
    4. Фиксация событий:
        - approach (EMPTY → OCCUPIED)
        - leave (OCCUPIED → EMPTY)
    5. Визуализация + сохранение видео
    6. Аналитика (время между событиями)
    """

    # ===== Чтение видео =====
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: cannot open video")
        return

    # ===== Инициализация модели YOLO =====
    model = YOLO("yolov8n.pt")
    model.overrides["verbose"] = False

    fps: float = cap.get(cv2.CAP_PROP_FPS)
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    # ROI (координаты стола) — фиксированы
    x, y, w, h = 340, 336, 885, 739

    # В dev-режиме YOLO вызывается раз в N кадров
    frame_skip: int = 10 if mode == "dev" else 1

    prev_state: Optional[str] = None
    current_state: str = "EMPTY"

    events: List[Dict] = []

    # ===== Кэш боксов =====
    last_boxes: List[Tuple[int, int, int, int]] = []

    # ===== Настройка выходного видео =====
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if mode == "dev":
        # В dev-режиме уменьшаем разрешение видео (ускоряет генерацию)
        out_width = int(width * 0.5)
        out_height = int(height * 0.5)
    else:
        out_width = width
        out_height = height

    out = cv2.VideoWriter("output/output.mp4", fourcc,
                          fps, (out_width, out_height))

    frame_count: int = 0
    pbar = tqdm(total=total_frames, desc="Processing")

    time_offset: Optional[float] = None

    # ===== Основной цикл по кадрам =====
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pbar.update(1)

        # Время текущего кадра
        timestamp: float = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if time_offset is None:
            time_offset = timestamp
        timestamp -= time_offset

        # ===== Детекция через YOLO =====
        # В dev-режиме детекция раз в frame_skip кадров
        if frame_count == 1 or frame_count % frame_skip == 0:
            results = model(frame, verbose=False)[0]
            last_boxes = []

            # Проходим по всем обнаруженным объектам
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Фильтруем только людей (cls==0) и минимальная уверенность 0.5
                if cls != 0 or conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                last_boxes.append((x1, y1, x2, y2))

        # ===== Используем кэш боксов для промежуточных кадров =====
        person_in_roi = False
        for x1, y1, x2, y2 in last_boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if x <= cx <= x + w and y <= cy <= y + h:
                person_in_roi = True
                color = (0, 0, 255)  # красный — человек в ROI
            else:
                color = (255, 0, 0)  # синий — человек вне ROI

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ===== Обновление состояния столa =====
        current_state = "OCCUPIED" if person_in_roi else "EMPTY"

        # ===== Запись событий =====
        if prev_state is not None:
            if prev_state == "EMPTY" and current_state == "OCCUPIED":
                events.append({"time": timestamp, "event": "approach"})
            elif prev_state == "OCCUPIED" and current_state == "EMPTY":
                events.append({"time": timestamp, "event": "leave"})
        prev_state = current_state

        # ===== ROI на кадре =====
        roi_color = (0, 0, 255) if current_state == "OCCUPIED" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), roi_color, 2)

        # ===== Запись кадра в выходное видео =====
        if mode == "dev":
            # Для dev-режима уменьшаем разрешение, чтобы ускорить генерацию
            frame_small = cv2.resize(frame, (out_width, out_height))
            out.write(frame_small)
        else:
            out.write(frame)

    # ===== Завершение =====
    pbar.close()
    cap.release()
    out.release()

    # ===== Вывод событий =====
    print("\nEvents:\n")
    for e in events:
        t = format_time(e["time"])
        if e["event"] == "approach":
            print(f"{C.YELLOW}{t}{C.END} - {C.RED}Стол занят{C.END}")
        else:
            print(f"{C.YELLOW}{t}{C.END} - {C.GREEN}Стол свободен{C.END}")

    # ===== Аналитика интервалов =====
    intervals = compute_intervals(events, video_duration)
    if intervals:
        avg = sum(i["delta"] for i in intervals) / len(intervals)
        print(
            f"\n{C.BLUE}{C.BOLD}Среднее время простоя стола (очищенное): "
            f"{format_time(avg)}{C.END}"
        )
        save_to_csv(intervals)
        print("CSV сохранён: output/analytics.csv")

    print("\nDone. Saved to output/output.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--mode",
        default="prod",
        choices=["dev", "prod"],
        help="dev = быстрее, prod = точнее",
    )

    args = parser.parse_args()
    main(args.video, args.mode)
