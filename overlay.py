"""Chinese game HUD and finishing-move presentation."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image, ImageDraw, ImageFont

from effects import ATTACK_NAMES, draw_finisher
from gesture_classifier import Gesture
from rps_game import GamePhase, RoundResult, Score


GESTURE_TEXT = {
    Gesture.UNKNOWN: "等待出拳",
    Gesture.ROCK: "石头",
    Gesture.SCISSORS: "剪刀",
    Gesture.PAPER: "布",
}
FONT_PATHS = (
    os.environ.get("RPS_FONT", ""),
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)


@lru_cache(maxsize=24)
def _font(size: int):
    for path in FONT_PATHS:
        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def _draw_texts(frame, labels) -> None:
    if not labels:
        return
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(image)
    for text, x, y, size, color, anchor in labels:
        drawer.text((int(x), int(y)), text, font=_font(max(11, int(size))), fill=color, anchor=anchor,
                    stroke_width=1 if size >= 24 else 0, stroke_fill=(10, 10, 17))
    frame[:] = cv2.cvtColor(__import__("numpy").asarray(image), cv2.COLOR_RGB2BGR)


def _panel(frame, x, y, width, height, color=(29, 32, 47)):
    layer = frame.copy()
    cv2.rectangle(layer, (x, y), (x + width, y + height), color, -1)
    cv2.addWeighted(layer, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (105, 116, 146), 1, cv2.LINE_AA)


def draw_overlay(
    frame,
    phase: GamePhase,
    stable_gestures: tuple[Gesture, Gesture],
    left_gesture: Optional[Gesture],
    right_gesture: Optional[Gesture],
    result: Optional[RoundResult],
    countdown: int,
    score: Score,
    elapsed: float,
    hand_centers=(None, None),
    *,
    armed: bool = True,
) -> None:
    height, width = frame.shape[:2]
    unit = min(width / 960, height / 540)
    unit = max(0.63, unit)
    labels = []

    if phase is GamePhase.ATTACK and result is not None and result is not RoundResult.DRAW:
        winner = left_gesture if result is RoundResult.WIN else right_gesture
        source, target = hand_centers if result is RoundResult.WIN else hand_centers[::-1]
        draw_finisher(frame, winner, source, target, elapsed)

    header_h = max(76, int(102 * unit))
    header = frame.copy()
    cv2.rectangle(header, (0, 0), (width, header_h), (13, 15, 25), -1)
    cv2.addWeighted(header, 0.81, frame, 0.19, 0, frame)
    gap = max(8, int(18 * unit))
    side_width = max(70, int(width * 0.27))
    score_width = max(80, width - 2 * side_width - 4 * gap)
    panel_y = max(7, int(12 * unit))
    panel_h = header_h - panel_y - max(7, int(12 * unit))
    _panel(frame, gap, panel_y, side_width, panel_h)
    _panel(frame, width - gap - side_width, panel_y, side_width, panel_h)
    _panel(frame, 2 * gap + side_width, panel_y, score_width, panel_h)

    small = max(13, int(18 * unit))
    large = max(20, int(30 * unit))
    top = panel_y + panel_h * 0.27
    bottom = panel_y + panel_h * 0.70
    labels += [
        ("左方", gap + side_width / 2, top, small, (160, 218, 255), "mm"),
        (GESTURE_TEXT[stable_gestures[0]], gap + side_width / 2, bottom, large, (255, 255, 255), "mm"),
        ("右方", width - gap - side_width / 2, top, small, (255, 170, 205), "mm"),
        (GESTURE_TEXT[stable_gestures[1]], width - gap - side_width / 2, bottom, large, (255, 255, 255), "mm"),
        (f"比分 · 平局 {score.draw}", width / 2, top, small, (197, 198, 206), "mm"),
        (f"{score.left}  :  {score.right}", width / 2, bottom, large, (255, 229, 140), "mm"),
    ]

    center_y = max(header_h + 75, int(height * 0.49))
    title_size = max(25, int(46 * unit))
    subtitle_size = max(16, int(23 * unit))
    if phase is GamePhase.WAITING:
        message = "请双方出拳" if armed else "更换手势或移开一只手，开始下一局"
        labels.append((message, width / 2, center_y, title_size if armed else subtitle_size, (255, 255, 255), "mm"))
        if armed:
            labels.append(("石头 · 剪刀 · 布", width / 2, center_y + 49 * unit, subtitle_size, (206, 218, 239), "mm"))
    elif phase is GamePhase.COUNTDOWN:
        labels.append((str(max(1, countdown)), width / 2, center_y, max(68, int(110 * unit)), (255, 221, 83), "mm"))
        labels.append(("保持手势，准备对决", width / 2, center_y + 78 * unit, subtitle_size, (255, 255, 255), "mm"))
    elif phase is GamePhase.ATTACK and result is not None:
        winner = left_gesture if result is RoundResult.WIN else right_gesture
        winner_name = "左方" if result is RoundResult.WIN else "右方"
        banner_y = height - max(96, int(112 * unit))
        _panel(frame, int(width * 0.13), banner_y, int(width * 0.74), int(78 * unit), (29, 18, 44))
        labels.append((f"{winner_name}处决 · {ATTACK_NAMES[winner]}", width / 2, banner_y + 38 * unit,
                       max(23, int(36 * unit)), (255, 242, 255), "mm"))
    elif phase is GamePhase.RESULT and result is not None:
        if result is RoundResult.DRAW:
            title, color = "平局", (255, 225, 92)
        else:
            title = "左方胜利" if result is RoundResult.WIN else "右方胜利"
            color = (119, 235, 255) if result is RoundResult.WIN else (255, 141, 187)
        inset = int((8 + 5 * abs(math.sin(elapsed * 8))) * unit)
        cv2.rectangle(frame, (inset, inset), (width - inset, height - inset), color[::-1], max(2, int(4 * unit)), cv2.LINE_AA)
        labels.append((title, width / 2, center_y - 20 * unit, title_size, color, "mm"))
        left_name = GESTURE_TEXT.get(left_gesture, "未知")
        right_name = GESTURE_TEXT.get(right_gesture, "未知")
        labels.append((f"{left_name}  VS  {right_name}", width / 2, center_y + 41 * unit, subtitle_size, (255, 255, 255), "mm"))
        labels.append(("下一局即将开始", width / 2, height - 41 * unit, small, (220, 224, 238), "mm"))

    labels.append(("Q 退出", gap, height - max(16, int(24 * unit)), small, (237, 239, 246), "lm"))
    _draw_texts(frame, labels)
