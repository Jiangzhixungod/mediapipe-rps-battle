"""Frame-based finishing moves drawn directly over the camera image."""

from __future__ import annotations

import math

import cv2
import numpy as np

from gesture_classifier import Gesture


ATTACK_DURATION = 2.2
ATTACK_NAMES = {
    Gesture.ROCK: "陨石流星雨",
    Gesture.SCISSORS: "快刀剪乱麻",
    Gesture.PAPER: "只手遮天",
}


def _blend(frame, layer, opacity: float) -> None:
    cv2.addWeighted(layer, opacity, frame, 1.0 - opacity, 0, frame)


def _point(center, width: int, height: int) -> tuple[int, int]:
    return int(center[0] * width), int(center[1] * height)


def _impact(frame, target: tuple[int, int], power: float, color) -> None:
    if power <= 0:
        return
    x, y = target
    scale = min(frame.shape[:2]) / 480
    radius = int((18 + 90 * power) * scale)
    cv2.circle(frame, target, radius, color, max(2, int(5 * (1 - power / 2))), cv2.LINE_AA)
    cv2.circle(frame, target, max(3, radius // 2), (245, 245, 255), 2, cv2.LINE_AA)
    for i in range(16):
        angle = i * math.tau / 16
        inner = radius * 0.65
        outer = radius * (1.12 + i % 3 * 0.17)
        start = (int(x + math.cos(angle) * inner), int(y + math.sin(angle) * inner))
        end = (int(x + math.cos(angle) * outer), int(y + math.sin(angle) * outer))
        cv2.line(frame, start, end, color if i % 2 else (255, 255, 255), 2, cv2.LINE_AA)


def _meteor(frame, source: tuple[int, int], target: tuple[int, int], progress: float) -> None:
    height, width = frame.shape[:2]
    tint = np.zeros_like(frame)
    tint[:] = (27, 13, 8)
    _blend(frame, tint, 0.31)
    scale = min(width, height) / 480
    sx, _ = source
    tx, ty = target
    for i in range(15):
        delay = i * 0.042
        travel = (progress - delay) / 0.52
        if travel <= 0:
            continue
        travel = min(1.0, travel)
        spread = ((i * 73) % 169 - 84) * scale
        ex = int(tx + spread)
        ey = int(ty + (((i * 47) % 115) - 57) * scale)
        start_x = int((sx + ex) / 2 - 110 * scale)
        start_y = -int((80 + (i * 59) % 130) * scale)
        head = (int(start_x + (ex - start_x) * travel), int(start_y + (ey - start_y) * travel))
        tail = (int(head[0] - 38 * scale), int(head[1] - 66 * scale))
        if travel < 1:
            cv2.line(frame, tail, head, (8, 65, 174), max(5, int(12 * scale)), cv2.LINE_AA)
            cv2.line(frame, tail, head, (60, 202, 255), max(2, int(5 * scale)), cv2.LINE_AA)
            cv2.circle(frame, head, max(3, int(7 * scale)), (206, 248, 255), -1, cv2.LINE_AA)
        else:
            age = min(1.0, (progress - delay - 0.52) / 0.17)
            if age < 1:
                cv2.circle(frame, (ex, ey), int((5 + 26 * age) * scale), (46, 175, 255), 2, cv2.LINE_AA)
    _impact(frame, target, min(1.0, (progress - 0.61) / 0.34), (30, 173, 255))


def _scissors(frame, source: tuple[int, int], target: tuple[int, int], progress: float) -> None:
    height, width = frame.shape[:2]
    scale = min(width, height) / 480
    sx, sy = source
    tx, ty = target
    # A pair of bright blades rushes from the winning hand to the target.
    dash = min(1.0, progress / 0.42)
    dash = 1 - (1 - dash) ** 3
    head = (int(sx + (tx - sx) * dash), int(sy + (ty - sy) * dash))
    if progress < 0.51:
        cv2.line(frame, source, head, (121, 33, 168), max(7, int(15 * scale)), cv2.LINE_AA)
        cv2.line(frame, source, head, (255, 230, 255), max(2, int(5 * scale)), cv2.LINE_AA)
        cv2.circle(frame, head, max(6, int(15 * scale)), (255, 255, 255), -1, cv2.LINE_AA)

    for i in range(5):
        delay = 0.33 + i * 0.105
        cut = min(1.0, max(0.0, (progress - delay) / 0.20))
        if cut <= 0:
            continue
        offset = (i - 2) * 29 * scale
        direction = -1 if i % 2 else 1
        reach = int((80 + i % 2 * 20) * scale)
        cx, cy = int(tx + offset), int(ty + offset * 0.3)
        start = (cx - reach, cy - direction * reach)
        end = (int(start[0] + 2 * reach * cut), int(start[1] + 2 * direction * reach * cut))
        cv2.line(frame, start, end, (171, 62, 177), max(7, int(13 * scale)), cv2.LINE_AA)
        cv2.line(frame, start, end, (255, 247, 255), max(2, int(4 * scale)), cv2.LINE_AA)
        if cut == 1 and progress - delay < 0.31:
            cv2.circle(frame, end, max(3, int(5 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
    _impact(frame, target, min(1.0, (progress - 0.70) / 0.25), (226, 107, 239))


def _paper(frame, source: tuple[int, int], target: tuple[int, int], progress: float) -> None:
    height, width = frame.shape[:2]
    scale = min(width, height) / 480
    darkness = np.zeros_like(frame)
    darkness[:] = (45, 16, 38)
    _blend(frame, darkness, min(0.56, progress * 0.75))
    move = min(1.0, progress / 0.72)
    move = 1 - (1 - move) ** 3
    cx = int(source[0] + (target[0] - source[0]) * move)
    cy = int(source[1] + (target[1] - source[1]) * move)
    size = (45 + 95 * move) * scale
    palm = np.array([
        (-0.65, -0.33), (0.65, -0.33), (0.77, 0.68),
        (0.35, 1.03), (-0.35, 1.03), (-0.77, 0.68),
    ], dtype=np.float32)
    palm = (palm * size + (cx, cy)).astype(np.int32)
    shadow = frame.copy()
    cv2.fillPoly(shadow, [palm], (87, 39, 89))
    cv2.polylines(shadow, [palm], True, (240, 134, 250), max(2, int(5 * scale)), cv2.LINE_AA)
    for i, length in enumerate((0.58, 0.76, 0.86, 0.75, 0.57)):
        x = int(cx + (i - 2) * size * 0.3)
        top = int(cy - size * (0.38 + length))
        bottom = int(cy - size * 0.14)
        thickness = max(8, int(size * 0.24))
        cv2.line(shadow, (x, bottom), (x, top), (240, 134, 250), thickness + 5, cv2.LINE_AA)
        cv2.line(shadow, (x, bottom), (x, top), (87, 39, 89), thickness, cv2.LINE_AA)
        cv2.circle(shadow, (x, top), thickness // 2, (87, 39, 89), -1, cv2.LINE_AA)
    _blend(frame, shadow, 0.88)
    if progress > 0.7:
        _impact(frame, target, min(1.0, (progress - 0.7) / 0.3), (228, 142, 250))


def draw_finisher(frame, winning_gesture: Gesture, source, target, elapsed: float) -> None:
    """Render one deterministic frame; centers use normalized image coordinates."""
    if source is None or target is None or winning_gesture not in ATTACK_NAMES:
        return
    height, width = frame.shape[:2]
    source_px = _point(source, width, height)
    target_px = _point(target, width, height)
    progress = min(1.0, max(0.0, elapsed / ATTACK_DURATION))
    if winning_gesture is Gesture.ROCK:
        _meteor(frame, source_px, target_px, progress)
    elif winning_gesture is Gesture.SCISSORS:
        _scissors(frame, source_px, target_px, progress)
    else:
        _paper(frame, source_px, target_px, progress)
