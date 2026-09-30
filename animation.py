"""OpenCV-only visuals for the rock-paper-scissors game."""

from __future__ import annotations

import math
from typing import Optional

import cv2

from gesture_classifier import Gesture
from rps_game import GamePhase, RoundResult, Score


GESTURE_TEXT = {
    Gesture.UNKNOWN: "Unknown",
    Gesture.ROCK: "Rock",
    Gesture.SCISSORS: "Scissors",
    Gesture.PAPER: "Paper",
}
RESULT_TEXT = {RoundResult.WIN: "YOU WIN!", RoundResult.LOSE: "YOU LOSE", RoundResult.DRAW: "DRAW"}
RESULT_COLOR = {RoundResult.WIN: (50, 210, 50), RoundResult.LOSE: (50, 50, 220), RoundResult.DRAW: (0, 190, 220)}


def _center_text(frame, text: str, y: int, scale: float, color, thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (frame.shape[1] - width) // 2)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _card(frame, x: int, y: int, width: int, height: int, title: str, value: str) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + height), (45, 45, 45), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (180, 180, 180), 2)
    cv2.putText(frame, title, (x + 18, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 2)
    cv2.putText(frame, value, (x + 18, y + 83), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def _hand_center(hand_landmarks) -> tuple[int, int] | None:
    if hand_landmarks is None:
        return None
    points = getattr(hand_landmarks, "landmark", hand_landmarks)
    if not points:
        return None
    x = sum(float(p.x) for p in points) / len(points)
    y = sum(float(p.y) for p in points) / len(points)
    return int(x), int(y)


def _draw_attack_animation(frame, attacker, target, elapsed: float, result: RoundResult | None) -> None:
    """Draw a cartoon strike from the winning hand to the losing hand."""
    if attacker is None or target is None or result is None or result is RoundResult.DRAW:
        return
    h, w = frame.shape[:2]
    ax, ay = int(attacker[0] * w), int(attacker[1] * h)
    tx, ty = int(target[0] * w), int(target[1] * h)
    progress = min(1.0, max(0.0, elapsed / 0.9))
    # Overshoot-and-settle motion gives the strike a cartoon feel.
    eased = 1.0 - (1.0 - progress) ** 3
    sx = int(ax + (tx - ax) * eased)
    sy = int(ay + (ty - ay) * eased)
    cv2.line(frame, (ax, ay), (sx, sy), (0, 215, 255), 8, cv2.LINE_AA)
    cv2.circle(frame, (sx, sy), 26, (0, 230, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (sx, sy), 30, (255, 255, 255), 3, cv2.LINE_AA)
    if progress > 0.72:
        impact = min(1.0, (progress - 0.72) / 0.28)
        radius = int(12 + impact * 70)
        cv2.circle(frame, (tx, ty), radius, (30, 30, 255), 4, cv2.LINE_AA)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.line(frame, (tx, ty), (tx + dx * radius, ty + dy * radius), (255, 255, 255), 2, cv2.LINE_AA)
        # Deterministic shards keep the effect stable from frame to frame.
        for index in range(12):
            angle = index * math.pi / 6.0
            distance = int(24 + impact * (35 + (index % 4) * 13))
            px = int(tx + math.cos(angle) * distance)
            py = int(ty + math.sin(angle) * distance)
            size = 4 + index % 4
            cv2.fillPoly(
                frame,
                [cv2.ellipse2Poly((px, py), (size, size // 2 + 1), index * 17, 0, 360, 45)],
                (40, 210, 255) if index % 2 else (255, 255, 255),
            )


def draw_overlay(frame, phase: GamePhase, stable_gestures: tuple[Gesture, Gesture], player_gesture: Optional[Gesture], computer_gesture: Optional[Gesture], result: Optional[RoundResult], countdown: int, score: Score, elapsed: float, hand_centers: tuple[tuple[int, int] | None, tuple[int, int] | None] = (None, None)) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 112), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    _card(frame, 18, 14, 190, 86, "PLAYER 1", GESTURE_TEXT[stable_gestures[0]])
    _card(frame, width - 208, 14, 190, 86, "PLAYER 2", GESTURE_TEXT[stable_gestures[1]])

    if phase is GamePhase.WAITING:
        _center_text(frame, "Show ROCK, PAPER or SCISSORS", height // 2, 0.9, (255, 255, 255))
        _center_text(frame, "Hold a stable gesture to start", height // 2 + 42, 0.65, (190, 190, 190))
    elif phase is GamePhase.COUNTDOWN:
        _center_text(frame, str(max(1, countdown)), height // 2 + 35, 3.2, (0, 220, 255), 5)
        _center_text(frame, "Keep your hand still", height // 2 + 88, 0.75, (255, 255, 255))
    elif phase is GamePhase.ATTACK:
        attacker, target = hand_centers
        if result is RoundResult.LOSE:
            attacker, target = target, attacker
        _draw_attack_animation(frame, attacker, target, elapsed, result)
        _center_text(frame, "ATTACK!", height // 2, 1.3, (0, 220, 255), 3)
    elif phase is GamePhase.RESULT and result is not None:
        color = RESULT_COLOR[result]
        pulse = int(8 + 7 * abs(math.sin(elapsed * 10)))
        cv2.rectangle(frame, (pulse, pulse), (width - pulse, height - pulse), color, 5)
        _center_text(frame, RESULT_TEXT[result], height // 2 - 18, 1.8, color, 4)
        _center_text(frame, f"You: {GESTURE_TEXT[player_gesture]}   CPU: {GESTURE_TEXT[computer_gesture]}", height // 2 + 40, 0.8, (255, 255, 255))
        _center_text(frame, "Next round starts soon", height - 35, 0.6, (210, 210, 210))
    _card(frame, width // 2 - 95, 14, 190, 86, "SCORE", f"{score.player} - {score.computer}")
    cv2.putText(frame, "Q: quit", (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2)
