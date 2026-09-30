"""Hand landmark classification for rock-paper-scissors."""

from __future__ import annotations

from collections import Counter, deque
from enum import Enum
from math import acos, degrees, hypot
from typing import Sequence


class Gesture(str, Enum):
    UNKNOWN = "unknown"
    ROCK = "rock"
    SCISSORS = "scissors"
    PAPER = "paper"


def _point(landmark) -> tuple[float, float]:
    return float(landmark.x), float(landmark.y)


def _distance(a, b) -> float:
    ax, ay = _point(a)
    bx, by = _point(b)
    return hypot(ax - bx, ay - by)


def _angle(a, b, c) -> float:
    """Return angle ABC in degrees."""
    ax, ay = _point(a)
    bx, by = _point(b)
    cx, cy = _point(c)
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)
    ba_len = hypot(*ba)
    bc_len = hypot(*bc)
    if ba_len == 0 or bc_len == 0:
        return 0.0
    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (ba_len * bc_len)
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _is_extended(landmarks: Sequence, mcp: int, pip: int, tip: int) -> bool:
    straight = _angle(landmarks[mcp], landmarks[pip], landmarks[tip]) > 145.0
    tip_reaches_farther = _distance(landmarks[tip], landmarks[mcp]) > (
        _distance(landmarks[pip], landmarks[mcp]) * 1.05
    )
    return straight and tip_reaches_farther


def classify_hand_landmarks(hand_landmarks) -> Gesture:
    """Classify MediaPipe's 21 hand landmarks."""
    landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
    if landmarks is None or len(landmarks) < 21:
        return Gesture.UNKNOWN

    thumb_extended = (
        _angle(landmarks[2], landmarks[3], landmarks[4]) > 135.0
        and _distance(landmarks[4], landmarks[0])
        > _distance(landmarks[3], landmarks[0]) * 1.05
    )
    fingers = [
        _is_extended(landmarks, 5, 6, 8),
        _is_extended(landmarks, 9, 10, 12),
        _is_extended(landmarks, 13, 14, 16),
        _is_extended(landmarks, 17, 18, 20),
    ]
    extended_count = sum(fingers)
    if extended_count == 0:
        return Gesture.ROCK
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return Gesture.SCISSORS
    if extended_count == 4 or (extended_count >= 3 and thumb_extended):
        return Gesture.PAPER
    return Gesture.UNKNOWN


class GestureStabilizer:
    """Suppress one-frame classification noise."""

    def __init__(self, window_size: int = 8, min_votes: int = 6) -> None:
        if window_size < 1 or not 1 <= min_votes <= window_size:
            raise ValueError("min_votes must be between 1 and window_size")
        self._history: deque[Gesture] = deque(maxlen=window_size)
        self._min_votes = min_votes

    def reset(self) -> None:
        self._history.clear()

    def update(self, gesture: Gesture) -> Gesture:
        self._history.append(gesture)
        candidate, votes = Counter(self._history).most_common(1)[0]
        if candidate is not Gesture.UNKNOWN and votes >= self._min_votes:
            return candidate
        return Gesture.UNKNOWN
