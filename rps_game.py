"""Pure game rules and round state helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gesture_classifier import Gesture


class GamePhase(str, Enum):
    WAITING = "waiting"
    COUNTDOWN = "countdown"
    ATTACK = "attack"
    RESULT = "result"


class RoundResult(str, Enum):
    WIN = "win"
    LOSE = "lose"
    DRAW = "draw"


WIN_RULES = {
    Gesture.ROCK: Gesture.SCISSORS,
    Gesture.SCISSORS: Gesture.PAPER,
    Gesture.PAPER: Gesture.ROCK,
}


@dataclass
class Score:
    left: int = 0
    right: int = 0
    draw: int = 0


def judge(left: Gesture, right: Gesture) -> RoundResult:
    """Return the result from the screen-left player's perspective."""
    if left is Gesture.UNKNOWN or right is Gesture.UNKNOWN:
        raise ValueError("Cannot judge an unknown gesture")
    if left is right:
        return RoundResult.DRAW
    if WIN_RULES[left] is right:
        return RoundResult.WIN
    return RoundResult.LOSE


def update_score(score: Score, result: RoundResult) -> None:
    if result is RoundResult.WIN:
        score.left += 1
    elif result is RoundResult.LOSE:
        score.right += 1
    else:
        score.draw += 1
