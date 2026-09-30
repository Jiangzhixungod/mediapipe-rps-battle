"""Export a camera-free preview of all three finishing moves."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from effects import ATTACK_DURATION
from gesture_classifier import Gesture
from overlay import draw_overlay
from rps_game import GamePhase, RoundResult, Score


WIDTH, HEIGHT = 960, 540
LEFT_CENTER = (0.27, 0.56)
RIGHT_CENTER = (0.73, 0.56)
MATCHUPS = (
    (Gesture.ROCK, Gesture.SCISSORS),
    (Gesture.SCISSORS, Gesture.PAPER),
    (Gesture.PAPER, Gesture.ROCK),
)


def _background():
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = np.clip(55 + y * 0.035, 0, 255)
    frame[:, :, 1] = np.clip(34 + x * 0.011, 0, 255)
    frame[:, :, 2] = np.clip(29 + y * 0.015, 0, 255)
    cv2.line(frame, (WIDTH // 2, 120), (WIDTH // 2, HEIGHT - 20), (95, 92, 102), 1, cv2.LINE_AA)
    for center, color in ((LEFT_CENTER, (255, 180, 92)), (RIGHT_CENTER, (105, 141, 255))):
        px = (int(center[0] * WIDTH), int(center[1] * HEIGHT))
        cv2.circle(frame, px, 56, color, 3, cv2.LINE_AA)
        cv2.circle(frame, px, 42, (52, 55, 78), -1, cv2.LINE_AA)
        cv2.circle(frame, px, 42, color, 2, cv2.LINE_AA)
        cv2.circle(frame, px, 16, color, -1, cv2.LINE_AA)
    return frame


def render_preview(output: Path, fps: int = 24) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("无法创建 MP4 视频；请检查 OpenCV 的视频编码支持。")
    score = Score()
    base = _background()
    try:
        for left, right in MATCHUPS:
            score.left += 1
            attack_frames = round(ATTACK_DURATION * fps)
            result_frames = round(0.8 * fps)
            for index in range(attack_frames + result_frames):
                frame = base.copy()
                if index < attack_frames:
                    phase = GamePhase.ATTACK
                    elapsed = index / fps
                else:
                    phase = GamePhase.RESULT
                    elapsed = (index - attack_frames) / fps
                draw_overlay(frame, phase, (left, right), left, right, RoundResult.WIN,
                             0, score, elapsed, (LEFT_CENTER, RIGHT_CENTER))
                writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出三种处决动画预览")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("rps_finishers_preview.mp4"))
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps 必须大于 0")
    render_preview(args.output, args.fps)
    print(f"动画预览已生成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
