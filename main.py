import argparse
import math
import sys
import time

import cv2
import mediapipe as mp

from overlay import draw_overlay
from effects import ATTACK_DURATION
from gesture_classifier import Gesture, GestureStabilizer, classify_hand_landmarks
from rps_game import GamePhase, RoundResult, Score, judge, update_score


COUNTDOWN_SECONDS = 3.0
ATTACK_SECONDS = ATTACK_DURATION
RESULT_SECONDS = 2.5


def _camera_source(value: str):
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play rock-paper-scissors with a camera.")
    parser.add_argument("--camera", default="0", help="Camera index or device path, for example /dev/video4.")
    return parser.parse_args()


def assign_screen_sides(detected_hands):
    """Assign hands to visible left/right slots (the camera image is mirrored)."""
    hands = list(detected_hands)
    hands.sort(key=lambda hand: sum(p.x for p in hand.landmark) / len(hand.landmark))
    if len(hands) >= 2:
        return hands[0], hands[-1]
    if hands:
        center_x = sum(p.x for p in hands[0].landmark) / len(hands[0].landmark)
        return (hands[0], None) if center_x < 0.5 else (None, hands[0])
    return None, None


def main() -> int:
    """Open the camera and play rock-paper-scissors."""
    args = parse_args()
    camera = cv2.VideoCapture(_camera_source(args.camera))
    if not camera.isOpened():
        print(f"错误：无法打开摄像头 {args.camera}。请检查连接、权限或设备编号。")
        return 1

    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    stabilizers = [GestureStabilizer(window_size=8, min_votes=6), GestureStabilizer(window_size=8, min_votes=6)]
    score = Score()
    phase = GamePhase.WAITING
    phase_started = time.monotonic()
    candidates = [Gesture.UNKNOWN, Gesture.UNKNOWN]
    locked_gestures = [None, None]
    result = None
    locked_centers = [None, None]
    armed = True
    rearm_since = None
    previous_locked = [None, None]

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("错误：摄像头已打开，但无法读取画面。")
                    return 1

                frame = cv2.flip(frame, 1)
                detection = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                side_hands = assign_screen_sides(detection.multi_hand_landmarks or [])
                stable_gestures = []
                raw_gestures = []
                centers = []
                for index, landmarks in enumerate(side_hands):
                    if landmarks is not None:
                        drawer.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                        detected = classify_hand_landmarks(landmarks)
                        centers.append((sum(p.x for p in landmarks.landmark) / len(landmarks.landmark), sum(p.y for p in landmarks.landmark) / len(landmarks.landmark)))
                    else:
                        detected = Gesture.UNKNOWN
                        centers.append(None)
                    raw_gestures.append(detected)
                    if detected is Gesture.UNKNOWN:
                        stabilizers[index].reset()
                        stable_gestures.append(Gesture.UNKNOWN)
                    else:
                        stable_gestures.append(stabilizers[index].update(detected))
                now = time.monotonic()
                elapsed = now - phase_started

                if phase is GamePhase.WAITING:
                    if not armed:
                        changed = any(raw_gestures[i] != previous_locked[i] for i in range(2))
                        rearm_since = (rearm_since or now) if changed else None
                        if rearm_since is not None and now - rearm_since >= 0.25:
                            armed = True
                            rearm_since = None
                            for stabilizer in stabilizers:
                                stabilizer.reset()
                    elif all(g is not Gesture.UNKNOWN for g in stable_gestures):
                        candidates = stable_gestures[:]
                        phase, phase_started = GamePhase.COUNTDOWN, now
                        elapsed = 0.0
                elif phase is GamePhase.COUNTDOWN:
                    if raw_gestures != candidates or stable_gestures != candidates:
                        phase, phase_started = GamePhase.WAITING, now
                        candidates = [Gesture.UNKNOWN, Gesture.UNKNOWN]
                        elapsed = 0.0
                    elif elapsed >= COUNTDOWN_SECONDS:
                        locked_gestures = candidates[:]
                        locked_centers = centers[:]
                        result = judge(locked_gestures[0], locked_gestures[1])
                        update_score(score, result)
                        phase = GamePhase.RESULT if result is RoundResult.DRAW else GamePhase.ATTACK
                        phase_started = now
                        elapsed = 0.0
                elif phase is GamePhase.ATTACK and elapsed >= ATTACK_SECONDS:
                    phase, phase_started = GamePhase.RESULT, now
                    elapsed = 0.0
                elif phase is GamePhase.RESULT and elapsed >= RESULT_SECONDS:
                    phase, phase_started = GamePhase.WAITING, now
                    armed = False
                    previous_locked = locked_gestures[:]
                    candidates = [Gesture.UNKNOWN, Gesture.UNKNOWN]
                    locked_gestures = [None, None]
                    locked_centers = [None, None]
                    result = None
                    elapsed = 0.0

                countdown = max(1, math.ceil(COUNTDOWN_SECONDS - elapsed))
                draw_overlay(
                    frame,
                    phase,
                    (stable_gestures[0], stable_gestures[1]) if phase is GamePhase.WAITING else (candidates[0], candidates[1]),
                    locked_gestures[0],
                    locked_gestures[1],
                    result,
                    countdown,
                    score,
                    elapsed,
                    (locked_centers[0], locked_centers[1]) if phase in (GamePhase.ATTACK, GamePhase.RESULT) else (centers[0], centers[1]),
                    armed=armed,
                )
                cv2.imshow("MediaPipe Rock Paper Scissors", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    return 0
                if cv2.getWindowProperty("MediaPipe Rock Paper Scissors", cv2.WND_PROP_VISIBLE) < 1:
                    return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
