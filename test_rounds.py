"""Exercise the real game loop with repeatable camera detections and time."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import main
from gesture_classifier import Gesture as G
from rps_game import GamePhase as Phase


def simulate(pairs):
    camera = MagicMock()
    cursor = [-1]
    ticks = [-1]
    snapshots = []

    def read():
        cursor[0] += 1
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def detect(_frame):
        detected = []
        for x, gesture in zip((0.25, 0.75), pairs[cursor[0]]):
            if gesture is not G.UNKNOWN:
                detected.append(SimpleNamespace(
                    landmark=[SimpleNamespace(x=x, y=0.5)] * 21,
                    gesture=gesture,
                ))
        return SimpleNamespace(multi_hand_landmarks=detected)

    def now():
        ticks[0] += 1
        return ticks[0] / 10

    def overlay(*args, **kwargs):
        score = args[7]
        snapshots.append((args[1], args[3], args[4], (score.left, score.right, score.draw)))

    camera.isOpened.return_value = True
    camera.read.side_effect = read
    detector = MagicMock()
    detector.__enter__.return_value = detector
    detector.process.side_effect = detect
    patches = [
        patch.object(main, "parse_args", return_value=SimpleNamespace(camera="0")),
        patch.object(main.cv2, "VideoCapture", return_value=camera),
        patch.object(main.mp.solutions.hands, "Hands", return_value=detector),
        patch.object(main.mp.solutions.drawing_utils, "draw_landmarks"),
        patch.object(main, "classify_hand_landmarks", side_effect=lambda hand: hand.gesture),
        patch.object(main.time, "monotonic", side_effect=now),
        patch.object(main, "draw_overlay", side_effect=overlay),
        patch.object(main.cv2, "imshow"),
        patch.object(main.cv2, "getWindowProperty", return_value=1),
        patch.object(main.cv2, "destroyAllWindows"),
        patch.object(main.cv2, "waitKey", side_effect=lambda _: ord("q") if cursor[0] == len(pairs) - 1 else -1),
    ]
    with ExitStack() as stack:
        for context in patches:
            stack.enter_context(context)
        assert main.main() == 0
    camera.release.assert_called_once()
    return snapshots


class RoundFlowTests(unittest.TestCase):
    def test_held_gestures_score_only_once(self):
        states = simulate([(G.ROCK, G.SCISSORS)] * 150)
        self.assertEqual(states[-1][3], (1, 0, 0))
        self.assertIn(Phase.ATTACK, [state[0] for state in states])
        self.assertEqual(states[-1][0], Phase.WAITING)

    def test_missing_hand_cancels_countdown(self):
        states = simulate([(G.ROCK, G.SCISSORS)] * 15 + [(G.ROCK, G.UNKNOWN)] * 80)
        self.assertEqual(states[-1][3], (0, 0, 0))
        self.assertNotIn(Phase.ATTACK, [state[0] for state in states])

    def test_changed_gestures_replace_pending_round(self):
        states = simulate([(G.ROCK, G.SCISSORS)] * 15 + [(G.SCISSORS, G.PAPER)] * 140)
        attacks = [state for state in states if state[0] is Phase.ATTACK]
        self.assertTrue(attacks)
        self.assertTrue(all(state[1:3] == (G.SCISSORS, G.PAPER) for state in attacks))
        self.assertEqual(states[-1][3], (1, 0, 0))

    def test_draw_skips_attack(self):
        states = simulate([(G.ROCK, G.ROCK)] * 120)
        self.assertEqual(states[-1][3], (0, 0, 1))
        self.assertNotIn(Phase.ATTACK, [state[0] for state in states])

    def test_right_player_wins(self):
        states = simulate([(G.ROCK, G.PAPER)] * 120)
        self.assertEqual(states[-1][3], (0, 1, 0))

    def test_new_gesture_rearms_next_round(self):
        states = simulate([(G.ROCK, G.SCISSORS)] * 100 + [(G.SCISSORS, G.PAPER)] * 120)
        self.assertEqual(states[-1][3], (2, 0, 0))


if __name__ == "__main__":
    unittest.main()
