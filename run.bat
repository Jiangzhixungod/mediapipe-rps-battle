@echo off
REM MediaPipe 0.10.x cannot load model files from a Chinese-path directory.
REM This junction exposes this same project through an ASCII-only path.
"E:\mediapipe_project\game_env\Scripts\python.exe" -B "E:\mediapipe_project\main.py" %*
if errorlevel 1 pause
