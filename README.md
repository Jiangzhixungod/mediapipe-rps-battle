# MediaPipe 双人石头剪刀布

摄像头同时识别两只手。画面经过镜像后，**画面左方**与**画面右方**分别是一位玩家。握拳为石头，伸出食指和中指为剪刀，张开手掌为布。

双方手势稳定后开始 3 秒倒计时；倒计时中有人离开画面或改变手势，会重新等待或重新计时。出拳锁定后判定胜负并计分：

| 胜方手势 | 处决动画 |
| --- | --- |
| 石头 | 陨石流星雨 |
| 剪刀 | 快刀剪乱麻 |
| 布 | 只手遮天 |

动画从胜方位置指向败方位置；平局只显示结果，不播放处决。结果结束后，任一方更换手势或移开手，再开始下一局。按 **Q**、**Esc** 或关闭窗口退出。

## 运行

在本机双击 `run.bat`，或运行：

```bat
E:\mediapipe_project\game_env\Scripts\python.exe E:\mediapipe_project\main.py
```

其他摄像头可指定设备编号：

```bat
E:\mediapipe_project\game_env\Scripts\python.exe E:\mediapipe_project\main.py --camera 1
```

在开发板上可指定视频设备路径：

```bash
python3 main.py --camera /dev/video4
```

首次配置环境时安装 `requirements.txt`。中文界面会优先使用系统中文字体；可通过环境变量 `RPS_FONT` 指定字体文件。当前 Windows 安装的 MediaPipe 版本从中文路径加载内置模型可能失败，因此本机的 `E:\mediapipe_project` 是指向本项目的英文路径目录连接。

## 无摄像头动画预览

运行 `python preview.py` 可重新生成 `rps_finishers_preview.mp4`，依次展示三种处决动画。预览使用模拟对战背景，不需要摄像头。

## 文件

- `main.py`：摄像头、左右方分配和回合状态
- `gesture_classifier.py`：手部关键点分类与稳定判断
- `rps_game.py`：胜负规则和计分
- `effects.py`：三种逐帧处决动画
- `overlay.py`：中文游戏界面
- `preview.py`：MP4 动画预览导出

`animation.py` 是旧版通用攻击界面，现已不再由游戏入口使用。

## 回合验证

运行 `python -B -m unittest test_rounds -v` 可验证完整主循环的六种场景：倒计时丢手、更换出拳、平局、右方获胜、持续保持手势不重复计分，以及更换手势开始下一局。这些检查使用模拟检测结果，无需摄像头。
