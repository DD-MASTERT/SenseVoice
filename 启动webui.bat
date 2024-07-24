@echo off
REM 设置 FFMPEG_PATH 环境变量
SET FFMPEG_PATH=%cd%\ffmpeg\bin

REM 将 FFMPEG_PATH 临时添加到 PATH 环境变量中
SET PATH=%PATH%;%FFMPEG_PATH%

REM 运行 Python 脚本
"py310/python.exe" webui.py

REM 暂停以查看输出
pause