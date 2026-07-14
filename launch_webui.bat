@echo off
setlocal
cd /d "%~dp0"

if exist "runtime\python.exe" (
    "runtime\python.exe" webui.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" webui.py
) else (
    uv run --locked --extra cu128 python webui.py
)

if errorlevel 1 pause
