@echo off
setlocal
cd /d "%~dp0"

if exist "runtime\python.exe" (
    echo Starting TensorBoard from the portable runtime...
    "runtime\python.exe" -m tensorboard.main --logdir "trained_models"
    exit /b %errorlevel%
)

if exist ".venv\Scripts\python.exe" (
    echo Starting TensorBoard from the uv environment...
    ".venv\Scripts\python.exe" -m tensorboard.main --logdir "trained_models"
    exit /b %errorlevel%
)

echo TensorBoard was not found. Run "uv sync --extra cu128" first.
pause
exit /b 1
