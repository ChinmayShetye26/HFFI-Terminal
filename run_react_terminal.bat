@echo off
setlocal
cd /d "%~dp0"

echo Starting HFFI FastAPI backend on http://127.0.0.1:8000
start "HFFI API" powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0scripts\run_terminal_api.ps1"

echo Starting HFFI React frontend on http://127.0.0.1:5173
start "HFFI React" powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0scripts\run_terminal_frontend.ps1"

echo.
echo Open http://127.0.0.1:5173 after both windows finish starting.
pause
