@echo off
rem Song to DTX: starts the local app and opens it in the browser. Close this window to stop it.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause & exit /b 1
)
".venv\Scripts\python.exe" app\server.py
pause
