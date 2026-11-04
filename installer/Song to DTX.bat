@echo off
title Song to DTX
rem Song to DTX (installed version): starts the local app and opens it in the browser.
rem Close this window to stop it.
cd /d "%~dp0"
set "PATH=%~dp0ffmpeg;%~dp0python;%PATH%"
set "TORCH_HOME=%~dp0models\torch"
set "HF_HOME=%~dp0models\huggingface"
set HF_HUB_OFFLINE=1
set PYTHONHOME=
set PYTHONPATH=
set PYTHONNOUSERSITE=1
set PYTHONIOENCODING=utf-8
if not exist "vendor\larsnet\larsnet.py" goto parts
if not exist "vendor\ADTOF-pytorch\src\adtof_pytorch\__init__.py" goto parts
goto run
:parts
"python\python.exe" download_parts.py || (pause & exit /b 1)
:run
"python\python.exe" app\server.py
pause
