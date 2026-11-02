@echo off
rem Song to DTX setup: Python packages, the two drum models (LarsNet, ADTOF) and a starter drum kit.
rem Needs: Python 3.11 (python.org), internet, about 7 GB of disk. Run it again any time to repair.
setlocal
cd /d "%~dp0"

echo === 1/6 Python
set PY=
py -3.11 -c "import sys" >nul 2>&1 && set PY=py -3.11
if not defined PY python -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1 && set PY=python
if not defined PY (
  echo Python 3.11 was not found. Install it from https://www.python.org/downloads/ and run setup.bat again.
  pause & exit /b 1
)
if not exist .venv\Scripts\python.exe %PY% -m venv .venv
set VPY=.venv\Scripts\python.exe
%VPY% -m pip install --upgrade pip >nul

echo === 2/6 PyTorch
set GPU=Y
set /p GPU="Do you have an NVIDIA graphics card? [Y/n] "
if /i "%GPU%"=="n" (
  %VPY% -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu || goto fail
  if not exist settings.json echo {"device": "cpu"}> settings.json
) else (
  %VPY% -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 || goto fail
)

echo === 3/6 Other packages
%VPY% -m pip install -r requirements.txt || goto fail

echo === 4/6 ADTOF-pytorch (drum transcription) and LarsNet (drum separation) code
if not exist vendor mkdir vendor
if not exist vendor\ADTOF-pytorch\pyproject.toml (
  curl -L -o vendor\adtof.zip https://github.com/xavriley/ADTOF-pytorch/archive/85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9.zip || goto fail
  tar -xf vendor\adtof.zip -C vendor || goto fail
  move vendor\ADTOF-pytorch-85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9 vendor\ADTOF-pytorch >nul
  del vendor\adtof.zip
)
%VPY% -m pip install -e vendor\ADTOF-pytorch || goto fail
if not exist vendor\larsnet\config.yaml (
  curl -L -o vendor\larsnet.zip https://github.com/polimi-ispl/larsnet/archive/17d631fd18e77ee2f1d23ee7b3fc0fb46ae629e2.zip || goto fail
  tar -xf vendor\larsnet.zip -C vendor || goto fail
  move vendor\larsnet-17d631fd18e77ee2f1d23ee7b3fc0fb46ae629e2 vendor\larsnet >nul
  del vendor\larsnet.zip
)

echo === 5/6 LarsNet model weights (562 MB)
if exist vendor\larsnet\pretrained_larsnet_models\kick goto weights_ok
echo The LarsNet weights are by Politecnico di Milano under CC BY-NC 4.0: free for
echo non-commercial use only, with credit. Song to DTX is a hobby tool and is never sold.
set OKW=
set /p OKW="Download them now under that license? [y/N] "
if /i not "%OKW%"=="y" goto weights_manual
%VPY% -m pip install gdown >nul
%VPY% -m gdown "https://drive.google.com/uc?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL" -O vendor\larsnet_models.zip || goto weights_manual
tar -xf vendor\larsnet_models.zip -C vendor\larsnet || goto weights_manual
del vendor\larsnet_models.zip
if exist vendor\larsnet\pretrained_larsnet_models\kick goto weights_ok
:weights_manual
echo.
echo Download the weights yourself: the page opens now. Unzip the file so that this folder exists:
echo   %~dp0vendor\larsnet\pretrained_larsnet_models\kick
echo then run setup.bat again.
start "" "https://drive.google.com/uc?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL&export=download"
pause & exit /b 1
:weights_ok

echo === 6/6 ffmpeg and drum kit
where ffmpeg >nul 2>&1 || (
  echo ffmpeg was not found. Install it, for example:  winget install Gyan.FFmpeg
  echo then open a new window and run setup.bat again.
  pause & exit /b 1
)
%VPY% make_kit.py || goto fail

echo.
echo Setup done. Start the app with "Song to DTX.bat".
pause
exit /b 0

:fail
echo.
echo Setup stopped with an error (see above). Fix it and run setup.bat again.
pause
exit /b 1
