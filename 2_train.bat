@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
set PY=
python -c "import sys" >nul 2>&1 && set PY=python
if not defined PY (py -3 --version >nul 2>&1 && set PY=py -3)
if not defined PY for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%D\python.exe" set PY="%%D\python.exe"
if not defined PY for /d %%D in ("C:\Python3*") do if exist "%%D\python.exe" set PY="%%D\python.exe"
if not defined PY for /d %%D in ("%ProgramFiles%\Python3*") do if exist "%%D\python.exe" set PY="%%D\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set PY="%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set PY="%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%ProgramData%\anaconda3\python.exe" set PY="%ProgramData%\anaconda3\python.exe"
if not defined PY if exist "%ProgramData%\miniconda3\python.exe" set PY="%ProgramData%\miniconda3\python.exe"
echo Downloading weather data and training the model... about 3-5 minutes.
echo PY=%PY% > logs\train.log
%PY% pipeline.py %* >> logs\train.log 2>&1
echo EXIT_CODE %errorlevel% >> logs\train.log
echo Done. You can close this window.
timeout /t 4
