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
echo SprayWindow Yuma is running at http://127.0.0.1:8000
echo Keep this window open. Close it to stop the app.
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"
%PY% server.py > logs\server.log 2>&1
