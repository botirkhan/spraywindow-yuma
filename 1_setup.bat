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
echo Installing packages... this can take a few minutes.
echo PY=%PY% > logs\setup.log
if not defined PY (
  echo NO PYTHON FOUND >> logs\setup.log
  where python >> logs\setup.log 2>&1
  where py >> logs\setup.log 2>&1
  dir /b "%LOCALAPPDATA%\Programs" >> logs\setup.log 2>&1
  dir /b "%LOCALAPPDATA%\Microsoft\WindowsApps\python*" >> logs\setup.log 2>&1
  goto end
)
%PY% --version >> logs\setup.log 2>&1
%PY% -m pip install -r requirements.txt >> logs\setup.log 2>&1
echo EXIT_CODE %errorlevel% >> logs\setup.log
:end
echo Done. You can close this window.
timeout /t 4
