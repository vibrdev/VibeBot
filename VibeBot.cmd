@echo off
setlocal
rem ---------------------------------------------------------------------------
rem  VibeBot - double-click to start.
rem
rem  The first run creates a virtual environment and downloads Chromium, which
rem  takes a few minutes. Every run after that starts in seconds.
rem
rem  To stop: press Quit in the page, double-click "Stop VibeBot", or close
rem  this window.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
title VibeBot

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto :have_venv

echo.
echo   First run - setting VibeBot up. This takes a few minutes.
echo.
call :find_python
if errorlevel 1 goto :no_python

echo   Creating the virtual environment...
%BASE_PY% -m venv ".venv"
if errorlevel 1 goto :setup_failed

echo   Installing packages...
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :setup_failed

echo   Downloading the browser VibeBot drives...
"%VENV_PY%" -m playwright install chromium
if errorlevel 1 goto :setup_failed

echo.
echo   Setup finished.
echo.

:have_venv
if exist "config.yaml" goto :run
if not exist "config.example.yaml" goto :run
echo   No config.yaml yet - starting from the example one.
copy /y "config.example.yaml" "config.yaml" >nul

:run
"%VENV_PY%" -m vibebot serve %*
set "EXITCODE=%ERRORLEVEL%"
if "%EXITCODE%"=="0" goto :done

echo.
echo   VibeBot exited with code %EXITCODE%.
echo   If that was not expected, the lines above say why.
echo.
pause

:done
endlocal & exit /b %EXITCODE%

rem --------------------------------------------------------------- helpers
:find_python
rem Set BASE_PY to a launcher that can create the venv. Not quoted when used:
rem it may be "py -3", which is two words.
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "BASE_PY=py -3" & exit /b 0
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "BASE_PY=python" & exit /b 0
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "BASE_PY=python3" & exit /b 0
exit /b 1

:no_python
echo.
echo   Python 3.10 or newer was not found.
echo   Install it from https://www.python.org/downloads/ , tick
echo   "Add python.exe to PATH" during setup, then run this again.
echo.
pause
endlocal & exit /b 1

:setup_failed
echo.
echo   Setup failed - the messages above say why.
echo.
pause
endlocal & exit /b 1
