@echo off
setlocal
rem ---------------------------------------------------------------------------
rem  Stop VibeBot - double-click to shut it down.
rem
rem  Asks the server to close politely first, so the browser it drives shuts
rem  down cleanly and the language model is released. Only if that does not
rem  work does it end the process outright.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
title Stop VibeBot

rem Full paths on purpose: "timeout", "find" and friends are completely
rem different programs when a Unix toolchain (git bash, msys, WSL) sits ahead
rem of System32 on PATH. Waiting uses ping rather than timeout.exe, because
rem timeout refuses to run at all with stdin redirected - which silently
rem turned the wait below into no wait, and killed a server that was already
rem shutting down politely.
set "PING=%SystemRoot%\System32\ping.exe"
set "TASKKILL=%SystemRoot%\System32\taskkill.exe"
set "TASKLIST=%SystemRoot%\System32\tasklist.exe"
set "FINDSTR=%SystemRoot%\System32\findstr.exe"

set "PIDFILE=%~dp0.vibebot\server.pid"
set "PORT=8765"
for /f "tokens=2 delims=: " %%P in ('"%FINDSTR%" /r /c:"^  *port:" "%~dp0config.yaml" 2^>nul') do set "PORT=%%P"

echo.
if not exist "%PIDFILE%" goto :not_running

echo   Asking VibeBot on port %PORT% to stop...
curl -s -m 10 -X POST "http://127.0.0.1:%PORT%/api/quit" >nul 2>&1

rem Up to ~20s to close Chromium, release the model and drop its pid file.
for /l %%I in (1,1,20) do if exist "%PIDFILE%" call :sleep
if not exist "%PIDFILE%" goto :stopped

set /p VBPID=<"%PIDFILE%"
if "%VBPID%"=="" goto :cleanup

rem Only kill it if that PID really is a Python process. A pid file left by a
rem window someone X-ed out can name a number Windows has since handed to
rem something else entirely.
"%TASKLIST%" /fi "PID eq %VBPID%" /fi "IMAGENAME eq python.exe" 2>nul | "%FINDSTR%" /i "python.exe" >nul
if errorlevel 1 goto :stale

echo   It did not close on its own. Ending process %VBPID%...
"%TASKKILL%" /pid %VBPID% /t /f >nul 2>&1
goto :cleanup

:stale
echo   Leftover file from a run that was closed abruptly - nothing to stop.

:cleanup
del "%PIDFILE%" >nul 2>&1

:stopped
echo.
echo   VibeBot is stopped.
echo.
call :sleep
call :sleep
endlocal & exit /b 0

:not_running
echo   VibeBot does not look like it is running.
echo.
call :sleep
call :sleep
endlocal & exit /b 0

:sleep
rem ~1 second, and unlike timeout.exe it works with stdin redirected.
"%PING%" -n 2 127.0.0.1 >nul 2>&1
exit /b 0
