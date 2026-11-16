@echo off
setlocal enabledelayedexpansion
title Market Sentiment Agent Platform
cd /d "%~dp0"

echo =====================================================================
echo           Starting Market Sentiment Agent Platform
echo =====================================================================
echo.

:: 1. Check & Start PostgreSQL 16
echo [1/3] Checking PostgreSQL service...
set "PG_DIR=D:\postgres\pgsql\bin"
set "PG_DATA=D:\postgres\data"

"%PG_DIR%\pg_isready.exe" -h localhost -p 5432 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   - PostgreSQL is already running on port 5432.
) else (
    echo   - PostgreSQL is not running. Checking for stale lock files...
    if exist "%PG_DATA%\postmaster.pid" (
        set /p STALE_PID=<"%PG_DATA%\postmaster.pid"
        tasklist /FI "PID eq !STALE_PID!" 2>nul | findstr /i "!STALE_PID!" >nul 2>&1
        if !ERRORLEVEL! neq 0 (
            echo   - Detected stale postmaster.pid (!STALE_PID!). Cleaning up...
            del /f /q "%PG_DATA%\postmaster.pid" >nul 2>&1
        )
    )
    echo   - Starting PostgreSQL 16 server...
    "%PG_DIR%\pg_ctl.exe" -D "%PG_DATA%" -l D:\postgres\logfile.log start
    timeout /t 2 /nobreak >nul
    "%PG_DIR%\pg_isready.exe" -h localhost -p 5432 >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo   - PostgreSQL started successfully.
    ) else (
        echo   - PostgreSQL server initialization in progress.
    )
)
echo.

:: 2. Check Virtual Environment
echo [2/3] Verifying Python environment...
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found at: %PYTHON_EXE%
    echo Please ensure the .venv folder exists in this directory.
    pause
    exit /b 1
)
echo   - Python interpreter: %PYTHON_EXE%
echo.

:: 3. Terminate Any Stale Server Instances on port 8080 & Start Clean
echo [3/3] Preparing Web Server & Realtime Pipeline...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do (
    echo   - Terminating stale process on port 8080 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
echo   - Local URL:   http://localhost:8080/

:: Automatically open browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8080/"

echo.
echo =====================================================================
echo  LIVE TEST CONSOLE RUNNING
echo  Press Ctrl+C in this window at any time to stop the server.
echo =====================================================================
echo.
"%PYTHON_EXE%" main.py serve --host 0.0.0.0 --port 8080

pause

