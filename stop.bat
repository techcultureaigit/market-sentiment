@echo off
title Stop Market Sentiment Agent Platform
cd /d "%~dp0"

echo =====================================================================
echo           Stopping Market Sentiment Agent Platform
echo =====================================================================
echo.

echo [1/2] Stopping Web Server on port 8080...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo   - Web server stopped.

echo [2/2] Stopping PostgreSQL server...
"D:\postgres\pgsql\bin\pg_ctl.exe" -D "D:\postgres\data" stop >nul 2>&1
echo   - PostgreSQL stopped.

echo.
echo All services have been stopped cleanly.
timeout /t 2 /nobreak >nul

