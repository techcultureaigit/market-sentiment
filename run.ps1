# PowerShell Startup Script for Market Sentiment Agent Platform
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "          Starting Market Sentiment Agent Platform                   " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check & Start PostgreSQL
Write-Host "[1/3] Checking PostgreSQL service..." -ForegroundColor Yellow
$pgReady = & "D:\postgres\pgsql\bin\pg_isready.exe" -h localhost -p 5432 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  - PostgreSQL is already running on port 5432." -ForegroundColor Green
} else {
    Write-Host "  - PostgreSQL not detected. Checking for stale lock file..." -ForegroundColor Yellow
    $pidFile = "D:\postgres\data\postmaster.pid"
    if (Test-Path $pidFile) {
        $stalePid = (Get-Content $pidFile)[0]
        $proc = Get-Process -Id $stalePid -ErrorAction SilentlyContinue
        if (-not $proc) {
            Write-Host "  - Cleaning up stale postmaster.pid ($stalePid)..." -ForegroundColor Yellow
            Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "  - Starting PostgreSQL 16 server..." -ForegroundColor Yellow
    & "D:\postgres\pgsql\bin\pg_ctl.exe" -D "D:\postgres\data" -l "D:\postgres\logfile.log" start
    Start-Sleep -Seconds 2
    $check = & "D:\postgres\pgsql\bin\pg_isready.exe" -h localhost -p 5432 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  - PostgreSQL started successfully." -ForegroundColor Green
    }
}
Write-Host ""

# 2. Check Virtual Environment
Write-Host "[2/3] Checking Python virtual environment..." -ForegroundColor Yellow
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Virtual environment not found at $pythonExe" -ForegroundColor Red
    exit 1
}
Write-Host "  - Using: $pythonExe" -ForegroundColor Green
Write-Host ""

# 3. Clean stale instances & launch Browser and Server
Write-Host "[3/3] Preparing Server & Realtime Pipeline..." -ForegroundColor Yellow
$staleConns = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($staleConns) {
    foreach ($c in $staleConns) {
        Write-Host "  - Terminating stale process on port 8080 (PID $($c.OwningProcess))..." -ForegroundColor Yellow
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "  - Dashboard URL: http://localhost:8080/" -ForegroundColor Cyan

# Open default browser after brief delay
Start-Process "http://localhost:8080/"

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "  LIVE CONSOLE RUNNING at http://localhost:8080/" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host ""

& $pythonExe main.py serve --host 0.0.0.0 --port 8080

