@echo off
setlocal enableextensions enabledelayedexpansion
chcp 65001 >nul 2>nul

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PIDS_FILE=.run\pids.env"

if exist "%PIDS_FILE%" (
  for /f "usebackq tokens=1,2 delims==" %%A in ("%PIDS_FILE%") do (
    set "NAME=%%A"
    set "PID=%%B"
    if not "!PID!"=="" (
      echo Stopping !NAME! (PID !PID!)...
      taskkill /PID !PID! /T /F >nul 2>nul
    )
  )

  del "%PIDS_FILE%" >nul 2>nul
) else (
  echo INFO: %PIDS_FILE% not found.
)

echo Cleaning leftover listeners on ports 8000 and 5173 (if any)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr LISTENING') do (
  taskkill /PID %%p /T /F >nul 2>nul
)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5173" ^| findstr LISTENING') do (
  taskkill /PID %%p /T /F >nul 2>nul
)

echo Done.
exit /b 0