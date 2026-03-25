@echo off
setlocal enableextensions enabledelayedexpansion
chcp 65001 >nul 2>nul

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist ".run" mkdir ".run" >nul 2>nul

del ".run\backend.out.log" >nul 2>nul
del ".run\backend.err.log" >nul 2>nul
del ".run\frontend.out.log" >nul 2>nul
del ".run\frontend.err.log" >nul 2>nul

del ".run\pids.env" >nul 2>nul

set "MODE=%~1"
if /I "%MODE%"=="" set "MODE=prod"

if /I not "%MODE%"=="prod" if /I not "%MODE%"=="dev" (
  echo Usage: start.cmd [prod^|dev]
  echo.
  echo   prod: build frontend and run backend at http://127.0.0.1:8000/
  echo   dev : run backend + vite dev server at http://127.0.0.1:5173/
  exit /b 1
)

echo Starting in %MODE% mode...

echo Checking Python/uvicorn...
python -V >nul 2>nul
if errorlevel 1 (
  echo ERROR: python not found in PATH.
  echo Please install Python 3.11+ and ensure `python` is available.
  exit /b 1
)

python -m uvicorn --version >nul 2>nul
if errorlevel 1 (
  echo ERROR: uvicorn is not installed.
  echo Run: python -m pip install -r requirements.txt
  exit /b 1
)

if /I "%MODE%"=="prod" (
  if not exist "frontend\node_modules" (
    echo WARN: frontend\node_modules not found. If build fails, run:
    echo   cd frontend ^&^& npm install
  )

  echo Building frontend...
  pushd "frontend"
  call npm run build
  if errorlevel 1 (
    popd
    echo ERROR: frontend build failed.
    exit /b 1
  )
  popd
)

echo Launching processes (logs in .run\*.log)...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$root = (Resolve-Path -LiteralPath '.' ).Path;" ^
  "$run = Join-Path $root '.run'; New-Item -ItemType Directory -Force -Path $run | Out-Null;" ^
  "$backendOut = Join-Path $run 'backend.out.log';" ^
  "$backendErr = Join-Path $run 'backend.err.log';" ^
  "$pidsPath = Join-Path $run 'pids.env';" ^
  "$backend = Start-Process -FilePath 'python' -WorkingDirectory $root -ArgumentList @('-m','uvicorn','backend.app.main:create_app','--factory','--reload','--host','127.0.0.1','--port','8000') -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr -PassThru;" ^
  "$lines = @('backend=' + $backend.Id);" ^
  "if ($env:MODE -eq 'dev') {" ^
  "  $fe = Join-Path $root 'frontend';" ^
  "  $feOut = Join-Path $run 'frontend.out.log';" ^
  "  $feErr = Join-Path $run 'frontend.err.log';" ^
  "  $front = Start-Process -FilePath 'npm' -WorkingDirectory $fe -ArgumentList @('run','dev','--','--host','127.0.0.1','--port','5173') -RedirectStandardOutput $feOut -RedirectStandardError $feErr -PassThru;" ^
  "  $lines += ('frontend=' + $front.Id);" ^
  "}" ^
  "Set-Content -LiteralPath $pidsPath -Value $lines -Encoding Ascii;" ^
  "function Wait-Port([int]$port,[int]$sec){ $deadline=(Get-Date).AddSeconds($sec); while((Get-Date) -lt $deadline){ if((Test-NetConnection 127.0.0.1 -Port $port).TcpTestSucceeded){ return $true }; Start-Sleep -Milliseconds 250 }; return $false }" ^
  "if(-not (Wait-Port 8000 20)){ Write-Host 'BACKEND_PORT_NOT_READY'; exit 2 }" ^
  "if ($env:MODE -eq 'dev' -and -not (Wait-Port 5173 25)) { Write-Host 'FRONTEND_PORT_NOT_READY'; exit 3 }" ^
  "exit 0"

if errorlevel 3 (
  echo ERROR: Vite dev server did not start listening on 5173.
  echo See: .run\frontend.err.log and .run\frontend.out.log
  exit /b 1
)

if errorlevel 2 (
  echo ERROR: Backend did not start listening on 8000.
  echo See: .run\backend.err.log and .run\backend.out.log
  echo.
  echo Last 40 lines of backend.err.log:
  powershell -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '.run\\backend.err.log'){ Get-Content -Tail 40 '.run\\backend.err.log' }"
  exit /b 1
)

if errorlevel 1 (
  echo ERROR: Failed to launch processes.
  echo See: .run\backend.err.log
  exit /b 1
)

if /I "%MODE%"=="dev" (
  start "" "http://127.0.0.1:5173/"
) else (
  start "" "http://127.0.0.1:8000/"
)

echo Started.
echo To stop: close.cmd
exit /b 0