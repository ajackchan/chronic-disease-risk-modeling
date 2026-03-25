@echo off
setlocal enableextensions
chcp 65001 >nul 2>nul

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo [1/3] Python tests...
python -m pytest -q
if errorlevel 1 (
  echo ERROR: pytest failed.
  exit /b 1
)

echo.
echo [2/3] Frontend tests...
pushd "frontend"
call npm test
if errorlevel 1 (
  popd
  echo ERROR: frontend tests failed.
  exit /b 1
)

echo.
echo [3/3] Frontend build...
call npm run build
if errorlevel 1 (
  popd
  echo ERROR: frontend build failed.
  exit /b 1
)

popd

echo.
echo OK: all checks passed.
exit /b 0