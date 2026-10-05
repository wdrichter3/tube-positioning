@echo off
REM Lists connected cameras and installed packages. Result is also saved to %USERPROFILE%\.tubemap\doctor.txt
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run START_TUBEMAP_LAPTOP.bat once first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m tubemap doctor
echo.
pause
