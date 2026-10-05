@echo off
REM Optional, once per camera: lets the Astra+ report its own frame timestamps on Windows.
REM Windows will ask for administrator permission. Plug the Astra+ in first.
cd /d "%~dp0windows"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows\obsensor_metadata_win10.ps1" -op install
echo.
echo If an administrator window opened and closed, the registry keys were added. Replug the Astra+.
pause
