@echo off
setlocal EnableExtensions EnableDelayedExpansion
title TubeMap
cd /d "%~dp0"
echo.
echo  TubeMap - laptop edition (OAK-D Pro + Astra+)
echo  ---------------------------------------------
set "DEPS_TAG=deps-1.2.0"

REM ---- 1. Find Python 3.10 - 3.12 (the Astra+ SDK has no build for 3.13+) -------------
set "PYEXE="
if exist ".venv\Scripts\python.exe" goto :have_venv
for %%V in (3.12 3.11 3.10) do (
  if not defined PYEXE (
    py -%%V -c "import sys" >nul 2>nul && set "PYEXE=py -%%V"
  )
)
if not defined PYEXE (
  for %%P in ("%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%LOCALAPPDATA%\Programs\Python\Python310\python.exe") do (
    if not defined PYEXE if exist %%P set "PYEXE=%%~P"
  )
)
if not defined PYEXE (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
  echo  Python 3.10, 3.11 or 3.12 is needed. The Astra+ camera SDK does not support newer versions.
  where winget >nul 2>nul
  if errorlevel 1 goto :nopython
  choice /C YN /M "  Install Python 3.12 for your user account now (winget)"
  if errorlevel 2 goto :nopython
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  if not defined PYEXE goto :nopython
)
echo  Using Python: !PYEXE!
if "!PYEXE:~0,3!"=="py " ( !PYEXE! -m venv .venv ) else ( "!PYEXE!" -m venv .venv )
if errorlevel 1 goto :venvfail

:have_venv
set "PY=%CD%\.venv\Scripts\python.exe"

REM ---- 2. Packages (first run only; needs internet) ----------------------------------
if not exist ".venv\%DEPS_TAG%" (
  echo  Installing packages - first run only, a few minutes...
  "%PY%" -m pip install --disable-pip-version-check -q --upgrade pip
  "%PY%" -m pip install --disable-pip-version-check -q -r requirements-windows.txt
  if errorlevel 1 goto :pipfail
  "%PY%" -m pip install --disable-pip-version-check -q --no-deps pyorbbecsdk-community==1.4.2
  if errorlevel 1 echo  WARNING: Astra+ SDK did not install. The OAK will still work.
  echo ok> ".venv\%DEPS_TAG%"
)

REM ---- 3. Bring in labels from the old Image Lab (once) ----------------------------
if exist "%USERPROFILE%\TubePositioningImageLabData\annotations" if not exist "%USERPROFILE%\.tubemap\.imagelab_imported" (
  echo  Importing your existing Image Lab labels ^(originals are not changed^)...
  "%PY%" -m tubemap import "%USERPROFILE%\TubePositioningImageLabData" && (
    if not exist "%USERPROFILE%\.tubemap" mkdir "%USERPROFILE%\.tubemap"
    echo done> "%USERPROFILE%\.tubemap\.imagelab_imported"
  )
)

REM ---- 4. Camera check, then start --------------------------------------------------
echo.
"%PY%" -m tubemap doctor
echo.
echo  Starting TubeMap at http://localhost:8088  (a browser tab will open)
echo  Keep this window open while you work. Close it, or press Ctrl+C, to stop.
echo.
"%PY%" -m tubemap serve --host 127.0.0.1 --open
echo.
echo  TubeMap stopped.
pause
exit /b 0

:nopython
echo.
echo  Install Python 3.12 from https://www.python.org/downloads/windows/
echo  (tick "Add python.exe to PATH"), then run this file again.
pause
exit /b 1
:venvfail
echo  Could not create the Python environment in "%CD%\.venv".
echo  Extract the ZIP to a normal folder such as Documents\TubeMap and try again.
pause
exit /b 1
:pipfail
echo.
echo  Package installation failed. Check the internet connection (company proxy?) and run this file again.
echo  Details are above.
pause
exit /b 1
