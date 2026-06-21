@echo off
setlocal
set "APPDIR=%~dp0"
set "PYTHONPATH=%APPDIR%;%PYTHONPATH%"
cd /d "%APPDIR%"
py -3 "%APPDIR%main.py"
if errorlevel 1 (
  echo.
  echo BungVision Label Studio exited with an error.
  pause
)
