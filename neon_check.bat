@echo off
rem Checks the connection to the Neon phone (no window, no participant data) and saves neon_check_report.txt here.
rem Optional argument: the phone address, e.g. neon_check.bat 192.168.43.1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\py.ps1" -m src.eye_tracking.neon_check %*
set RC=%ERRORLEVEL%
pause
exit /b %RC%
