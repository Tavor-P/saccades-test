@echo off
rem Runs the test suite (extra arguments go to pytest), then keeps the window open.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\py.ps1" -m pytest -q %*
set RC=%ERRORLEVEL%
pause
exit /b %RC%
