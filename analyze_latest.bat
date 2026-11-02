@echo off
rem Analyses the newest run (or the one named as an argument), then keeps the window open.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\py.ps1" -m src.experiment.analyze_session %*
set RC=%ERRORLEVEL%
pause
exit /b %RC%
