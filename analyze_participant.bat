@echo off
rem Pools all runs of one participant into one suppression index (asks for the participant ID if none is given), then keeps the window open.
set "PID=%~1"
if not defined PID set /p "PID=Participant ID: "
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\py.ps1" -m src.experiment.analyze_participant "%PID%" %2 %3 %4 %5 %6 %7 %8 %9
set RC=%ERRORLEVEL%
pause
exit /b %RC%
