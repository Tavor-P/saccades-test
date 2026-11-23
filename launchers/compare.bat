@echo off
rem Compares participants (1 vs 1, 1 vs group, group vs group). Asks for set A and set B if not given, then keeps the window open.
rem A set is comma-separated IDs (P01,P02), group:TBI (labels: scripts\py.ps1 -m src.experiment.participants list) or all.
set "SETA=%~1"
set "SETB=%~2"
if not defined SETA set /p "SETA=Set A (IDs separated by commas, or group:TBI): "
if not defined SETB set /p "SETB=Set B (IDs separated by commas, or group:control): "
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\py.ps1" -m src.experiment.compare "%SETA%" "%SETB%" %3 %4 %5 %6 %7 %8 %9
set RC=%ERRORLEVEL%
pause
exit /b %RC%
