@echo off
rem Writes data\study_summary.csv (one row per participant, for SPSS / R / Excel) and data\study_summary.txt, then keeps the window open.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\py.ps1" -m src.experiment.study_summary %*
set RC=%ERRORLEVEL%
pause
exit /b %RC%
