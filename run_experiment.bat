@echo off
rem Starts the experiment. Double-click it, or type run_experiment.bat in any folder.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\py.ps1" -m src.experiment.run_experiment %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" pause
exit /b %RC%
