@echo off
cd /d %~dp0
call runtime python -s bridge_scribe.py %*
IF %ERRORLEVEL% EQU 0 (
    EXIT /B
)
%0 %*