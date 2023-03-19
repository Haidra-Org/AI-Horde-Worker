@echo off
cd /d %~dp0
set NATAILI_CACHE_HOME=./
call runtime python -s bridge_stable_diffusion.py %*
IF %ERRORLEVEL% EQU 0 (
    EXIT /B
)
%0 %*