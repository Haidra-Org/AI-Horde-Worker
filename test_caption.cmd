@echo off
cd /d %~dp0
call runtime python -m pip list
call runtime python test.py %*
%0 %*