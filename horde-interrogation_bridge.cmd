@echo off
cd /d %~dp0
call runtime python -s bridge_interrogation.py %*
%0 %*