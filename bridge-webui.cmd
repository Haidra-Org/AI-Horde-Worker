@echo off
cd /d %~dp0
call runtime python webui.py %*
%0 %*