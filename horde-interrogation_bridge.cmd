@echo off
cd /d %~dp0
call runtime python bridge_interrogation.py %*
%0 %*