@echo off
cd /d %~dp0
call runtime python bridge_stable_diffusion.py %*
%0 %*