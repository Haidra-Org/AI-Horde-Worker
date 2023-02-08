@echo off
cd /d %~dp0
set NATAILI_CACHE_HOME=./
call runtime python bridge_stable_diffusion.py %*
%0 %*