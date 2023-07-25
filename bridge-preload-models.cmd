@echo off
cd /d %~dp0
call runtime python -s preload_models.py %*
