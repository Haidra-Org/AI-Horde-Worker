@echo off
cd /d %~dp0
call runtime python -s show_available_models.py %*