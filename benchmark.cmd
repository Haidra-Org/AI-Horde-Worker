@echo off
cd /d %~dp0
call runtime python -m hordelib.benchmark
