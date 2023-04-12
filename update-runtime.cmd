@echo off
cd /d "%~dp0"

:Isolation
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
:We do this twice the first time to workaround a conda bug where pip is not installed correctly the first time - Henk
IF EXIST CONDA GOTO WORKAROUND_END
umamba create --no-shortcuts -r conda -n windows -f environment.yaml -y
:WORKAROUND_END
umamba create --no-shortcuts -r conda -n windows -f environment.yaml -y
umamba run -r conda -n windows python -s -m pip install -r requirements.txt
echo If there are no errors above everything should be correctly installed (If not, try running update_runtime.cmd as admin).