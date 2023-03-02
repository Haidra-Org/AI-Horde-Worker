@echo off
cd /d "%~dp0"

SET CONDA_SHLVL=

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
IF EXIST CONDA GOTO APP

:INSTALL
call update-runtime

:APP
call conda\condabin\activate.bat windows
%*
IF [%1] == [] TITLE Runtime Command Prompt && cmd /k