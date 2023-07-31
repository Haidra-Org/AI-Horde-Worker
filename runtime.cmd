@echo off
cd /d "%~dp0"

:Isolation
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET MAMBA_ROOT_PREFIX=%~dp0conda

echo %MAMBA_ROOT_PREFIX%

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
IF EXIST CONDA GOTO APP

:INSTALL
call update-runtime

:APP
umamba.exe shell hook -s cmd.exe -p "%MAMBA_ROOT_PREFIX%" -v
call "%MAMBA_ROOT_PREFIX%\condabin\mamba_hook.bat"
call "%MAMBA_ROOT_PREFIX%\condabin\micromamba.bat" activate windows
%*
call micromamba deactivate
