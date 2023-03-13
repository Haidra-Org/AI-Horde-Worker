@echo off
cd /d "%~dp0"

:Isolation
SET CONDA_SHLVL=
set path=%windir%\system32;%windir%;%windir%\System32\Wbem;%windir%\System32\WindowsPowerShell\v1.0\
set appdata=%~dp0conda\windows-appdata
set pythonpath=

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
IF EXIST CONDA GOTO APP

:INSTALL
call update-runtime

:APP
call conda\condabin\activate.bat windows
%*
IF [%1] == [] TITLE Runtime Command Prompt && cmd /k