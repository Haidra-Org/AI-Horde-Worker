@echo off
cd /d %~dp0

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
umamba create --no-shortcuts -r conda -n windows -f environment.yaml -y
echo If there are no errors above everything should be correctly installed (If not, try running update_runtime.cmd as admin).
