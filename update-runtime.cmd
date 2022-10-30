@echo off
cd /d %~dp0

IF not exist "xformers-0.0.14.dev0-cp38-cp38-win_amd64.whl" (
    curl -L -o "xformers-0.0.14.dev0-cp38-cp38-win_amd64.whl" "https://github.com/ninele7/xfromers_builds/releases/download/3352937371/xformers-0.0.14.dev0-cp38-cp38-win_amd64.whl"
)

SET CONDA_SHLVL=

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
umamba create --no-shortcuts -r conda -n windows -f environment.yaml -y
echo If there are no errors above everything should be correctly installed (If not, try running update_runtime.cmd as admin).
