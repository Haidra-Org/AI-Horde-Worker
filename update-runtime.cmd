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
umamba run -r conda -n windows python -s -m pip uninstall nataili -y
umamba run -r conda -n windows python -s -m pip install -r requirements.txt
umamba run -r conda -n windows python -s -m pip uninstall triton -y
umamba run -r conda -n windows python -s -m pip install --pre torch torchvision torchaudio torchtriton --extra-index-url https://download.pytorch.org/whl/nightly/cu118 --force
#export TORCH_CUDA_ARCH_LIST="8.9" # Set this according to your GPU
##The below can take quite a lot of time
umamba run -r conda -n windows python -s -m pip install ninja
umamba run -r conda -n windows python -s -m pip install -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers  
umamba run -r conda -n windows python -s -m pip install -U numba
echo If there are no errors above everything should be correctly installed (If not, try deleting the folder /conda/envs/ and try again).