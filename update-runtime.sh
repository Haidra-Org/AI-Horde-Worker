#!/bin/bash

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
if [ ! -f "conda/envs/linux/bin/python" ]; then
 bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
fi
bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
bin/micromamba run -r conda -n linux python -s -m pip uninstall nataili -y
bin/micromamba run -r conda -n linux python -s -m pip install -r requirements.txt
bin/micromamba run -r conda -n linux python -s -m pip uninstall triton -y
bin/micromamba run -r conda -n linux python -s -m pip install --pre torch torchvision torchaudio torchtriton --extra-index-url https://download.pytorch.org/whl/nightly/cu118 --force
#export TORCH_CUDA_ARCH_LIST="8.9" # Set this according to your GPU
##The below can take quite a lot of time
bin/micromamba run -r conda -n linux python -s -m pip install ninja
bin/micromamba run -r conda -n linux python -s -m pip install -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers  
bin/micromamba run -r conda -n linux python -s -m pip install -U numba