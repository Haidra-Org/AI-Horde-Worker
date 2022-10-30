#!/bin/bash
if [ ! -f xformers-0.0.14.dev0-cp38-cp38-linux_x86_64.whl ]; then
  wget https://github.com/ninele7/xfromers_builds/releases/download/3352937371/xformers-0.0.14.dev0-cp38-cp38-linux_x86_64.whl
fi

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
if [ ! -f "conda/envs/linux/bin/python" ]; then
 bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
fi
bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
