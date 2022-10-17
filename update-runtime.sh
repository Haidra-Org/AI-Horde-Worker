#!/bin/bash
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
