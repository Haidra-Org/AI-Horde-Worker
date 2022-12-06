FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
# Base scripts
RUN apt-get update --fix-missing
RUN apt install -y python3 python3-dev python3-pip
RUN apt install -y --no-install-recommends git nano wget curl
# Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install dependencies
RUN pip3 install diffusers transformers==4.21.3 accelerate
RUN pip3 install tqdm timm ftfy ray pynvml basicsr gfpgan realesrgan
RUN pip3 install omegaconf fairscale tensorboard python-slugify piexif
RUN pip3 install einops facexlib opencv-python-headless open-clip-torch clip
RUN pip3 install loguru pytorch-lightning==1.7.7 GitPython

RUN mkdir /nataili
WORKDIR /nataili
ADD worker /nataili/worker
ADD ./configs /nataili/configs
ADD ./data /nataili/data
RUN mkdir /nataili/models
ADD ./ldm /nataili/ldm
ADD ./nataili /nataili/nataili
ADD ./*.py /nataili/
ADD ./*.json /nataili/
ADD ./*.png /nataili/
ADD ./*.sh /nataili/
ADD ./*.txt /nataili/
ADD ./*.yaml /nataili/

# Install nataili
RUN pip3 install -e .
