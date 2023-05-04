FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        wget

RUN mkdir /worker

WORKDIR /worker

COPY . .

RUN ./update-runtime.sh

ENTRYPOINT [ "/worker/docker_entrypoint.sh" ]
