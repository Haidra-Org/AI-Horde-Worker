FROM ubuntu:22.04

RUN mkdir /worker

WORKDIR /worker

COPY . .

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        bzip2 \
        wget \
      && ./update-runtime.sh \
      && rm -rf /var/lib/apt/lists/* \
      && bin/micromamba run -r conda -n linux python -s -m pip cache purge

ENTRYPOINT [ "/worker/docker_entrypoint.sh" ]
