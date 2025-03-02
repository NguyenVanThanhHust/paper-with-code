FROM nvcr.io/nvidia/pytorch:23.12-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pytorch-lightning

WORKDIR /workspace/