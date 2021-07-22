FROM nvcr.io/nvidia/pytorch:20.12-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing

RUN pip install --upgrade numpy

RUN pip install wandb
RUN pip install timm

RUN pip install einops
RUN pip install tensorboardX
