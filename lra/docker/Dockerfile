FROM nvcr.io/nvidia/pytorch:20.12-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing

RUN pip install --upgrade numpy

RUN pip install wandb
RUN pip install timm

RUN pip install einops
RUN pip install tensorboardX

RUN pip install fvcore

RUN pip install jax==0.2.4
RUN pip install flax==0.2.2
RUN pip install ml-collections==0.1.0
RUN pip install tensorboard==2.3.0
RUN pip install tensorflow==2.3.1
RUN pip install tensorflow-datasets==4.0.1