FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget vim git install ffmpeg libsm6 libxext6  -y

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_23.9.0-0-Linux-x86_64.sh \
    && mkdir -p /root/.conda \
    && bash ./Miniconda3-py38_23.9.0-0-Linux-x86_64.sh -b \
    && rm -f ./Miniconda3-py38_23.9.0-0-Linux-x86_64.sh

COPY . /src 
WORKDIR /src

#RUN conda install -y matplotlib=3.0.3 numpy=1.16.2 pillow=5.4.1 pandas tqdm=4.32.2 -c conda-forge
RUN pip3 install wilds pytorch_transformers jupyter 
#RUN pip install wilds matplotlib==3.0.3 numpy==1.19.1 pandas pillow==5.4.1

