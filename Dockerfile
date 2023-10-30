FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget vim git

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_23.9.0-0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash ./Miniconda3-py38_23.9.0-0-Linux-x86_64.sh -b \
    && rm -f ./Miniconda3-py38_23.9.0-0-Linux-x86_64.sh

COPY . /src 
WORKDIR /src

RUN pip install -r requirements.txt

