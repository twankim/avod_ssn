# base notebook, contains Jupyter and relevant tools
ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.3-42158c8

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
# ARG BASE_CONTAINER=ucsdets/datascience-notebook:2021.2-stable

# scipy/machine learning (tensorflow, pytorch)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
# ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.3-42158c8

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

# TODO: install g++, build-essential
RUN apt-get update && apt-get install \
    protobuf-compiler \
    build-essential \
    g++ -y \
    cmake \
    libboost-all-dev \
    build-essential \
    gnuplot \
    gnuplot5

# 3) install packages using notebook user
# USER jovyan

# RUN conda install -y scikit-learn
COPY requirements.txt .
RUN pip3 install --upgrade \
    pip \
    setuptools \
    --upgrade wheel \
    --upgrade protobuf \
    --upgrade tf_slim \
    tensorflow-addons

RUN pip install -r requirements.txt

RUN mkdir wavedata
COPY ./wavedata/requirements.txt ./wavedata
RUN pip3 install -r ./wavedata/requirements.txt

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]

ENV PYTHONPATH "${PYTHONPATH}:./"
ENV PYTHONPATH "${PYTHONPATH}:./avod_ssn"
ENV PYTHONPATH "${PYTHONPATH}:./avod_ssn/avod"
ENV PYTHONPATH "${PYTHONPATH}:./avod_ssn/wavedata"

USER root

COPY ./scripts/install/build_integral_image_lib.bash .
COPY ./wavedata/wavedata/tools/core/lib/src ./wavedata/wavedata/tools/core/lib/src
# RUN cmake wavedata/wavedata/tools/core/lib/src
# RUN make wavedata/wavedata/tools/core/lib
RUN sh build_integral_image_lib.bash

# COPY ./avod/protos/*.proto ./avod/protos/
# RUN protoc ./avod/protos/*.proto --python_out=.
