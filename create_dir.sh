#!/bin/bash

mkdir -p {checkpoint,data,docker,nbs,python_scripts,static,templates,static/css/}

touch {predict.py,server.py,templates/home.html,templates/predict.html,templates/error.html,docker/docker-compose.yml,docker/Dockerfile}

cat <<EOT>> predict.py
import sys
sys.path.append('./python_scripts/')

def load_model(*args):
    '''Load and return the model from the checkpoint/ dir.'''

    pass

def load_data(input):
    '''Perform the required pre-processing on the given input required for the model and return.'''

    pass

def predict(model, input):
    '''Pass the input to the model and return the result.'''

    pass

EOT


cat <<EOT>> server.py

from predict import load_model, load_data, predict

@app.route('/', methods=['GET', 'POST'])
def home():
    '''Create your API'''
    pass

EOT

cat <<EOT>> docker/Dockerfile

ARG BASE_CONTAINER=ubuntu:bionic-20180526@sha256:c8c275751219dadad8fa56b3ac41ca6cb22219ff117ca98fe82b42f24e1ba64e
FROM $BASE_CONTAINER
SHELL ["/bin/bash", "-c"]

ENV LANG:C.UTF-8 LC_ALL=C.UTF-8
<>Soumya Ranjan <se.Make>@srmsoumya"

RUN apt-get update --fix-missing && apt-get install -y \
    wget bzip2 ca-certificates \
    htop tmux unzip tree \
    libglib2.0-0 libxext6 libsm6 libxrender1 libgl1-mesa-glx \
    git && \
    apt-get clean

RUN mkdir -p /home/ubuntu
ENV HOME=/home/ubuntu
VOLUME $HOME
WORKDIR $HOME

EXPOSE 5000

ENTRYPOINT [ "/bin/bash" ]

EOT

cat <<EOT>> docker/docker-compose.yml
version: '3'
services:
    planet:
        build: .
        container_name: <container-name>
        ports:
            - 5000:5000
        tty: true
        volumes:
            - <project-name>-code:/home/ubuntu
            - <project-name>-opt:/opt
            - <project-name>-profile:/etc/profile.d/
volumes:
    <project-name>-code:
    <project-name>-opt:
    <project-name>-profile:

EOT
