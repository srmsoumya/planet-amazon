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
