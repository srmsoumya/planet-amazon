ARG BASE_CONTAINER=ubuntu:bionic-20180526@sha256:c8c275751219dadad8fa56b3ac41ca6cb22219ff117ca98fe82b42f24e1ba64e
FROM $BASE_CONTAINER

ENV LANG:C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install necessary packages
RUN apt-get update --fix-missing && apt-get install -y \
    wget bzip2 ca-certificates \
    htop tmux unzip tree \
    libglib2.0-0 libxext6 libsm6 libxrender1 libgl1-mesa-glx \
    git && \
    apt-get clean


# Download Anaconda and store in bashrc
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Download the model
ENV MODEL="./checkpoint/trial.jpg"

RUN mkdir checkpoint && \
    if [ -f ${MODEL} ] ; then echo "${MODEL} exist, skipping download." ;\
    else \
    echo "Downloading model ... " ;fi;


# Each Docker container is a PID namespace, which means that the processes in your container are isolated from other processes on your host. A PID namespace is a tree, which starts at PID 1, which is commonly called init.
# In other words, someone has to clean up after "irresponsible" parents that leave their children un-waited, and that's PID 1's job. That's what Tini does.

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN mkdir -p /home/ubuntu
ENV HOME=/home/ubuntu
VOLUME $HOME
WORKDIR $HOME

# For creating your environment.yml file, use conda env export --no-build > environment.yml

ADD environment.yml $HOME/.

# 'temp' is my environment's name
RUN conda env create -f $HOME/environment.yml && \
    echo "conda activate temp" >> ~/.bashrc

EXPOSE 5000

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
