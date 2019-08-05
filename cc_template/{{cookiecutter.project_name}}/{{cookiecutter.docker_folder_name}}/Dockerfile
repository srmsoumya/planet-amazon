ARG BASE_CONTAINER=ubuntu:bionic-20180526@sha256:c8c275751219dadad8fa56b3ac41ca6cb22219ff117ca98fe82b42f24e1ba64e
FROM $BASE_CONTAINER
SHELL ["/bin/bash", "-c"]

ENV LANG:C.UTF-8 LC_ALL=C.UTF-8
#<>Soumya Ranjan <se.Make>@srmsoumya"

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


# Download the model


ENV MODEL="./checkpoint/trial.jpg"

RUN mkdir checkpoint && \
    if [ -f ${MODEL} ] ; then echo "${MODEL} exist, skipping download." ;\
    else \
    wget https://akm-img-a-in.tosshub.com/sites/all/themes/itg/images/itg_image370x208.jpg ;fi;


RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
       /bin/bash ~/miniconda.sh -b -p /opt/conda && \
       rm ~/miniconda.sh && \
       ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
       echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
       echo "conda activate base" >> ~/.bashrc && \
       source ~/.bashrc && \
       conda update -y conda


ENV PS1='$ '
#RUN source ~/.bashrc

# Update conda
#RUN conda update -y conda

# Create an environment
RUN conda env create -f environment.yml
RUN echo "conda activate <project-name>" >> ~/.bashrc
RUN source ~/.bashrc
