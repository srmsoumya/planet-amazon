# Deep Learning Workflow to productionize a model

1. [Initial project setup](#initial-project-setup)
2. [Train your model and save the checkpoints](#train-your-model-and-save-the-checkpoints)
3. [Create `predict.py` file](#create-predict.py-file)
4. [Expose your model using Flask or (Gramex: our preferred data-server)](#expose-your-model-using-Flask)
5. [Create a Docker and Docker Compose file](#Create-a-Docker-and-Docker-Compose-file)
6. [Create `setup.sh` script, to create the required environment to host your model](#Create-setup.sh-script,-to-create-the-required-environment-to-host-your-model)
7. [Push your code into github or bitbucket](#Push-your-code-into-github-or-bitbucket)
8. [Host your application in the cloud](#Host-your-application-in-the-cloud)
9. [Cookie Cutter Usage](#Cookie-Cutter-Usage)

---------------------------------------------------------------------------------------------------

## Initial project setup

1.1 Create a conda environment and install the required packages, eg: pytorch
```
conda create -y -n <project-name> python=3
conda activate <project-name>
conda install pytorch-cpu torchvision-cpu -c pytorch
```

1.2 Recommended directory structure to start your project
```
<project-name>\
├── checkpoint\
├── data\
├── docker\
├── nbs
│   └── experiment with your code here
├── python-scripts
```


## Train your model and save the checkpoints

Use your favourite framework to train your model. Store the model checkpoints inside the `checkpoint` directory.


## Create `predict.py` file

Define 3 functions in here:
```
def load_model(*args):
    '''Load and return the model from the checkpoint/ dir.'''
    pass

def load_data(input):
    '''Perform the required pre-processing on the given input required for the model and return.'''
    pass

def predict(model, input):
    '''Pass the input to the model and return the result.'''
    pass
```


## Expose your model using Flask
eg: Flask server  

4.1 Create `server.py` inside the <project-name> dir and add two folders to hold your html templates and static files 
```
<project-name>\
├── static
│   └── css
└── templates
│   └── home.html
│   └── predict.html
│   └── error.html etc
│   server.py
```

4.2 Create your @routes

Import the `load_model`, `load_data` & `predict` fn inside 'server.py'.  
Create your REST API.

```
from predict import load_model, load_data, predict

@app.route('/', methods=['GET', 'POST'])
def home():
    '''Create your API'''
    pass
```


## Create a Docker and Docker Compose file

5.1 Directory structure
```
docker/
├── docker-compose.yml
└── Dockerfile
```

5.2
**Dockerfile**
```
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
```

5.3
**docker-compose.yml**  
- Give a suitable name to your container <container-name>.  
- Make sure you expose the port where you wish to deploy your model, in our case port: 5000.  
- Give suitable names to your data volumes <project-name>, which we will create in our next steps.
```
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
```


## Create `setup.sh` script, to create the required environment to host your model.

6.1 Host your model on a storage account to which you have required access.  
eg: [Steps](https://docs.aws.amazon.com/AmazonS3/latest/user-guide/upload-download-objects.html) to upload your model in Amazon S3 Storage.

6.2 Export your conda environment into `environment.yml` file.
- Make sure you are inside the project conda environment.
```
conda env export > environment.yml
```

6.3
**setup.sh**
```
#!/bin/bash

# Download the model
MODEL=./checkpoint/cp_best.pt.tar
if [ -f "$MODEL" ]; then
    echo "$MODEL exist, skipping download."
else
    wget <path-to-model-cloud-storage> -O checkpoint/<model-name>
fi

# Download the installer, install and then remove the installer
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
/bin/bash ~/miniconda.sh -b -p /opt/conda
rm ~/miniconda.sh
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Set conda path
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

PS1='$ '
source ~/.bashrc

# Update conda
conda update -y conda 

# Create an environment
conda env create -f environment.yml
echo "conda activate <project-name>" >> ~/.bashrc
source ~/.bashrc
```


## Push your code into github or bitbucket


## Host your application in the cloud.

8.1 We will be using Digital Ocean. [Steps](https://www.digitalocean.com/docs/droplets/how-to/create/) to create an ubuntu instance in Digital-Ocean.

8.2 SSH into your server
```
ssh -L5000:localhost:5000 root@IP
```

8.3 Install [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [Docker Compose](https://docs.docker.com/compose/install/).

8.4. Setting up the docker environment
``` 
# Create docker-volumes
docker volume create <project-name>-code
docker volume create <project-name>-opt
docker volume create <project-name>-profile

# Pull the code to get the Dockerfile & docker-compose.yml
git clone <path-to-project>

# Run docker-compose in detached mode and enter inside the container
cd <project-name>/docker
docker-compose up -d
docker exec -it <container-name> /bin/bash
```

8.5. Inside the docker container, run the following commands to host the application
```
git clone <path-to-project>
cd <project-name>
source setup.sh

# Once the setup is done, run server in detached mode, use tmux or nohup
python server.py
```

8.6. Browse to localhost:5000 in your local browser, to upload and test the application

## Cookie Cutter Usage (For Dev)

```
cookiecutter CookieCutter_ProDL/
```
