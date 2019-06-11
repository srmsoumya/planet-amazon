Prerequisite
    - Docker
    - Docker Compose

Steps:

1. Setting up the docker environment
``` 
# Create docker-volumes
docker volume create planet-code
docker volume create planet-opt
docker volume create planet-profile

# Pull the code to get the Dockerfile & docker-compose.yml
git clone https://github.com/srm-soumya/planet-amazon.git

# Run docker-compose in detached mode and enter inside the container
cd planet-amazon/docker
docker-compose up -d
docker exec -it srm-planet /bin/bash
```

2. Inside the docker container, run the following commands to host the application
```
git clone https://github.com/srm-soumya/planet-amazon.git
cd planet-amazon
source setup.sh

# Once the setup is done, run server in detached mode, use tmux or nohup
python server.py
```

3. Browse to localhost:5000 in your local browser, to upload and test the application