#!/bin/bash
set -x #echo on

# Download the model
wget https://srm-model-zoo.s3-us-west-1.amazonaws.com/cp_best.pt.tar -O checkpoint/cp_best.pt.tar

# Download the installer, install and then remove the installer
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
/bin/bash ~/miniconda.sh -b -p /opt/conda
rm ~/miniconda.sh
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Set conda path
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda activate base" >> ~/.bashrc

PS1 = '$ '
source ~/.bashrc

# Update conda
conda update -y conda 

# Create an environment
conda env create -f environment.yml

# Activate the environment
# conda activate planet-amazon

# Run the model
# python predict.py
