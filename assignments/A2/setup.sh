#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
sudo apt-get install -y python3-opencv
pip install -r requirements.txt

# Deactivate virtual environment
deactivate