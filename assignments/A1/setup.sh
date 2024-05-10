#!/bin/bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
sudo apt-get update
sudo apt-get install -y python3-opencv
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate