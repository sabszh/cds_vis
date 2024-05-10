#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the script
python src/image_search.py "$@"

# Deactivate the virtual environment
deactivate
