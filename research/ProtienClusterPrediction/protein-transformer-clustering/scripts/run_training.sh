#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Set the configuration file
CONFIG_FILE="configs/default.yaml"

# Run the training script
python src/training/train.py --config $CONFIG_FILE

# Deactivate the virtual environment
deactivate