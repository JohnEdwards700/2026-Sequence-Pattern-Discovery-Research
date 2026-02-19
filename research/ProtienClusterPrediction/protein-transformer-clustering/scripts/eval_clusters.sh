#!/bin/bash

# This script evaluates the clustering results of the transformer model on protein sequences.

# Activate the Python environment
source venv/bin/activate

# Set the path to the evaluation script
EVAL_SCRIPT="src/evaluation/cluster_analysis.py"

# Set the path to the processed data
PROCESSED_DATA_DIR="data/processed"

# Set the output directory for evaluation results
OUTPUT_DIR="results/evaluation"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the evaluation script
python $EVAL_SCRIPT --data_dir $PROCESSED_DATA_DIR --output_dir $OUTPUT_DIR

# Deactivate the Python environment
deactivate

echo "Clustering evaluation completed. Results saved to $OUTPUT_DIR."