#!/bin/bash

########################################### DOWNLOAD DATASET:

# Download and extract the dataset
python -u parser.py download_and_extract_camera_primus_dataset
# Clean the dataset
python -u parser.py clean_dataset

########################################### CONVERT SEMANTIC TO MIDI:

# Path to the parent directory containing the subfolders
PARENT_DIR="Corpus"

# Iterate over each subfolder in the parent directory
for SUBFOLDER in "$PARENT_DIR"/*; do
    # Check if it's a directory
    if [ -d "$SUBFOLDER" ]; then
        # Iterate over each file in the subfolder
        for FILE in "$SUBFOLDER"/*.semantic; do
            # Check if file exists to avoid error with empty directories
            if [ -f "$FILE" ]; then
                # Create output filename by replacing .semantic with .mid
                OUTPUT_FILE="${FILE%.semantic}.mid"

                # Run the command with the current file as input and new file as output
                sh semantic_conversor.sh "$FILE" "$OUTPUT_FILE"
            fi
        done
    fi
done

########################################### CONVERT MIDI TO AUDIO:

python -u parser.py create_multimodal_dataset