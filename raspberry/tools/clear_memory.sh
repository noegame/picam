#!/bin/bash

# This script is designed to clear specific image files from all subdirectories
# within the project's 'output' folder.

# Determine the absolute path to the script's directory to ensure a reliable starting point.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Navigate up to the project's root directory from the script's location.
PROJECT_ROOT_DIR="$SCRIPT_DIR/../.."

# Define the target 'output' directory at the project root.
OUTPUT_DIR="$PROJECT_ROOT_DIR/output"

echo "This script will delete all .png, .jpg, and .jpeg files from all subfolders of the '$OUTPUT_DIR' directory."
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
echo    # Move to a new line

if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Check if the target output directory exists before proceeding.
    if [ ! -d "$OUTPUT_DIR" ]; then
      echo "Error: Directory '$OUTPUT_DIR' not found. Nothing to do."
      exit 1
    fi

    echo "Searching for and deleting image files..."
    
    # Use find to locate and delete files with specified extensions (case-insensitively).
    # The -print option is used to display the path of each file being deleted.
    find "$OUTPUT_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print -delete

    echo "Image files have been successfully deleted."
else
    echo "Operation cancelled."
fi