#!/bin/bash

# Remove all .csv files in subdirectories
find . -type f -name "*.csv" -not -path "./*" -delete

# Remove folders named compileErrors and debug
find . -type d \( -name "compileErrors" -o -name "debug" \) -exec rm -rf {} +

# Print confirmation message
echo "All .csv files in subdirectories and folders named 'compileErrors' or 'debug' have been deleted."