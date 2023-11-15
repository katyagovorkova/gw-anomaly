#!/bin/bash

# Define the directory containing the files
FILES="/n/home00/emoreno/gw-anomaly/output/O3av2/1243382418_1248652818/*"

# Initialize a counter
counter=0

# Loop through each file and submit a job
for f in $FILES
do
  # Submit the job
  sbatch --export=FILE="$f" run_single_job.sh
  echo "File: $f"
  echo "Counter: $counter"

  # Increment the counter
  ((counter++))

  # Break the loop if the counter reaches 2
  if [ $counter -eq 800 ]; then
    break
  fi
done

