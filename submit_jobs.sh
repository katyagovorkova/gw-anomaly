#!/bin/bash

# Get the list of files in the directory
FILES=/n/home00/emoreno/gw-anomaly/output/O3av2/1243382418_1248652818/*

# Loop through each file and submit a job
for f in $FILES
do
  sbatch --export=FILE="$f" run_single_job.sh
  break
done