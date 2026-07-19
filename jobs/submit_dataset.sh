#!/bin/bash
cd "$(dirname "$0")/.."

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

qsub -l select=1:ncpus=96:mem=70gb:host=aluf3 \
     -v SEED=42,NUM_SAMPLES=1500,OUTPUT_DIR=data/bistable/dataset_0,RUN_LOCATION=data/bistable/instances_0 \
     jobs/run_dataset.pbs

qsub -l select=1:ncpus=96:mem=70gb:host=aluf5 \
     -v SEED=43,NUM_SAMPLES=1500,OUTPUT_DIR=data/bistable/dataset_1,RUN_LOCATION=data/bistable/instances_1 \
     jobs/run_dataset.pbs
