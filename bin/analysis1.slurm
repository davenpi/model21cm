#!/bin/bash

#SBATCH -J analysis1 ##Job name
#SBATCH -o analysis1.out ##Write stdout to this file
#SBATCH -e analysis1.err ##Write error messages to this file
#SBATCH --open-mode=append ##Not sure
#SBATCH -p serial_requeue ##Use the 'test' partition 
#SBATCH -t 0-03:00 ##Run up to 0 days, 2 hours, 0 minutes
#SBATCH -n 8 ##Number of cores 
#SBATCH --mem=4000 ##Use no more than 2Gb RAM 
#SBATCH -N 1 ##Use no more than 1 compute node

## SETUP STORAGE ##
STORAGE_DIR="/n/scratchlfs/phys201/users/$USER/model21cm.${SLURM_JOB_ID}"

echo "STORAGE_DIR: $STORAGE_DIR"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_STEP_NUM_TASKS: $SLURM_STEP_NUM_TASKS"

export STORAGE_DIR
export SLURM_STEP_NUM_TASKS
mkdir $STORAGE_DIR

## COPY INPUT FILES AND MOVE TO WORKING DIR ##

## LOAD REQUIRED MODULES/ENVIRONMENT ##
module load python 
source activate model21cm 

## RUN PROGRAM ##
python "$SLURM_SUBMIT_DIR/analysis1.py" >> analysis1.out
##output=analysis1_output.csv

## COPY OUTPUT TO SECURE STORAGE ##
##cp $output $SLURM_SUBMIT_DIR/$output
