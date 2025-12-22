#!/bin/bash
#$ -M ysong25@nd.edu          #
#$ -m abe                     # a:abort, b:begin, e:end
#$ -N gmm_pipeline_full       # job name
#$ -q gpu                     #
#$ -l gpu=1                   #
#$ -l h="qa-a10*"             # Target GPU hosts
#$ -j y                       # stdout stderr
#$ -o /users/ysong25/wafer_project/logs/gmm_pipeline_full.log   # log 

# -----------------------------
cd /users/ysong25/wafer_project

source ~/.bashrc
conda activate wafer_cluster

# -----------------------------
# run Python
# -----------------------------
python gmm_pipeline_full.py
