#!/bin/bash
#$ -m abe
#$ -M ysong25@nd.edu
#$ -N Dino_Cluster_L2
#$ -q gpu
#$ -l gpu=1
#$ -l h="qa-a10*"
#$ -pe smp 8
#$ -j y
#$ -o logs/cluster_l2.log

cd /users/ysong25/wafer_project

conda activate dino311

python cluster_train_l2_full.py
