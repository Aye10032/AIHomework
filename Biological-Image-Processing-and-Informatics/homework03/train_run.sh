#!/bin/bash
module load anaconda/2022.10 cuda/11.6
export  PYTHONUNBUFFERED=1
source activate unet_assignment
dataset='ER'
loss='BCELoss'
optimizer='SGD'
python train.py --dataset $dataset --loss $loss --optimizer $optimizer
