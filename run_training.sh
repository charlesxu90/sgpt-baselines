#!/bin/bash

# CharRNN
# python charRNN/train.py --train_data /home/xiaopeng/Desktop/drug_design/sgpt/data/moses/train.csv --valid_data /home/xiaopeng/Desktop/drug_design/sgpt/data/moses/test.csv --output_dir ./moses/30-epoch --eval --n_epochs 30

# MCMG
# python MCMG/data_structs.py /home/xiaopeng/Desktop/drug_design/sgpt/data/moses/train.csv

# MolGPT

# GPT
python gpt/train.py --train_data /home/xiaopeng/Desktop/drug_design/sgpt/data/moses/train.csv --valid_data /home/xiaopeng/Desktop/drug_design/sgpt/data/moses/test.csv --output_dir ./moses/gpt-30-epoch/ --eval --n_epochs 30