#!/bin/bash


# CharRNN
# python charRNN/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/charRNN/30_epochs/ --eval --n_epochs 30

# MCMG
python MCMG/data_structs.py --input_file data/moses/train.csv --output_dir data/moses/MCMG/
# python MCMG/train_prior.py --train-data data/moses/train.csv --valid-data data/moses/test.csv --voc_path data/moses/MCMG/Voc_RE --output_dir result/MCMG/30_epochs/ --num_epochs 30

# MolGPT
# python molgpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/molgpt/30_epochs/ --num_props 0 --max_epochs 30 --eval

# GPT
# python gpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/gpt/30_epochs/ --eval --n_epochs 30

# SGPT
# python sgpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/sgpt/30_epochs/ --eval --n_epochs 30
