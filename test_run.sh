#!/bin/bash

# SmilesRNN
# python SmilesRNN/train.py --train_data data/moses/data_10k.csv --valid_data data/moses/data_10k.csv --output_dir result/SmilesRNN/test_10k/ --eval --n_epochs 10 --num_to_sample 1000

# MolGPT
# python molgpt/train.py --train_data data/moses/data_10k.csv --valid_data data/moses/data_10k.csv --output_dir result/molgpt/test_10k/ --num_props 0 --max_epochs 10 --eval --num_to_sample 1000

# MCMG
# python MCMG/data_structs.py --input_file data/moses/data_10k.csv --output_dir data/moses/MCMG/
# python MCMG/train_prior.py --train-data data/moses/data_10k.csv --valid-data data/moses/data_10k.csv --voc_path data/moses/MCMG/Voc_RE --output_dir result/MCMG/test_10k/ --num_epochs 10 --eval --num_to_sample 1000

# SGPT
# python sgpt/train.py --train_data data/moses/data_10k.csv --valid_data data/moses/data_10k.csv --output_dir result/charRNN/test_10k/ --eval --n_epochs 10 --num_to_sample 1000 --n_embd 256 --n_head 16 --n_layers 8 --batch_size 512

# GPT
python gpt/train.py --train_data data/moses/data_10k.csv --valid_data data/moses/data_10k.csv --output_dir result/gpt/test_10k/ --eval --n_epochs 10 --num_to_sample 1000 --n_embd 256 --n_head 16 --n_layers 8 --batch_size 512
