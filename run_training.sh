#!/bin/bash

# SmilesRNN
# python SmilesRNN/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/SmilesRNN/80_epochs/ --eval --n_epochs 80

# MCMG
# python MCMG/data_structs.py --input_file data/moses/train.csv --output_dir data/moses/MCMG/
# python MCMG/train_prior.py --train-data data/moses/train.csv --valid-data data/moses/test.csv --voc_path data/moses/MCMG/Voc_RE --output_dir result/MCMG/10_epochs/ --num_epochs 10

# MolGPT
# python molgpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/molgpt/10_epochs/ --num_props 0 --max_epochs 10 --eval

# SGPT
# python sgpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/sgpt/10_epochs/ --eval --n_epochs 10

# GPT
# python gpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/gpt/20_epochs_16h/ --eval --n_epochs 20 --n_embd 256 --n_head 16 --n_layers 8 --batch_size 512
# python gpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/gpt/20_epochs_8h/  --eval --n_epochs 20 --n_embd 256 --n_head 8 --n_layers 8 --batch_size 512
# python gpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/gpt/20_epochs_4h/ --eval --n_epochs 20 --n_embd 256 --n_head 4 --n_layers 8 --batch_size 512
# python gpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/gpt/20_epochs_16h_1024b/ --eval --n_epochs 20 --n_embd 256 --n_head 16 --n_layers 8 --batch_size 1024
# python gpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/gpt/20_epochs_4h_1024b/ --eval --n_epochs 20 --n_embd 256 --n_head 4 --n_layers 8 --batch_size 1024