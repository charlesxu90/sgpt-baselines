#!/bin/bash
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --job-name=baseline
#SBATCH --output=./%x-%j-slurm.out
#SBATCH --error=./%x-%j-slurm.err

# CharRNN
# python charRNN/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/charRNN/10_epochs/ --eval --n_epochs 10

# MCMG
# python MCMG/data_structs.py --input_file data/moses/train.csv --output_dir data/moses/MCMG/
# python MCMG/train_prior.py --train-data data/moses/train.csv --valid-data data/moses/test.csv --voc_path data/moses/MCMG/Voc_RE --output_dir result/MCMG/10_epochs/ --num_epochs 10

# MolGPT
# python molgpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/molgpt/10_epochs/ --num_props 0 --max_epochs 10 --eval

# GPT
# python gpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/gpt/10_epochs/ --eval --n_epochs 10

# SGPT
# python sgpt/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/sgpt/10_epochs/ --eval --n_epochs 10

# SmilesRNN
python SmilesRNN/train.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --output_dir result/SmilesRNN/10_epochs/ --eval --n_epochs 10