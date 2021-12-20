import pandas as pd
import argparse
from utils import set_seed
import os
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
import moses
from moses.utils import get_mol
from utils import check_novelty, sample, canonic_smiles
from rdkit import Chem
from tqdm import tqdm


def sample_smiles(model, content, block_size, num_to_sample=15000, device='cuda'):
	gen_iter = math.ceil(num_to_sample / 512)

	stoi = {ch: i for i, ch in enumerate(content)}
	itos = {i: ch for i, ch in enumerate(content)}
	context = "C"

	all_dfs = []
	# for j in [0.3, 0.5, 0.7, 0.9]:
	# for j in [[0.5, 0], [0.5, 4], [0.9, 0], [0.9, 4]]:
	# for j in condition:
	# for c in [1.0, 2.0, 3.0]:
	smiles = []
	for i in tqdm(range(gen_iter)):
		x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(512, 1).to(device)
		p = None
		sca = None
		y = sample(model, x, block_size, temperature=1.6, sample=True, top_k=None)
		for gen_mol in y:
			completion = ''.join([itos[int(i)] for i in gen_mol])
			completion = completion.replace('<', '')
			smiles.append(completion)

	return smiles

def run_eval(model, content, output_dir, max_len=140):
	print(f'Generate samples...')

	smiles = sample_smiles(model, content, block_size=max_len, num_to_sample=15000)
	print(f'Evaluate on moses...')
	metrics = moses.get_all_metrics(smiles)
	print(metrics)
	# Save smiles
	df_smiles = pd.DataFrame(smiles, columns=['smiles'])
	df_smiles.to_csv(output_dir + "sampled.smiles")
	print(f'Evaluation finished!')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	# parser.add_argument('--run_name', type=str, help="name for wandb run", required=False)
	parser.add_argument('--train_data', '-t', type=str, help='Full path to SMILES file containing training data')
	parser.add_argument('--valid_data', '-v', type=str, help='Full path to SMILES file containing validation data')
	parser.add_argument('--output_dir', '-o', type=str, help='Output directory')

	parser.add_argument('--debug', action='store_true', default=False, help='debug')
	# parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold') # in moses dataset, on average, there are only 5 molecules per scaffold
	parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
	# parser.add_argument('--data_name', type=str, default='moses2', help="name of the dataset to train on",
	# 					required=False)
	parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
	parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
	parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
	parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
	parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
	parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
	parser.add_argument('--max_epochs', type=int, default = 10, help="total epochs", required=False)
	parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
	parser.add_argument('--learning_rate', type=int, default = 6e-4, help="learning rate", required=False)
	parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)
	parser.add_argument('--eval', action="store_true", help='Evaluate with moses or not, default False')

	args = parser.parse_args()

	set_seed(42)

	# wandb.init(project="lig_gpt", name=args.run_name)

	train_data = pd.read_csv(args.train_data).dropna(axis=0).reset_index(drop=True)
	val_data = pd.read_csv(args.valid_data).dropna(axis=0).reset_index(drop=True)

	# data = data.dropna(axis=0).reset_index(drop=True)
	# data.columns = data.columns.str.lower()

	# train_data = data[data['split']=='train'].reset_index(drop=True)
	# val_data = data[data['split']=='test'].reset_index(drop=True)

	smiles = train_data['SMILES']
	vsmiles = val_data['SMILES']
	
	# prop = train_data[['qed', 'logp']]
	# vprop = val_data[['qed', 'logp']]

	# prop = train_data['logp']
	# vprop = val_data['logp']

	# scaffold = train_data['scaffold_smiles']
	# vscaffold = val_data['scaffold_smiles']

	pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
	regex = re.compile(pattern)

	lens = [len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values))]
	max_len = max(lens)

	# lens = [len(regex.findall(i.strip())) for i in (list(scaffold.values) + list(vscaffold.values))]
	# scaffold_max_len = max(lens)

	smiles = [ i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles]
	vsmiles = [ i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in vsmiles]

	# scaffold = [ i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
	# vscaffold = [ i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold]

	# whole_string = ' '.join(smiles + vsmiles + scaffold + vscaffold)
	whole_string = ' '.join(smiles + vsmiles)
	whole_string = sorted(list(set(regex.findall(whole_string))))
	# print(whole_string)

	train_dataset = SmileDataset(args, smiles, whole_string, max_len, aug_prob = 0)
	valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, aug_prob = 0)

	mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props = args.num_props,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,  # scaffold = args.scaffold, scaffold_maxlen = scaffold_max_len,
	               lstm = args.lstm, lstm_layers = args.lstm_layers)
	model = GPT(mconf)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	ckpt_path = args.output_dir + 'moses_nocond_12layer.pt'
	tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
	                      lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
	                      num_workers=10, ckpt_path=ckpt_path)
	trainer = Trainer(model, train_dataset, valid_dataset, tconf)
	# trainer.train(wandb)
	trainer.train()

	run_eval(trainer.model, whole_string, output_dir=args.output_dir, max_len=max_len)
