import logging
import argparse
import json
import os
import pandas as pd
from model import SmilesRnn
from trainer import SmilesRnnTrainer
from dataset import load_smiles_from_list, get_tensor_dataset, SmilesCharDictionary
from sampler import sample
import moses
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger.addHandler(logging.NullHandler())

sd = SmilesCharDictionary()


def train(train_data, val_data, save_dir, device='cuda', max_len=140, n_epochs=10, lr=1e-3, hidden_size=512, n_layers=3,
          batch_size=64, rnn_dropout=0.2, print_every=100, valid_every=100):
    logger.info('Training...')

    train_seqs, _ = load_smiles_from_list(train_data, max_len=max_len)
    valid_seqs, _ = load_smiles_from_list(val_data, max_len=max_len)

    train_set = get_tensor_dataset(train_seqs)
    test_set = get_tensor_dataset(valid_seqs)

    model = SmilesRnn(input_size=sd.get_char_num(), hidden_size=hidden_size, output_size=sd.get_char_num(),
                      n_layers=n_layers, rnn_dropout=rnn_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=sd.pad_idx)
    trainer = SmilesRnnTrainer(model, [criterion], optimizer, save_dir=save_dir, device=device)
    trainer.fit(train_set, test_set,
                batch_size=batch_size, print_every=print_every, valid_every=valid_every, n_epochs=n_epochs)
    return model


def run_eval(model, output_dir, max_len=140, num_to_sample=15000):
    logger.info(f'Generate samples...')
    smiles = sample(model, num_to_sample=num_to_sample, device='cuda', batch_size=64, max_seq_length=max_len)
    logger.info(f'Evaluate on moses...')
    metrics = moses.get_all_metrics(smiles)
    logger.info(metrics)
    # Save smiles
    df_smiles = pd.DataFrame(smiles, columns=['smiles'])
    df_smiles.to_csv(output_dir + "sampled.smiles")
    logger.info(f'Evaluation finished!')


def main(args):
    df_train = pd.read_csv(args.train_data)
    df_valid = pd.read_csv(args.valid_data)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.output_dir + 'commandline_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger.info(f"Training prior model started, the results are saved in {args.output_dir}")
    model = train(df_train.SMILES.tolist(), df_valid.SMILES.tolist(), save_dir=args.output_dir, n_epochs=args.n_epochs)
    logger.info(f'Training done, the trained model is in {args.output_dir}')
    if args.eval:
        run_eval(model, args.output_dir, max_len=args.max_len)


def parse_args():
    parser = argparse.ArgumentParser(description='Distribution learning benchmark for SMILES RNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_data', '-t', type=str, help='Full path to SMILES file containing training data')
    parser.add_argument('--valid_data', '-v', type=str, help='Full path to SMILES file containing validation data')
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--n_epochs', default=10, type=int, help='Number of training epochs')
    optional.add_argument('--lr', default=1e-3, type=float, help='RNN learning rate')
    optional.add_argument('--n_layers', default=8, type=int, help='Number of layers for training')
    optional.add_argument('--batch_size', default=512, type=int, help='Size of a mini-batch for gradient descent')
    optional.add_argument('--n_embd', default=512, type=int, help='Number of embeddings for GPT model')
    optional.add_argument('--n_head', default=8, type=int, help='Number of attention heads for GPT model')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--max_len', default=140, type=int, help='Max length of a SMILES string')
    optional.add_argument('--eval', action="store_true", help='Evaluate with moses or not, default False')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
