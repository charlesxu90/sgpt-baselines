#!/usr/bin/env python
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_structs import MolData, Vocabulary
from model_MCMG import transformer_RL
from torch.optim import Adam
from Optim import ScheduledOptim
from pytorchtools import EarlyStopping
import os
import moses
from utils import seq_to_smiles
import pandas as pd


def train(train_data, valid_data, model, optim, num_epochs, save_prior_path):
    model.decodertf.to(device)
    model.decodertf.train()
    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    early_stopping = EarlyStopping(patience=5, verbose=False)

    for epoch in range(num_epochs):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        total_loss = 0
        for step, batch in tqdm(enumerate(train_data), total=len(train_data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss, each_molecule_loss is the loss of  each molecule

            loss, each_molecule_loss = model.likelihood(seqs)
            # loss = - log_p.mean()

            # Calculate gradients and take a step
            optim.zero_grad()
            loss.backward()
            optim.step_and_update_lr()
            # print(loss)

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))
            # if step % print_every == print_every - 1:

            if step % 200 == 0 and step != 0:
                # decrease_learning_rate(optim, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data))

        print('average epoch loss:', total_loss / len(train_data))
        val_loss = validate(valid_data, model)
        val_losses.append((total_step, val_loss))

        early_stopping(val_loss, model.decodertf, 'RE1_Prior')

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Save the Prior
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model.decodertf.state_dict(), save_prior_path)
        print(f'Val Loss: {val_loss}')
    return model, train_losses, val_losses


def validate(valid_data, model):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.decodertf.to(device)
    model.decodertf.eval()
    total_loss = 0

    for step, batch in tqdm(enumerate(valid_data), total=len(valid_data)):
        with torch.no_grad():
            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss, each_molecule_loss is the loss of  each molecule
            loss, each_molecule_loss = model.likelihood(seqs)
            # loss = - log_p.mean()

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))
    return total_loss / len(valid_data)


def sample(model, voc, batch_size=128, n_steps=5000):
    smiles_list = []
    token_list = ['is_DRD2', 'high_QED', 'good_SA']
    for i in range(n_steps):
        seqs = model.generate(batch_size, max_length=140, con_token_list=token_list)
        smiles = seq_to_smiles(seqs, voc)
        smiles_list.extend(smiles)
        print('step: ', i)
    return smiles_list


def run_eval(model, output_dir, voc, max_len=140):
    print(f'Generate samples...')
    smiles = sample(model, voc, batch_size=max_len, n_steps=50)
    print(f'Evaluate on moses...')
    metrics = moses.get_all_metrics(smiles)
    print(metrics)
    # Save smiles
    df_smiles = pd.DataFrame(smiles, columns=['smiles'])
    df_smiles.to_csv(output_dir + "sampled.smiles")
    print(f'Evaluation finished!')

def main(train_data, valid_data, voc_path, output_dir, num_epochs=600):

    """Trains the Prior decodertf"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_path)

    # Create a Dataset from a SMILES file
    moldata = MolData(train_data, voc)
    valid = MolData(valid_data, voc)

    train_data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    valid_data = DataLoader(valid, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = transformer_RL(voc, d_model, nhead, num_decoder_layers,
                           dim_feedforward, max_seq_length,
                           pos_dropout, trans_dropout)

    optim = ScheduledOptim(
        Adam(Prior.decodertf.parameters(), betas=(0.9, 0.98), eps=1e-09),
        d_model * 8,n_warmup_steps)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_prior_path = output_dir + "prior.ckpt"
    model, train_losses, val_losses = train(train_data, valid_data, Prior, optim, num_epochs, save_prior_path)
    run_eval(model, output_dir, voc, max_len=140)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    max_seq_length = 140
    # num_tokens=71
    # vocab_size=71
    d_model = 128
    # num_encoder_layers = 6
    num_decoder_layers = 12
    dim_feedforward = 512
    nhead = 8
    pos_dropout = 0.1
    trans_dropout = 0.1
    n_warmup_steps = 500
    batch_size = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--train-data', action='store')
    parser.add_argument('--valid-data', action='store')

    parser.add_argument('--voc_path', action='store', help='Path to vocabulary file.')
    parser.add_argument('--output_dir', action='store', default='./result/', help='Dir to save results.')
    parser.add_argument('--num_epochs', type=int, default=600, help='Num epochs')

    arg_dict = vars(parser.parse_args())

    main(**arg_dict)
