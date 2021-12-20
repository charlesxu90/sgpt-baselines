import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class SGPTTrainer:

    def __init__(self, model, batch_size=64, lr=1e-3, step_size=10, gamma=0.5, n_epochs=10, save_dir=None, save_step=10,
                 device='cpu'):
        self.model = model.to(device)
        self.n_batch = batch_size
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.train_epochs = n_epochs
        self.save_dir = save_dir
        self.save_frequency = save_step
        self.device = device
        self.n_workers = 0

    def _run_epoch(self, tqdm_data, criterion, optimizer=None):
        model = self.model
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix = {'loss': 0, 'running_loss': 0}

        for i, (prevs, nexts) in enumerate(tqdm_data):
            prevs = prevs.to(self.device)
            nexts = nexts.to(self.device)
            lens = torch.tensor([len(t) for t in prevs], dtype=torch.long, device='cpu')

            outputs, _, _ = model(prevs, lens)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            postfix['loss'] = loss.item()
            postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)
            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizer is None else 'Train'
        return postfix

    def _train(self, train_loader, val_loader=None):
        def get_params():
            return (p for p in self.model.parameters() if p.requires_grad)

        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.step_size, self.gamma)

        model.zero_grad()
        for epoch in range(self.train_epochs):
            scheduler.step()

            tqdm_data = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch))
            postfix = self._run_epoch(tqdm_data, criterion, optimizer)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                postfix = self._run_epoch(tqdm_data, criterion)

            if (self.save_dir is not None) and (epoch % self.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(), self.save_dir + 'lstm_{0:03d}.pt'.format(epoch))
                model = model.to(self.device)

    def fit(self, train_data, val_data=None):
        print(f"Model device: {self.model.device}")
        train_loader = DataLoader(train_data, batch_size=self.n_batch, shuffle=True, num_workers=self.n_workers,
                                  collate_fn=None)
        val_loader = DataLoader(val_data, batch_size=self.n_batch, shuffle=True, num_workers=self.n_workers,
                                collate_fn=None)

        self._train(train_loader, val_loader)
