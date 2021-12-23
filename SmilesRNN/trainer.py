import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import save_model
from tqdm.auto import tqdm
from dataset import SmilesCharDictionary
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class SmilesRnnTrainer:

    def __init__(self, model, n_epochs, device='cuda', save_dir=None, clip_gradients=True, batch_size=512,
                 lr=1e-3, num_workers=0) -> None:
        self.sd = SmilesCharDictionary()
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.lr = lr
        self.clip_gradients = clip_gradients
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.writer = SummaryWriter(self.save_dir)

    def fit(self, training_data, test_data=None):
        self._train(training_data, test_data)

    def get_grad_params(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    def _train(self, training_data, test_data=None):
        self.optimizer = torch.optim.Adam(self.get_grad_params(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.sd.pad_idx).to(self.device)

        best_loss = float('inf')
        for epoch in range(self.n_epochs):
            loss = self._run_epoch('train', training_data, epoch)

            if test_data is not None:
                loss = self._run_epoch('test', test_data, epoch)

            if self.save_dir is not None:
                if test_data and loss < best_loss:  # save best test loss
                    best_loss = loss
                    self._save_model(self.save_dir, str(epoch+1), loss)
                elif test_data is None and loss < best_loss:  # save best train loss
                    best_loss = loss
                    self._save_model(self.save_dir, str(epoch+1), loss)

            self.writer.add_scalar('loss', loss, epoch + 1)  # visualize only test or train loss

        if self.save_dir is not None:
            self._save_model(self.save_dir, 'final', loss)

    def _run_epoch(self, split, dataset, epoch):
        is_train = split == 'train'
        self.model.train(is_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, pin_memory=True)
        losses = []
        pbar = tqdm(enumerate(loader), desc=f"{split} (epoch {epoch})", total=len(loader)) if is_train \
            else enumerate(loader)
        for it, (x, y) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.set_grad_enabled(is_train):
                hidden = self.model.init_hidden(x.size(0), self.device)
                output, hidden = self.model(x, hidden)
                output = output.view(output.size(0) * output.size(1), -1)
                loss = self.loss_fn(output, y.view(-1))
                loss = loss.mean()  # collapse all losses if they're scattered on multiple GPUs
                losses.append(loss.item())

            if is_train:
                self.model.zero_grad()
                loss.backward()

                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {self.lr:e}")

        loss = float(np.mean(losses))
        logger.info(f"{split}, epoch: {epoch+1}, loss: {loss:.4f}")  # log both train and test loss
        return loss

    def _save_model(self, base_dir, info, loss):
        base_name = f'model_{info}_{loss:.4f}'
        logger.info(base_name)
        save_model(self.model, base_dir, base_name)
