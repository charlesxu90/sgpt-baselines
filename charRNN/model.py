import torch.nn as nn
from dataset import SmilesCharDictionary

sd = SmilesCharDictionary()


class CharRNN(nn.Module):
    def __init__(self, vocab_size=47, hidden_size=768, num_layers=3, dropout=0.2, padding_idx=sd.pad_idx):
        super(CharRNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, vocab_size, padding_idx=padding_idx)
        self.lstm_layer = nn.LSTM(vocab_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear_layer = nn.Linear(hidden_size, vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        # x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)  # What's the function of this
        x, hiddens = self.lstm_layer(x, hiddens)
        # x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, hiddens