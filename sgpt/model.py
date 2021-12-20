import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset import SmilesCharDictionary

logger = logging.getLogger(__name__)
sd = SmilesCharDictionary()

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, block_size=142, n_embd=512, n_head=8, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, block_size=142, n_embd=512, n_head=8, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(block_size=block_size, n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop,
                                        resid_pdrop=resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SGPT(nn.Module):
    def __init__(self, vocab_size=47, n_embd=512, lstm_layers=3, dropout=0.2, padding_idx=sd.pad_idx,
                 attn_layers=2):
        super(SGPT, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, vocab_size, padding_idx=padding_idx)
        self.lstm_layer = nn.LSTM(vocab_size, n_embd, lstm_layers, dropout=dropout, batch_first=True)
        self.linear_layer = nn.Linear(n_embd, vocab_size)
        # transformer
        self.attn_layers = nn.Sequential(*[Block(n_embd=512) for _ in range(attn_layers)])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        # x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)  # What's the function of this
        x, hiddens = self.lstm_layer(x, hiddens)
        x = self.attn_layers(x)
        # x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, hiddens