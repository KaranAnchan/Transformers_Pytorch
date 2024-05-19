import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix Of Shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Vector Of Shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Sine Function applied on Even Positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Cosine Function applied on Odd Positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Expand Dimension to include Batches (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register the Tensor in Buffer to Save Along the Model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean)/ (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, dff: int, dropout: float):
        super().__init__()

        # Weights are already true in the Linear Layer
        self.linear_1 = nn.Linear(d_model, dff) # W1 + b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff, d_model) # W2 + b2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)