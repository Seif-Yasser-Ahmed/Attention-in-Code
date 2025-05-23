import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=2,num_heads=1,row_dim=0,col_dim=1):
        super().__init__()
        self.heads=nn.ModuleList([Attention(d_model=d_model,row_dim=row_dim,col_dim=col_dim) for _ in range(num_heads)])
        self.col_dim=col_dim

    def forward(self,encodings_for_q,encodings_for_k,encodings_for_v):
        return torch.cat([head(encodings_for_q,encodings_for_k,encodings_for_v) for head in self.heads],dim=self.col_dim)