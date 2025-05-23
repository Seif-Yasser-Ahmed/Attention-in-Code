import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()
        self.W_q=nn.Linear(d_model, d_model,bias=False) #Query
        self.W_k=nn.Linear(d_model, d_model,bias=False) #Key
        self.W_v=nn.Linear(d_model, d_model,bias=False) #Value
        self.row_dim=row_dim
        self.col_dim=col_dim

    def forward(self, token_embeddings):
        q=self.W_q(token_embeddings)
        k=self.W_k(token_embeddings)
        v=self.W_v(token_embeddings)

        sims=torch.matmul(q,k.transpose(dim0=self.row_dim,dim1=self.col_dim))
        scaled_sims=sims/torch.tensor(k.size(self.col_dim)**0.5)
        attention_percents=F.softmax(scaled_sims,dim=self.col_dim)
        attention_output=torch.matmul(attention_percents,v)
        return attention_output