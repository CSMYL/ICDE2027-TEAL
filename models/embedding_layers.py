import numpy as np
import torch
import torch.nn as nn

class UciEmbedding(nn.Module):
    ''' Simplified version for ALL numerical features '''
    def __init__(self, nfeat, nemb):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(nfeat, nemb))  # F×E
        nn.init.xavier_uniform_(self.embedding)

    def forward(self, x):
        """
        :param x:   FloatTensor B×F
        :return:    embeddings B×F×E
        """
        return torch.einsum('bf,fe->bfe', x, self.embedding)    # B×F×E

class Embedding(nn.Module):
    def __init__(self, nfeat, nemb):
        super().__init__()
        self.embedding = nn.Embedding(nfeat, nemb)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B×F, 'vals': FloatTensor B×F}
        :return:    embeddings B×F×E
        """
        emb = self.embedding(x['ids'])                      # B×F×E
        return emb * x['vals'].unsqueeze(2)                # B×F×E

class Linear(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        self.weight = nn.Embedding(nfeat, 1)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B×F, 'vals': FloatTensor B×F}
        :return:    linear transform of x  -> B
        """
        linear = self.weight(x['ids']).squeeze(2) * x['vals']  # B×F
        return torch.sum(linear, dim=1) + self.bias           # B

# =====================
# Decoding layers: for embedding reverse mapping when parameter sharing
# =====================
class DecodingEmbedding(nn.Module):
    """
    Decoding layer: when sharing parameters, use the transpose of encoder embedding as output layer weights.
    Supports processing categorical feature IDs with offset adjustments.
    """
    def __init__(self, encoder_embedding, num_classes_dict, cat_idx, cat_offsets=None):
        super().__init__()
        self.embedding = encoder_embedding.embedding  # nn.Embedding instance
        self.num_classes_dict = num_classes_dict
        self.cat_idx = cat_idx
        self.cat_offsets = cat_offsets or {}

    def forward(self, x):
        # x: [B, F, D], output: each feature corresponds to its class number output
        # Decode each feature separately
        B, F, D = x.shape
        outputs = []
        for i, idx in enumerate(self.cat_idx):
            # Get number of classes for current feature
            n_class = self.num_classes_dict.get(idx.item(), 10)
            # Get offset for current feature
            offset = self.cat_offsets.get(idx.item(), 0)
            # Linear transformation for current feature using corresponding embedding weight range
            out = torch.matmul(x[:, i, :], self.embedding.weight[offset:offset+n_class, :].T)  # [B, n_class]
            outputs.append(out)
        return outputs

class DecodingUciEmbedding(nn.Module):
    """
    Decoding layer: when sharing parameters, use the transpose of encoder UciEmbedding parameters as output layer weights.
    """
    def __init__(self, encoder_embedding):
        super().__init__()
        self.embedding = encoder_embedding.embedding  # nn.Parameter [F, D]

    def forward(self, x):
        # x: [B, F, D], output: [B, F]
        # Use einsum for efficient batch matrix multiplication
        # [B, F, D] × [F, D] -> [B, F]
        outputs = torch.einsum('bfd,fd->bf', x, self.embedding)
        return outputs