import torch
import numpy as np
from torch import nn

class CombinedModel(nn.Module):
    def __init__(self, one_model, two_model):
        super(CombinedModel, self).__init__()
        self.one_model = one_model
        self.two_model = two_model

    def forward(self, token_idx_batch, x_batch,edge_index_batch,edge_attr_batch,batch_batch):
        out_1 = self.one_model(token_idx_batch)
        out_2 = self.two_model(x_batch,edge_index_batch,edge_attr_batch,batch_batch)
        out_cat = torch.cat((out_1, out_2), dim=1)

        return out_cat










