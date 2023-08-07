import torch
import torch.nn as nn


class construct_loss(nn.Module):

    def __init__(self):
        super(construct_loss, self).__init__()

    def forward(self, rgcn_out, constr_out):
        mat = constr_out - rgcn_out
        l1_norm = torch.norm(mat, p=1)
        loss = l1_norm * l1_norm
        return loss
