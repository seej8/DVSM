import torch
import torch.nn as nn
import numpy as np


class gat():
    def __init__(self, attn_head, kernels, attn_kernels, a, attn_head_reduction):
        super(gat, self).__init__()
        self.attn_head = attn_head
        self.kernels = kernels
        self.A = a
        self.attn_kernels = attn_kernels
        self.attn_head_reduction = attn_head_reduction

    def forward(self, x):
        outputs = []
        for head in range(self.attn_head):
            attn_kernel = self.attn_kernels[head]
            kernel = self.kernels[head]
            features = torch.mul(x, kernel)  # N F

            attn_for_self = torch.mul(features, attn_kernel[0])
            attn_for_neighs = torch.mul(features, attn_kernel[1])
            dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
            lea = nn.LeakyReLU()
            dense = lea(dense)

            mask = -10e9 * (1 - self.A)
            dense += mask
            dense = nn.Softmax(dense)
            drop_attn = nn.Dropout(0.3)(dense)
            drop_feat = nn.Dropout(0.3)(features)

            node_features = torch.mul(drop_attn, drop_feat)
            outputs.append(node_features)

        if self.attn_head_reduction == 'concat':
            output = torch.cat(outputs)

        else:
            output = torch.mean(torch.stack(outputs, dim=0))

        output = nn.ReLU()(output)
        return output
