# import torch
# import scipy.io as scio
# from sklearn.metrics.pairwise import pairwise_distances
# import numpy as np
# from mymodel import GAT, GAT_by_torch, Net
# from RGCN import RGCN
# import torch.nn.functional as F
# import dgl
# import argparse
# from dgl.nn.pytorch import RelGraphConv
# from util import evaluate, data_loader2, get_global_edge
import scipy.io as sio
# import pandas as pd
# from torch_geometric.data import Data
# import time
# import math

mat = sio.loadmat(r'data/emotions.mat')
print(sorted(mat.keys()))
x1 = mat['data'][0][0]
print(x1.shape)
x2 = mat['data'][0][1]
print(x2.shape)
