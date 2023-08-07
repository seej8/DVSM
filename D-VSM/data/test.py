import numpy as np
import torch
import dgl

src = torch.LongTensor(
    [0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 10, 1, 2, 3, 3, 3, 4, 5, 5, 6, 5, 8, 6, 8, 9, 8, 11, 11, 10, 11])
dst = torch.LongTensor(
    [1, 2, 3, 3, 3, 4, 5, 5, 6, 5, 8, 6, 8, 9, 8, 11, 11, 10, 11, 0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 10])
g = dgl.graph((src, dst))
li = torch.tensor([0,1,2])
li1 = torch.tensor([0,1,2])
x = torch.cat([li,li1],dim=0)
sampler = dgl.dataloading.NeighborSampler([1, 2])
cre = torch.nn.CrossEntropyLoss()
logits = torch.tensor([.0,.4,.6],dtype=float)
labels = torch.zeros(3).to('cpu').float()
loss = cre(logits,labels)
print(loss)
# dataloader = dgl.dataloading.DataLoader(
#     g, [2], sampler,
#     batch_size=1, shuffle=True, drop_last=False)
# for input_nodes, output_nodes, blocks in dataloader:
#     print(input_nodes) # 输入的节点数
#     print(output_nodes) # 输出的节点数
#     print(blocks)
#     for i in range(len(blocks)):
#         left, right = blocks[i].edges()
#         u = [int(input_nodes[l]) for l in left]
#         v = [int(input_nodes[r]) for r in right]
#         print("u  v", u, v)
#         print("此块输入节点", blocks[i].srcdata)
#         print("此块输出节点", blocks[i].dstdata)
#         print("*" * 10)
#     print("="*10)
# print(x)
