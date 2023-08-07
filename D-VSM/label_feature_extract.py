import torch
import torch.nn as nn
import numpy as np


class l_feature_extract_layer(nn.Module):
    def __init__(self):
        super(l_feature_extract_layer, self).__init__()

    def forward(self, y, train_view_list):
        for j in range(len(train_view_list)):
            train_ins = train_view_list[j]
            class_num = y.shape[1]
            print("class_num ", class_num)
            node_num = train_ins.shape[0]
            print("node_num", node_num)
            ins_dim = train_ins.shape[1]
            print("ins_dim", ins_dim)
            l_init = torch.Tensor()
            for i in range(class_num):
                print("class num : ", i)
                class_mask = y[:, i:i + 1].view(node_num, 1)
                # print("class mask ",class_mask.shape , class_mask)

                # print(class_mask)
                class_mask = torch.stack([class_mask for i in range(ins_dim)], dim=1)
                print("mask", class_mask.shape)
                class_mask = torch.squeeze(class_mask)
                # class_mask.view(1,node_num,-1)
                print("mask", class_mask)
                print("mask 前", train_ins)
                masked_train_ins = torch.zeros((node_num,ins_dim))

                for k in range(node_num):
                    for l in range(ins_dim):
                        # masked_train_ins[k][l] = class_mask[k][l] == 0 ? train_ins[k][l] : 0
                        if class_mask[k][l]==0:
                            masked_train_ins[k][l] = train_ins[k][l]
                print("后", masked_train_ins)
                y_class_i = torch.sum(masked_train_ins, dim=0)
                print("y_class_i", y_class_i.shape)
                l_init = torch.cat([l_init, y_class_i], dim=0)
                print("l_init", l_init.shape)


a = torch.randint(0, 10, (500, 100))
b = torch.randint(0, 10, (500, 200))
c = torch.randint(0, 10, (500, 140))
y = torch.randint(0, 2, (500, 10))
e1 = np.random.randint(low=0, high=499, size=(2, 2500), dtype=int)
e2 = np.random.randint(low=0, high=499, size=(2, 2500), dtype=int)
e3 = np.random.randint(low=0, high=499, size=(2, 2500), dtype=int)
in_dim = [100, 200, 150]
epoch = 5
train_view = []
train_view.append(a)
train_view.append(b)
train_view.append(c)
edge_inx_list = []
edge_inx_list.append(e1)
edge_inx_list.append(e2)
edge_inx_list.append(e3)
model = l_feature_extract_layer()

h_final = model(y, train_view)
