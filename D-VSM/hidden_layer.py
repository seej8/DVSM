import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class update_h_net(nn.Module):
    def __init__(self, in_feat_dim_list):
        super(update_h_net, self).__init__()
        self.hidden_dim = 256

        # self.H_ins = nn.Linear(in_feat,self.hidden_dim)
        self.H_ins = nn.ModuleList()
        for i in range(len(in_feat_dim_list)):
            self.H_ins.append(nn.Linear(in_feat_dim_list[i], self.hidden_dim))
        self.w0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU()

    # 传入视图列表，以及k近邻的k
    def forward(self, train_view, k, edge_inx_list, update_epoch,alpha_list):
        # 所有视图下的输出 5 ,3999 ,256
        h_final_list = []
        h_init_list = []
        # x : 视图索引
        for x in range(len(train_view)):
            train_ins = train_view[x]
            h_init = self.H_ins[x](train_ins)
            h_init_list.append(h_init)
            print("h_init : ", h_init.shape)

        for x in range(len(train_view)):
            print("view :{}".format(x))
            edge_inx = edge_inx_list[x]
            print("edge_inx ", edge_inx.shape)
            train_ins = train_view[x]
            print("train_ins", train_ins.shape)
            # 得到初始的H 500 * 256

            for y in range(update_epoch):
                print("epoch : {}".format(y))
                h_init = h_init_list[x]
                # print("h_init : " ,h_init.shape)
                # 自画像 500,256
                h_init = self.w0(h_init)
                print("h_init : ", h_init.shape)

                node_num = train_ins.shape[0]
                # 迭代每个节点
                for i in range(node_num):
                    h_neighbor_sum = torch.Tensor(1, self.hidden_dim)
                    # 综合邻居节点特征
                    for j in range(k):
                        node_inx = edge_inx[1][i * 5 + j]
                        # 1,256
                        neighbor_hidden = torch.tensor(h_init[node_inx])
                        h_neighbor = self.w1(neighbor_hidden).reshape(1, -1)  # 1,256

                        h_neighbor_sum += h_neighbor

                    h_neighbor_sum /= k
                    print("h_init[{}] : ".format(i), h_init[i].sum())
                    print("h_neighbor_sum : ", h_neighbor_sum.sum())
                    h_init[i] += h_neighbor_sum[0]
                    print("h_init", h_init[i].sum())

                    # 综合不同视图下的自己的特征
                    h_self_sum = torch.Tensor(1, self.hidden_dim)
                    for k in range(len(train_view)):
                        if k != x:
                            diff_self_h = torch.tensor(h_init_list[k][i])
                            h_self = self.w2(diff_self_h)
                            h_self_sum += h_self

                    h_self_sum /= (k - 1)
                    h_init[i] += h_self_sum[0]

                train_ins = self.relu(h_init)
            # 500,256
            h_final_list.append(train_ins)
            h_out = torch.Tensor(h_final_list[0].shape[0], h_final_list[0].shape[1])
            for i in range(len(h_final_list)):
                h_out += h_final_list[i]
            h_out /= len(h_final_list)
            print(h_out)
        return h_out


# a = torch.rand(500, 100)
# b = torch.rand(500, 200)
# c = torch.rand(500, 150)
# e1 = np.random.randint(low=0, high=499, size=(2, 2500), dtype=int)
# e2 = np.random.randint(low=0, high=499, size=(2, 2500), dtype=int)
# e3 = np.random.randint(low=0, high=499, size=(2, 2500), dtype=int)
# in_dim = [100, 200, 150]
# epoch = 5
# train_view = []
# train_view.append(a)
# train_view.append(b)
# train_view.append(c)
# edge_inx_list = []
# edge_inx_list.append(e1)
# edge_inx_list.append(e2)
# edge_inx_list.append(e3)
# model = update_h_net(in_feat_dim_list=in_dim)
#
# h_final = model(train_view=train_view, k=5, edge_inx_list=edge_inx_list, update_epoch=1)
