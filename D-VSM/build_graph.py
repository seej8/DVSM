import os

import numpy as np
import torch
import scipy.io as scio
from sklearn.metrics.pairwise import pairwise_distances
import h5py
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import MetaLayer
import itertools


class Graph_Generator(torch.nn.Module):
    def __init__(self):
        super(Graph_Generator, self).__init__()

    def build_graph(self, train_feats, label_feats, target, k):
        num_view, num_ins, num_label = len(train_feats), target.shape[0], label_feats.shape[0]  # 5   # 20
        print("train feat : {} , label feat : {} , target : {}".format(train_feats[0].shape,label_feats.shape,target.shape))
        """build instance graph 同一视图下不同实例之间的图"""
        view_graph_list = []
        for i in range(num_view):
            view_feats = train_feats[i]  # (5011, 100) 5011 nodes each 100 dim feats
            data_dist = pairwise_distances(view_feats, metric='euclidean')  # (5011, 5011)
            print("data dist : {} ".format(data_dist))
            print("data dist : {} ".format(data_dist.shape))
            topk_dist, topk_index = torch.from_numpy(data_dist).topk(dim=1, k=k, largest=False, sorted=True)  # (5011,5)
            print("topk dist : {} , topk inx : {}".format(topk_dist,topk_index))
            dst_idx = topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
            print("list(range(num_ins)) :{}".format(list(range(num_ins))))
            src_idx = torch.from_numpy(np.array(list(range(num_ins))).repeat(k).reshape(-1, 1)).type(
                torch.long)  # (25055,1)
            print("src inx : {} ".format(src_idx))
            # 每个node选择最近的5个node，建立边索引
            edge_idx_ins = torch.cat((src_idx, dst_idx), dim=1).transpose(1, 0)  # (2,25055)
            # edge_attr_ins=torch.cat((view_feats[edge_idx_ins[0,:]],view_feats[edge_idx_ins[1,:]]),dim=1) # (25055,200)
            print("edge inx :{}".format(edge_idx_ins))
            graph_ins = Data(x=view_feats, edge_index=edge_idx_ins, kwargs=[num_ins, num_label])
            # 获得一个存放所有单体特征图的列表，元素为含x，边索引，label的data object
            view_graph_list.append(graph_ins)

        """build instance graph 不同视图下同一实例之间的图"""
        ins_idx_C = torch.from_numpy(np.array(list(range(num_ins))).repeat(num_view).reshape(-1, 1)).type(
            torch.long)  # （25055,1） 0,0,0,0, 40个0,0,0,1,1,1,40个1,1,1...
        view_idx_C = torch.from_numpy(np.array(list(range(num_view))).reshape(-1, 1)).repeat(num_ins, 1).type(
            torch.long)  # （25055，1）0, 1, 2, ..., 19, 0, 1,2,...,19,1,2,..
        view_edge_index_C = torch.cat((ins_idx_C, view_idx_C), dim=1).cuda()  # (25055, 2)

        """build global-inst-label cross edge"""
        gins_idx_C = torch.from_numpy(np.array(list(range(num_ins))).repeat(num_label).reshape(-1, 1)).type(
            torch.long)  # （800,1） 0,0,0,0, 40个0,0,0,1,1,1,40个1,1,1...
        label_idx_C = torch.from_numpy(np.array(list(range(num_label))).reshape(-1, 1)).repeat(num_ins, 1).type(
            torch.long)  # （800，1）1, 2, ..., 19, 1,2,...,19,1,2,..
        gedge_index_C = torch.cat((gins_idx_C, label_idx_C), dim=1).cuda()  # (800, 2)

        """build label graph"""
        if torch.cuda.is_available():
            graph_label = Data(x=label_feats.cuda(), kwargs1=view_edge_index_C.cuda(), kwargs2=gedge_index_C.cuda())
        else:
            graph_label = Data(x=label_feats, kwargs1=view_edge_index_C, kwargs2=gedge_index_C)

        return view_graph_list, graph_label
        # return view_graph_list

    def forward(self, ins_feats, label_feats, target, k):
        graph_ins, graph_label = self.build_graph(ins_feats, label_feats, target, k)

        return graph_ins, graph_label


class GraphData(Dataset):
    def __init__(self):
        super(GraphData, self).__init__()
        self.graph_generator = Graph_Generator()

    def __len__(self):
        'Denotes the total number of samples'
        return 10000

    def __getitem__(self, index):
        'Generates one sample of data'
        dataset = scio.loadmat('N:/Dataset/mat/lyu_data/lost.mat')

        ins_feature = torch.from_numpy(dataset['data'])
        partial_target = torch.from_numpy(dataset['partial_target'])
        print("-" * 66)
        graph_ins, graph_label, graph_cross = self.graph_generator(ins_feature, partial_target)
        return graph_ins, graph_label, graph_cross


file_list = []
path = 'D:/GNN/D-vsm-dataset/D-vsm-dataset/Corel5k.mat'

train_feats = []
feat_1 = torch.rand((10,20))
feat_2 = torch.rand((10,20))
feat_3 = torch.rand((10,20))
train_feats.append(feat_1)
train_feats.append(feat_2)
train_feats.append(feat_3)
label_feats = torch.rand((10,30))
target = torch.rand((10,30))
model = Graph_Generator()
model.build_graph(train_feats,label_feats,target,3)
# print(feat_1)




# for root, dirs, files in os.walk(direc):
#     for file in files:
#         file_path = os.path.join(root, file)
#         # print(file)
#         # print(file_path)
#         if file == 'Iaprtc12.mat' or file == 'Espgame.mat' or file == 'Mirflickr.mat':
#             dataset = h5py.File(file_path, 'r')
#             # dataset = scio.loadmat(file_path)
#             print(file, "   ")
#             for key in dataset:
#                 print(key)
#                 print(dataset[key])
#                 if key == 'data':
#                     print(dataset[key].shape)
#                 elif key == 'target':
#                     print(dataset[key].shape)
#
#             del dataset
#             # dataset.close()
#         else:
#
#             dataset = scio.loadmat(file_path)
#             print(file, "   ")
#             for key in dataset:
#                 print(key)
#                 # print(dataset[key])
#                 if key == 'data' and dataset[key].shape[0] == 1:
#                     print(dataset[key][0][0].shape)
#                 elif key == 'data' and dataset[key].shape[0] != 1:
#                     print(dataset[key][1][0].shape)
#                 elif key == 'target':
#                     print(dataset[key].shape)
#             del dataset
#

# dataset = GraphData()
# loader = DataLoader(dataset, batch_size=2)
#
# for idx, data in enumerate(loader):
#     graph_ins, graph_label, graph_cross = data
#     print(graph_ins["x"])
#     print(graph_ins["edge_index"])
#     print(graph_ins["edge_attr"])
#     print("*" * 66)
#     print(graph_label["x"])
#     print(graph_label["edge_index"])
#     print(graph_label["edge_attr"])
#     print("*" * 66)
#     print(graph_cross["x"])
#     print(graph_cross["edge_index"])
#     print(graph_cross["edge_attr"])
#     print("*" * 66)
#     input()
#
# # print(bbox_info)
# # print(lable_feats)
# print("-" * 66)
# # for graph in graph_generator(bbox_info,lable_feats):
# # 	print(graph["x"])
# # 	print(graph["edge_index"])
# # 	print(graph["edge_attr"])
# # 	print("*" * 66)
