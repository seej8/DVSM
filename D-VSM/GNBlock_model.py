import torch
import time
import math
from torch.nn import Sequential as Seq, Linear as Lin, LayerNorm, ReLU, Sigmoid, Tanh, LeakyReLU, BatchNorm1d
from torch_scatter import scatter_mean, scatter_max, scatter_min
from conv_layer import ConvLayer
from build_graph import Graph_Generator
from util import generate_cross_graph

latent_dim = 512
final_latent_dim = 256


class GlobalNorm(torch.nn.Module):
    def __init__(self):
        super(GlobalNorm, self).__init__()

    def forward(self, data):
        # [h, l] = data.shape
        mean_ = torch.mean(data, dim=0).detach()
        var_ = torch.var(data, dim=0).detach()

        return (data - mean_) / torch.sqrt(var_ + 0.00001)


class NodeModel(torch.nn.Module):
    def __init__(self, insN_dim, insE_dim, labelN_dim, crossE_dim):
        super(NodeModel, self).__init__()
        # 32 32  8   32
        self.latent_insN_dim = final_latent_dim

        self.node_encoder_ins = Seq(Lin(insN_dim, latent_dim),
                                    ReLU(),
                                    GlobalNorm(),
                                    Lin(latent_dim, self.latent_insN_dim),
                                    ReLU(),
                                    GlobalNorm())

        self.node_mlp_ins_inner = Seq(Lin(self.latent_insN_dim + self.latent_insE_dim, self.latent_insN_dim * 2),
                                      ReLU(),
                                      GlobalNorm(),
                                      Lin(self.latent_insN_dim * 2, self.latent_insN_dim),
                                      ReLU(),
                                      GlobalNorm())
        self.node_mlp_ins_inter = Seq(Lin(self.latent_labelN_dim + self.latent_crossE_dim, self.latent_insN_dim * 2),
                                      ReLU(),
                                      GlobalNorm(),
                                      Lin(self.latent_insN_dim * 2, self.latent_insN_dim),
                                      ReLU(),
                                      GlobalNorm())
        self.node_mlp_ins_1 = Seq(Lin(self.latent_insN_dim * 3, self.latent_insN_dim),
                                  ReLU(),
                                  GlobalNorm(),
                                  Lin(self.latent_insN_dim, insN_dim),
                                  ReLU(),
                                  GlobalNorm())

        self.node_mlp_ins_2 = Seq(Lin(self.latent_insN_dim * 3, self.latent_insN_dim),
                                  ReLU(),
                                  GlobalNorm(),
                                  Lin(self.latent_insN_dim, insN_dim),
                                  ReLU(),
                                  GlobalNorm())

    def forward(self, state_ins, state_label, state_cross):
        # node_ins, edge_index_ins, edge_attr_ins = state_ins # 未知state_ins的输入形式
        node_ins, edge_index_ins = state_ins
        node_label = state_label
        edge_index_cross, edge_attr_cross = state_cross
        # 使用多层感知机处理
        """mapping the attributes into a latent space"""
        node_ins = self.node_encoder_ins(node_ins)

        """cross edges are directed from instances to labels"""
        """instance node attribute update"""

        row_ins, col_ins = edge_index_ins  # 5610, 5610
        out_ins = torch.cat([node_ins[row_ins], edge_attr_ins], dim=1)  # (5610,node_ins.shape[1]) , (?)
        out_ins = self.node_mlp_ins_1(out_ins)  # (5610,32)
        out_ins = scatter_mean(out_ins, col_ins, dim=0, dim_size=node_ins.size(0))  # (1122,32)
        node_ins = torch.cat([node_ins, out_ins], dim=1)
        node_ins = self.node_mlp_ins_2(node_ins)

        row_ins, col_ins = edge_index_ins
        row_cross, col_cross = edge_index_cross
        out_ins_inner = torch.cat([node_ins[row_ins], edge_attr_ins], dim=1)
        out_ins_inner = self.node_mlp_ins_inner(out_ins_inner)
        out_ins_inner = scatter_mean(out_ins_inner, col_ins, dim=0, dim_size=node_ins.size(0))

        out_ins_inter = torch.cat([node_label[col_cross], edge_attr_cross], dim=1)
        out_ins_inter = self.node_mlp_ins_inter(out_ins_inter)
        out_ins_inter = scatter_mean(out_ins_inter, row_cross, dim=0, dim_size=node_ins.size(0))

        node_ins = torch.cat([node_ins, out_ins_inner, out_ins_inter], dim=1)
        node_ins = self.node_mlp_ins(node_ins)

        """label node attribute update"""
        # row_label, col_label = edge_index_label
        # row_cross, col_cross = edge_index_cross
        # out_label_inner = torch.cat([node_label[row_label], edge_attr_label], dim=1)
        # out_label_inner = self.node_mlp_label_inner(out_label_inner)
        # out_label_inner = scatter_mean(out_label_inner, col_label, dim=0, dim_size=node_label.size(0))

        out_label_inter = torch.cat([node_ins[row_cross], edge_attr_cross], dim=1)  # (2504,64)
        out_label_inter = self.node_mlp_label_inter(out_label_inter)  # (2504,8)
        out_label_inter = scatter_mean(out_label_inter, col_cross, dim=0, dim_size=node_label.size(0))  # (16,8)
        node_label = torch.cat([node_label, out_label_inter], dim=1)
        node_label = self.node_mlp_label(node_label)

        state_ins = node_ins, edge_index_ins, edge_attr_ins
        state_label = node_label
        state_cross = edge_index_cross, edge_attr_cross

        return state_ins, state_label, state_cross


class _Model(torch.nn.Module):
    def __init__(self, dim_view):
        super(_Model, self).__init__()

        self.graph_generator = Graph_Generator()

        "dimension reduction"
        self.red_node_ins_view0 = Seq(Lin(dim_view[0], latent_dim), ReLU(), GlobalNorm(),
                                      Lin(latent_dim, final_latent_dim), ReLU(), GlobalNorm())
        self.red_node_ins_view1 = Seq(Lin(dim_view[1], latent_dim), ReLU(), GlobalNorm(),
                                      Lin(latent_dim, final_latent_dim), ReLU(), GlobalNorm())
        self.red_node_ins_view2 = Seq(Lin(dim_view[2], latent_dim), ReLU(), GlobalNorm(),
                                      Lin(latent_dim, final_latent_dim), ReLU(), GlobalNorm())
        self.red_node_ins_view3 = Seq(Lin(dim_view[3], latent_dim), ReLU(), GlobalNorm(),
                                      Lin(latent_dim, final_latent_dim), ReLU(), GlobalNorm())
        self.red_node_ins_view4 = Seq(Lin(dim_view[4], latent_dim), ReLU(), GlobalNorm(),
                                      Lin(latent_dim, final_latent_dim), ReLU(), GlobalNorm())

        self.pro_layer1 = ConvLayer(NodeModel(final_latent_dim))
        self.pro_layer2 = ConvLayer(NodeModel(final_latent_dim))

        self.decoder_edge_cross = Seq(
            Lin(final_latent_dim * len(dim_view), 256), ReLU(), GlobalNorm(), Lin(256, 1), Sigmoid()
        )

    def forward(self, graph_ins_list, view_cross_edges, global_cross_edges):
        node_ins_list, edge_index_ins_list = [], []
        for i in range(len(graph_ins_list)):
            node_ins_list.append(graph_ins_list[i]['x'])
            edge_index_ins_list.append(graph_ins_list[i]['edge_index'])

        """dimension reduction"""
        node_ins_list[0] = self.red_node_ins_view0(node_ins_list[0])
        node_ins_list[1] = self.red_node_ins_view0(node_ins_list[1])
        node_ins_list[2] = self.red_node_ins_view0(node_ins_list[2])
        node_ins_list[3] = self.red_node_ins_view0(node_ins_list[3])
        node_ins_list[4] = self.red_node_ins_view0(node_ins_list[4])

        """convolution layers"""
        state_ins = [node_ins_list, edge_index_ins_list]

        state_ins, view_cross_edges, state_cross = self.pro_layer1(state_ins, view_cross_edges, global_cross_edges)
        state_ins, view_cross_edges, state_cross = self.pro_layer2(state_ins, view_cross_edges, global_cross_edges)

        _, edge_attr_cross = state_cross  # (2504,32)

        prediction = self.decoder_edge_cross(edge_attr_cross)

        return prediction

    def data_normlize(self, data):
        return -1 + 2 * data
