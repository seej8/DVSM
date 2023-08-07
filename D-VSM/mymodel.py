import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F
import numpy as np


# gcn_msg = fn.copy_src(src="h", out="m")
# gcn_reduce = fn.sum(msg="m", out="h")  # 聚合邻居节点的特征

# 用于传递信息和聚合信息的两个函数
def gcn_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


def gcn_msg(edges):
    return {'m': edges.src['h']}


# 定义节点的UDF apply_nodes  他是一个完全连接层
class NodeApplyModule(nn.Module):
    # 初始化
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    # 前向传播
    def forward(self, node):
        h = self.linear(node.data["h"])
        if self.activation is not None:
            h = self.activation(h)
        return {"h": h}


# 定义GCN模块  GCN模块的本质是在所有节点上执行消息传递  然后再调用NodeApplyModule全连接层
class GCN(nn.Module):
    # 初始化
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        # 调用全连接层模块
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    # 前向传播
    def forward(self, g, feature):
        g.ndata["h"] = feature  # feature应该对应的整个图的特征矩阵
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)  # 将更新操作应用到节点上

        return g.ndata.pop("h")


# 利用cora数据集搭建网络然后训练
class Net(nn.Module):
    # 初始化网络参数
    def __init__(self, in_feats, out_feats, nlabel):
        super(Net, self).__init__()
        self.gcn1 = GCN(in_feats, out_feats, F.relu)  # 第一层GCN
        self.gcn2 = GCN(out_feats, nlabel, None)

    # 前向传播
    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x


class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # input = (108,in_features)
        h = torch.mm(input, self.W)  # (108, 1024)
        # N = h.size()[0]

        f_1 = torch.matmul(h, self.a1)  # (1122,1)
        f_2 = torch.matmul(h, self.a2)  # (1122,1)
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))  # (1122, 1122)
        # 设置极小注意力值
        zero_vec = -9e15 * torch.ones_like(e)  # (1122, 1122)
        attention = torch.where(adj > 0, e, zero_vec)  # (1122, 1122) torch.where(condition, x,
        # y) 若满足condition，返回x中的结果，否则返回y中的结果
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # (1122, 1024)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# 使用torch中的函数来实现gat，用不上
class GAT_by_torch(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_by_torch, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.bn = nn.BatchNorm1d(nfeat)
        # 输出注意力层，由于输入x是由多个头拼接起来所以形状为nhid * nheads
        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = self.bn(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # 先经过所有注意力层
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # 经过输出层
        x = self.out_att(x, adj)
        return x

# 一个可用的GAT模块
class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        self.bn = nn.BatchNorm1d(in_dim)

    def forward(self, inputs):
        h = inputs
        h = self.bn(h)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bn = nn.BatchNorm1d(input_dim)

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(self.bn(x))
        else:
            # If MLP
            h = self.bn(x)
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)
