import torch
import scipy.io as scio
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from RGCN import RGCN
import torch.nn.functional as F
import dgl
import argparse
from dgl.nn.pytorch import RelGraphConv
from util import evaluate, data_loader2, get_global_edge
import pandas as pd
from torch_geometric.data import Data

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='Pascal07_5view.mat', type=str, metavar='N', help='run_data')
parser.add_argument('--data_root', default='data/', type=str, metavar='PATH',
                    help='root dir')
parser.add_argument('--word2vec', default='data/voc_glove_word2vec.pkl', type=str, metavar='PATH',
                    help='root dir')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--k', default=3, type=int, help='KNN')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end_epochs', default=500, type=int, metavar='H-P',
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--alpha', default=0.9, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--nhidden', default=256, type=int,
                    metavar='H-P', help='n_hidden')

args = parser.parse_args()

train_view, Y_train, test_view, Y_test, dim_view = data_loader2(args.data_root + args.dataset)
ntrain, ntest, nlabel, nview = Y_train.shape[0], Y_test.shape[0], Y_train.shape[1], len(dim_view)
g_list = []
test_g_list = []
feat_list = []
test_feat_list = []
for i in range(len(train_view)):
    print('----------------------------------------view:', i)
    view_feats = train_view[i]  # (5011, 100)
    test_feat = test_view[i]
    feat_list.append(view_feats)
    test_feat_list.append(test_feat)

    ins_num, fea_num = view_feats.size()
    test_ins_num, test_fea_num = test_feat.size()

    # train data
    data_dist = pairwise_distances(view_feats, metric='euclidean')  # (5011, 5011)
    topk_dist, topk_index = torch.from_numpy(data_dist).topk(dim=1, k=args.k, largest=False, sorted=True)  # (5011,5)
    knn_idx = topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
    ins_idx = torch.from_numpy(np.array(list(range(ins_num))).repeat(args.k).reshape(-1, 1)).type(
        torch.long)  # (25055,1)
    edge_idx_ins = torch.cat((ins_idx, knn_idx), dim=1).transpose(1, 0)  # (2,25055)

    # test data
    test_data_dist = pairwise_distances(test_feat, metric='euclidean')  # (5011, 5011)
    test_topk_dist, test_topk_index = torch.from_numpy(test_data_dist).topk(dim=1, k=args.k, largest=False,
                                                                            sorted=True)  # (5011,5)
    test_knn_idx = test_topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
    test_ins_idx = torch.from_numpy(np.array(list(range(test_ins_num))).repeat(args.k).reshape(-1, 1)).type(
        torch.long)  # (25055,1)
    test_edge_idx_ins = torch.cat((test_ins_idx, test_knn_idx), dim=1).transpose(1, 0)

    # Construct Network
    edges = edge_idx_ins[0].int(), edge_idx_ins[1].int()
    g = dgl.graph(edges)
    g.ndata['label'] = torch.tensor(Y_train).float().argmax(1).reshape(-1,
                                                                       1)  # [[0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1]
    g_list.append(g)

    # test
    test_edges = test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int()
    test_g = dgl.graph(test_edges)
    test_g.ndata['label'] = torch.tensor(Y_test).float().argmax(1).reshape(-1, 1)  #
    test_g_list.append(test_g)

G = dgl.batch(g_list)
print('g_list:', g_list)
print('G1:', G)
u, v = get_global_edge(ntrain, nview)
u, v = torch.tensor(u, dtype=torch.int32), torch.tensor(v, dtype=torch.int32)
print('u:{} {}'.format(u.shape, u[0:20]))
print('v:{} {}'.format(v.shape, v[0:20]))
G.add_edges(u, v)
print('G2:', G)
e_type = torch.zeros(G.number_of_edges())
e_type[G.edge_ids(u, v).long()] = 1

# test
test_G = dgl.batch(test_g_list)
test_u, test_v = get_global_edge(ntest, nview)
test_u, test_v = torch.tensor(test_u, dtype=torch.int32), torch.tensor(test_v, dtype=torch.int32)
test_G.add_edges(test_u, test_v)
test_e_type = torch.zeros(test_G.number_of_edges())
test_e_type[test_G.edge_ids(test_u, test_v).long()] = 1

model = RGCN(dim_view, 128, nlabel, 2, .2, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_acc = 0
best_epoch = 0
print(model)
print('feat_list_shape:', len(feat_list))
for i in range(len(feat_list)):
    print('i:{} shape:{}', i, feat_list[i].shape)
print('G3:', G)
print('e_type_shape', e_type.shape)
# print('Y_train0:', Y_train[0])
# aaa = input()
for epoch in range(args.end_epochs):
    model.train()
    pred = model(feat_list, G, e_type)  # (5011, 20)
    loss1 = ((1 - torch.tensor(Y_train)) * pred).sum()
    # loss2 = (-pred*torch.log(pred)).sum()
    loss = loss1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        acc, pred_index = evaluate(model, test_feat_list, test_G, test_e_type, torch.tensor(Y_test).argmax(1))
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            beat_pred_index = pred_index
        print('epoch: ' + str(epoch) + '  loss:' + str(loss.item()) + '    acc:' + str(acc))

print('best epoch:' + str(best_epoch) + '    ' + 'best acc:' + str(best_acc))

# def msg(edges):
#     return {'m':edges.src['label']==edges.dst['label']}
#
# g.apply_edges(msg)
# print('homelity: '+str(torch.true_divide(g.edata['m'].sum(), g.number_of_edges())))

# # Mymodel = Net(g, args.nlayers, train_data.shape[1], args.nhidden, lab_num, heads, F.elu, .2, .2, .2, True)
# Mymodel = Net(fea_num, args.nhidden, lab_num)
#
# if torch.cuda.is_available():
#     Mymodel = Mymodel.cuda()
