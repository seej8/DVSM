import torch
import scipy.io as scio
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
# from mymodel import GAT, GAT_by_torch, Net
from RGCN import RGCN
import torch.nn.functional as F
import dgl
import argparse
from dgl.nn.pytorch import RelGraphConv
from util import *
import scipy.io as sio
import pandas as pd
# from torch_geometric.data import Data
import time
import math
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import sys
import h5py
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_loss
from scipy.io import arff


def evaluate_new(training, model, feats, g, e_type, labels, crit, thres=0):
    if training:
        model.train()
        logits = model(feats, g, e_type)  # (1010, 16)
        ap2 = avgp(logits - thres, labels)
        return ap2, crit(logits, labels)
    model.eval()
    with torch.no_grad():
        logits = model(feats, g, e_type)  # (1010, 16)
        for i in range(0):
            print('logits:', logits[i])
            print('labels:', labels[i])

        cnt = 0
        num = 0
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i][j] == 1:
                    num += 1
                    if logits[i][j] >= thres:
                        cnt += 1
        # 准确率
        print('acc_cnt:', cnt / num)

        # ap.add(logits - thres, labels)
        # print(ap.value())
        ap2 = avgp(logits - thres, labels)
        return ap2, crit(logits, labels)
        # return cnt * 1.0 / (labels.shape[0] * labels.shape[1]), crit(logits, labels)


"""这段代码定义了一个名为 get_global_edge_new 的函数，
用于生成全局边的索引列表。函数采用两个参数：
nins 表示实例的数量，nview 表示视图的数量。
在函数的实现中，使用了三个嵌套的循环。外层循环控制视图的索引，中层循环控制与当前视图不同的其他视图的索引
，内层循环控制实例的索引。
对于每对不同的视图索引 (i, j)，内部循环遍历实例索引 k，并根据索引生成两对边 (u, v)：
第一对边：(i * nins + k, j * nins + k)
第二对边：(j * nins + k, i * nins + k)
这样就生成了 u 和 v 两个列表，其中 u 列表存储所有的起始节点索引，而 v 列表存储所有的终止节点索引。
最后，函数返回生成的 u 和 v 列表作为结果。"""


def get_global_edge_new(nins, nview):
    # print('nins:{} nview:{}'.format(nins, nview))
    u = []
    v = []
    """在每次迭代过程中，通过以下方式生成两对边 (u, v)：
    第一对边 (u, v)：起始节点 u 的索引为 i * nins + k，终止节点 v 的索引为 j * nins + k。
    第二对边 (v, u)：起始节点 v 的索引为 j * nins + k，终止节点 u 的索引为 i * nins + k。"""
    for i in range(nview):
        for j in range(i + 1, nview, 1):
            for k in range(nins):
                u.append(i * nins + k)
                v.append(j * nins + k)

                v.append(i * nins + k)
                u.append(j * nins + k)
    return u, v


def data_loader(path):
    print('loadmat begin')

    if args.dataset == 'Iaprtc12.mat' or args.dataset == 'Espgame.mat' or args.dataset == 'Mirflickr.mat':
        mat = h5py.File(path, 'r')
        view = []
        # 添加五个视图

        for i in range(6):
            los = mat['data'][0][i]
            dat = np.array(mat[los])
            view.append(torch.from_numpy(dat).transpose(1, 0))
            # print('dat:', dat.shape)
        # print('tar:', mat['target'])
        # lo = mat['target'][0][0]
        da = np.array(mat['target'])
        target = torch.from_numpy(da)

    elif args.dataset == 'scene.arff' or args.dataset == 'tmc2007.arff':
        data = arff.loadarff(path)
        df = pd.DataFrame(data)
        print(df)
        x = input()
    else:
        mat = sio.loadmat(path)
        print('loadmat end')
        view = []
        target = torch.from_numpy(mat['target'])
        print('data:', mat['data'].shape)
        for i in range(mat['data'].shape[0]):
            for j in range(mat['data'].shape[1]):
                print('i:{} j:{} shape:{}'.format(i, j, mat['data'][i][j].shape))
        # mat['data']的两个维度中，预先设定其中一个为一
        if mat['data'].shape[0] == 1:
            for i in range(mat['data'].shape[1]):
                view.append(torch.from_numpy(mat['data'][0][i].astype(np.float64)))
        elif mat['data'].shape[1] == 1:
            for i in range(mat['data'].shape[0]):
                view.append(torch.from_numpy(mat['data'][i][0].astype(np.float64)))
        else:
            print('View data type error!')
    print('len_view:', len(view))
    for i in range(len(view)):
        print('i:{} shape:{}'.format(i, view[i].shape))
    print('target:', target.shape)
    # x = input()
    return view, target
    # return torch.tensor(view), torch.tensor(mat['target'])


def construct_graph_k(x, dim_view):
    # 总之，这段代码的目的是构建一个测试图，该图使用输入的数据 x 和 dim_view 维度信息，
    # 并为测试集生成全局边的索引
    return construct_graph(x, dim_view)
    ntest, nview = x[0].shape[0], len(dim_view)
    # print('ntest:{} nview:{}'.format(ntest, nview))
    test_u, test_v = get_global_edge_new(ntest, nview)
    test_u, test_v = torch.tensor(test_u, dtype=torch.int32), torch.tensor(test_v, dtype=torch.int32)
    test_edges = test_u, test_v
    test_G = dgl.graph(test_edges)

    test_g_list = []
    test_feat_list = []
    test_e_type = torch.zeros(ntest * (args.k - 1) * nview * 2 + test_G.number_of_edges())
    knn_edge_id_u = []
    knn_edge_id_v = []
    # print('edge_num:', test_G.number_of_edges())
    # 构造knn的边
    for i in range(args.k - 1):
        knn_edge_id_v.append([])
        knn_edge_id_u.append([])
    for i in range(len(x)):
        test_feat = x[i]
        test_feat_list.append(test_feat.float())
        test_ins_num, test_fea_num = test_feat.size()
        test_data_dist = pairwise_distances(test_feat, metric='euclidean')  # (5011, 5011)
        test_topk_dist, test_topk_index = torch.from_numpy(test_data_dist).topk(dim=1, k=args.k, largest=False,
                                                                                sorted=True)  # (5011,5)
        # print('top_pre:', test_topk_index)
        test_topk_index += i * ntest
        # print('top_aft:', test_topk_index)
        test_topk_index = test_topk_index[:, 1:]
        test_knn_idx = test_topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        test_ins_idx = torch.from_numpy(np.array(list(range(test_ins_num))).repeat(args.k - 1).reshape(-1, 1)).type(
            torch.long)  # (25055,1)
        test_edge_idx_ins = torch.cat((test_ins_idx, test_knn_idx), dim=1).transpose(1, 0)
        test_edges = test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int()
        # print('test_dges:', test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int())
        test_g = dgl.graph(test_edges)
        test_G.add_edges(test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int())
        test_g_list.append(test_g)
        test_G.add_edges(test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int())
        for j in range(args.k - 1):
            knn_edge_id_v[j].append(test_topk_index[:, j].reshape(-1, 1).type(torch.long).squeeze())
            knn_edge_id_u[j].append(
                torch.from_numpy(np.array(list(range(ntest)))).reshape(-1, 1).type(torch.long).squeeze())

            knn_edge_id_u[j].append(test_topk_index[:, j].reshape(-1, 1).type(torch.long).squeeze())
            knn_edge_id_v[j].append(
                torch.from_numpy(np.array(list(range(ntest)))).reshape(-1, 1).type(torch.long).squeeze())
            # print('j:{} u:{} v:{}'.format(j, knn_edge_id_u[j][-1].shape, knn_edge_id_v[j][-1].shape))

    # test_G = dgl.batch(test_g_list)

    # print('test_u:', test_u)
    # print('test_v:', test_v)
    # print('edge_num:', test_G.number_of_edges())
    # print('u:', knn_edge_id_u[0].shape)
    # print('v:', knn_edge_id_v[0].shape)
    for i in range(args.k - 1):
        knnv = knn_edge_id_v[i][0]
        knnu = knn_edge_id_u[i][0]
        for j in range(len(knn_edge_id_v[i])):
            if j == 0:
                continue
            knnv = torch.cat((knnv, knn_edge_id_v[i][j]), dim=0)
            knnu = torch.cat((knnu, knn_edge_id_u[i][j]), dim=0)
        knnu = knnu.int()
        knnv = knnv.int()
        # knnv = torch.cat(knn_edge_id_v, dim=0).int()
        # knnu = torch.cat(knn_edge_id_u, dim=0).int()
        # knn_edge_id_u[i] = torch.tensor(knn_edge_id_u[i])
        # knn_edge_id_v[i] = torch.tensor(knn_edge_id_v[i])
        # knnu, knnv = torch.tensor(knnu.clone().detach(), dtype=torch.int32), torch.tensor(knnv.clone().detach(), dtype=torch.int32)
        # for j in range(knnu.shape[0]):
        # print('knnu:{} knnv:{}'.format(knnu[j], knnv[j]))
        # print('knnu:{} knnv:{}'.format(knnu.shape, knnv.shape))

        mid = 23
        # print('knnu:{} knnv:{}'.format(knnu[mid], knnv[mid]))
        # print('edge:', test_G.edge_ids(knnu[0:mid], knnv[0:mid]))
        test_e_type[test_G.edge_ids(knnu, knnv).long()] = i
    # for i in range(args.k - 1):
    # test_e_type[test_G.edge_ids(knn_edge_id_u[i], knn_edge_id_v[i]).long()] = i
    test_e_type[test_G.edge_ids(test_u, test_v).long()] = args.k - 1
    return test_feat_list, test_G, test_e_type


def construct_graph(x, dim_view):
    ntest, nview = x[0].shape[0], len(dim_view)
    test_g_list = []
    test_feat_list = []
    for i in range(len(x)):
        test_feat = x[i]
        test_feat_list.append(test_feat.float())
        test_ins_num, test_fea_num = test_feat.size()
        # test data
        test_data_dist = pairwise_distances(test_feat, metric='euclidean')  # (5011, 5011)
        test_topk_dist, test_topk_index = torch.from_numpy(test_data_dist).topk(dim=1, k=args.k, largest=False,
                                                                                sorted=True)  # (5011,5)
        # print('topk_index_shape', test_topk_index.shape)
        test_topk_index = test_topk_index[:, 1:]
        test_knn_idx = test_topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        test_ins_idx = torch.from_numpy(np.array(list(range(test_ins_num))).repeat(args.k - 1).reshape(-1, 1)).type(
            torch.long)  # (25055,1)
        test_edge_idx_ins = torch.cat((test_ins_idx, test_knn_idx), dim=1).transpose(1, 0)
        # print('test_ins_num:{} test_fea_num:{}'.format(test_ins_num, test_fea_num))
        # print('test_data_dist:', test_data_dist)
        # print('test_topk_dist:', test_topk_dist)
        # print('test_knn_idx:', test_knn_idx)
        # print('test_edge_idx_inx', test_edge_idx_ins)
        # test
        test_edges = test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int()
        # print('test_edges:', test_edges)
        # x = input()
        test_g = dgl.graph(test_edges)
        test_g.add_edges(test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int())
        # test_g.ndata['label'] = torch.tensor(Y_test.clone().detach()).float().argmax(1).reshape(-1, 1)  #
        # test_g.ndata['label'] = y.float().argmax(1).reshape(-1, 1).clone().detach()
        # test_g.add_edges(test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int())

        # test_edges2 = test_edge_idx_ins[1].int(), test_edge_idx_ins[0].int()

        test_g_list.append(test_g)
    # test
    test_G = dgl.batch(test_g_list)
    test_u, test_v = get_global_edge_new(ntest, nview)
    test_u, test_v = torch.tensor(test_u, dtype=torch.int32), torch.tensor(test_v, dtype=torch.int32)
    test_G.add_edges(test_u, test_v)
    # test_G = dgl.graph((test_u, test_v))
    test_e_type = torch.zeros(test_G.number_of_edges())
    test_e_type[test_G.edge_ids(test_u, test_v).long()] = 1
    return test_feat_list, test_G, test_e_type


def add_test(x_train, y_train, x_test, y_test, model, crit, dim_view):
    # acc_test, loss_test = 0, 0
    test_label = torch.zeros(y_test.shape[0], y_test.shape[1])
    ap = AveragePrecisionMeter()
    model.eval()
    for i in range(y_test.shape[0]):
        x_fus = []  # 混合测试集和训练集，用于测试
        for j in range(len(x_test)):
            # print('xtj:', x_train[j].shape)

            x_fusj = torch.cat((x_train[j], x_test[j][i].view(1, -1)), dim=0)
            x_fus.append(x_fusj)
        # y_fus = torch.cat((y_train, y_test[i].view(1,-1)), dim=0)
        feats, g, e_type = construct_graph(x_fus, dim_view)
        with torch.no_grad():
            logits = model(feats, g, e_type)  # (1010, 16)
            test_label[i] = logits[-1]
    cnt = 0
    num = 0
    laebl_ham = torch.zeros(test_label.shape[0], test_label.shape[1])
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if test_label[i][j] > 0:
                laebl_ham[i][j] = 1
            else:
                laebl_ham[i][j] = 0
            if y_test[i][j] == 1:
                num += 1
                if test_label[i][j] >= 0:
                    cnt += 1

    # print('acc_cnt:', cnt / num)
    ap.add(test_label, y_test)
    # print('ap:', ap.value(), ap.value().mean())
    ham = hamming_loss(y_test, laebl_ham)
    # print('ham:', ham)
    # cov = coverage_error(y_test, test_label)
    # print('cov:', cov)
    rank = label_ranking_loss(y_test, test_label)
    # print('rank:', rank)
    one = one_error(test_label, y_test)
    # print('one:', one)
    return ap.value().mean(), crit(test_label, y_test)


def add_test_whole(x_train, y_train, x_test, y_test, model, crit, dim_view):
    return 0, 10000, 0, 100000, 100000, 100000, 0, 1000000, 0, 0
    # acc_test, loss_test = 0, 0
    # test_label = torch.zeros(y_test.shape[0], y_test.shape[1])
    # ap = AveragePrecisionMeter()
    x_fus = []  # 混合测试集和训练集，用于测试
    for j in range(len(x_test)):
        # print('xtj:', x_train[j].shape)
        x_fusj = torch.cat((x_train[j], x_test[j]), dim=0)
        x_fus.append(x_fusj)
    # y_fus = torch.cat((y_train, y_test), dim=0)
    feats, g, e_type = construct_graph(x_fus, dim_view)
    model.eval()

    with torch.no_grad():
        logits = model(feats, g, e_type)  # (1010, 16)
        test_label = logits[x_train[0].shape[0]:]
    # print('test_label:{}  y_test:{}'.format(test_label.shape, y_test.shape))
    label_ham = torch.zeros(test_label.shape[0], test_label.shape[1])
    # print('logits:', logits > 0, logits.shape)
    label_ham[test_label > 0] = 1
    # print('label_ham:', label_ham.shape)
    # print('y_test:', y_test.shape)
    mif1 = (label_ham * y_test * 2).sum() / (label_ham.sum() + y_test.sum())
    cnt = 0
    sa = subset_accuracy(label_ham, y_test)
    maf1 = macro_f1(label_ham, y_test)
    # print('mif1:', mif1)
    hamnum = (label_ham != y_test).sum() / (y_test.shape[0] * y_test.shape[1])
    # print('acc_cnt:', cnt / num)
    # ap.add(test_label, y_test)
    # print('ap:', ap.value(), ap.value().mean())
    # ham = hamming_loss(y_test, laebl_ham)
    # print('ham:', ham)
    # print('hamnum:', hamnum)
    cov = coverage(test_label, y_test)
    # print('cov:', cov)
    rank = ranking_loss(test_label, y_test)
    # print('rank:', rank)
    one = one_error(test_label, y_test)
    # print('one:', one)
    ap2 = avgp(test_label, y_test)
    # loss_test = crit(test_label, y_test)
    loss_test = 5
    return ap2, loss_test, cnt, hamnum, rank, one, mif1, cov, sa, maf1


def add_test_pre(x_train, y_train, x_test, y_test, model, crit, dim_view):
    # acc_test, loss_test = 0, 0
    test_label = torch.zeros(y_test.shape[0], y_test.shape[1])
    # ap = AveragePrecisionMeter()
    x_fus = []  # 混合测试集和训练集，用于测试
    model.eval()
    for j in range(len(x_test)):
        # print('xtj:', x_train[j].shape)
        # x_fusj = torch.cat((x_train[j], x_test[j]), dim=0)
        x_fus.append(x_test[j])

    feats, g, e_type = construct_graph_k(x_fus, dim_view)

    with torch.no_grad():
        logits, _ = model(feats, g, e_type)  # (1010, 16)
        test_label = logits
    # print('test_label:{}  y_test:{}'.format(test_label.shape, y_test.shape))
    num = 0

    label_ham = torch.zeros(test_label.shape[0], test_label.shape[1])
    # print('logits:', logits > 0, logits.shape)
    label_ham[test_label > 0] = 1
    # print('label_ham:', label_ham.shape)
    # print('y_test:', y_test.shape)
    mif1 = (label_ham * y_test * 2).sum() / (label_ham.sum() + y_test.sum())
    cnt = 0
    sa = subset_accuracy(label_ham, y_test)
    maf1 = macro_f1(label_ham, y_test)
    hamnum = (label_ham != y_test).sum() / (y_test.shape[0] * y_test.shape[1])
    # print('mif1:', mif1)
    # print('acc_cnt:', cnt / num)
    # ap.add(test_label, y_test)
    # print('ap:', ap.value(), ap.value().mean())
    # ham = hamming_loss(y_test, laebl_ham)
    # print('ham:', ham)
    cov = coverage(test_label, y_test)
    # print('cov:', cov)
    rank = ranking_loss(test_label, y_test)
    # print('rank:', rank)
    one = one_error(test_label, y_test)
    # print('one:', one)
    ap2 = avgp(test_label, y_test)
    # loss_test = crit(test_label, y_test)
    loss_test = 5
    return ap2, loss_test, cnt, hamnum, rank, one, mif1, cov, sa, maf1


def test_epoch(train_view, Y_train, x_test, y_test, model, crit, dim_view):
    size = x_test[0].shape[0]  # * max(args.k_fold - 1, 1) // args.batch_size
    num = 1
    # print('size:{} num:{}'.format(size, num))
    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0
    acc_cnt_sum, ham_sum, rank_sum, one_sum = 0, 0, 0, 0
    cov_sum = 0
    mif1_sum = 0
    sa_sum = 0
    maf2_sum = 0
    for i in range(num):
        test_v = []
        l = i * size
        r = min((i + 1) * size, train_view[0].shape[0])
        # print('l:{} r:{}'.format(l,r))
        for j in range(len(x_test)):
            test_v.append(x_test[j][l:r])
        acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = add_test_pre(
            train_view, Y_train, test_v, y_test[l:r], model, crit, dim_view)
        test_loss_sum += loss_test2
        test_acc_sum += acc_test2
        acc_cnt_sum += acc_cnt2
        ham_sum += ham2
        rank_sum += rank2
        one_sum += one2
        mif1_sum += mif12
        cov_sum += cov2
        sa_sum += sa2
        maf2_sum += maf12
        '''
        print('i:', i)
        print('test_acc2:{} test_loss2:{}:'.format(acc_test2, loss_test2))
        print(
            'cnt2:{} ham2:{} rank2:{} one2:{} mif12:{} cov2:{} sa2:{} maf12:{}'.format(acc_cnt2, ham2, rank2, one2,
                                                                                       mif12, cov2, sa2, maf12))
        print('cnt_b:{} ham_b:{} rank_b:{} one_b:{} mif1_b:{} cov_b:{} sa_b:{} maf1_b:{}'.format(acc_cnt_b, ham_b,
                                                                                                 rank_b, one_b,
                                                                                                 mif1_b, cov_b,
                                                                                                 sa_b, maf1_b))
        '''
    k = num
    test_loss_sum /= k
    test_acc_sum /= k
    sa_sum /= k
    acc_cnt_sum /= k
    ham_sum /= k
    rank_sum /= k
    one_sum /= k
    mif1_sum /= k
    cov_sum /= k
    maf2_sum /= k
    return test_acc_sum, test_loss_sum, acc_cnt_sum, ham_sum, rank_sum, one_sum, mif1_sum, cov_sum, sa_sum, maf2_sum


def train_batch(train_view, Y_train, dim_view, model, optimizer, loss_f, test_view, Y_test):
    model.train()
    feat_list, G, e_type = construct_graph_k(train_view, dim_view)
    logits, gm = model(feat_list, G, e_type)  # (1010, 16)
    acc_train = avgp(logits, Y_train)
    loss = loss_f(logits, Y_train) + gm * args.gamma
    # print('loss:{} gm:{}'.format(loss, gm * args.gamma))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = test_epoch(train_view,
                                                                                             Y_train,
                                                                                             test_view,
                                                                                             Y_test, model,
                                                                                             loss_f, dim_view)
    return acc_train, loss, acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12


def train_epoch(train_view, Y_train, dim_view, model, optimizer, loss_f, test_view, Y_test):
    size = train_view[0].shape[0] // args.batch_size
    acc_train, loss = 0, 0
    num = train_view[0].shape[0] // size
    acc_test_b = 0
    acc_cnt_b = 0
    ham_b = 100000
    rank_b = 100000
    one_b = 100000
    mif1_b = 0
    maf1_b = 0
    cov_b = 100000
    sa_b = 0
    loss_test_b = 100000
    for i in range(num):
        train_v = []
        l = i * size
        r = min((i + 1) * size, train_view[0].shape[0])
        # print('l:{} r:{}'.format(l,r))
        for j in range(len(train_view)):
            train_v.append(train_view[j][l:r])

        acc_train_, loss_, acc_test2, loss_test2, acc_cnt2, ham2, rank2, \
            one2, mif12, cov2, sa2, maf12 = train_batch(
            train_v, Y_train[l:r], dim_view, model, optimizer, loss_f, test_view, Y_test)
        acc_test_b = max(acc_test_b, acc_test2)
        acc_cnt_b = max(acc_cnt_b, acc_cnt2)
        ham_b = min(ham_b, ham2)
        rank_b = min(rank_b, rank2)
        one_b = min(one_b, one2)
        mif1_b = max(mif1_b, mif12)
        maf1_b = max(maf12, maf1_b)
        cov_b = min(cov_b, cov2)
        sa_b = max(sa_b, sa2)
        loss_test_b = min(loss_test_b, loss_test2)
        '''
        print('i:', i)
        print('test_acc2:{} test_loss2:{}:'.format(acc_test2, loss_test2))
        print(
            'cnt2:{} ham2:{} rank2:{} one2:{} mif12:{} cov2:{} sa2:{} maf12:{}'.format(acc_cnt2, ham2, rank2, one2,
                                                                                       mif12, cov2, sa2, maf12))
        print('cnt_b:{} ham_b:{} rank_b:{} one_b:{} mif1_b:{} cov_b:{} sa_b:{} maf1_b:{}'.format(acc_cnt_b, ham_b,
                                                                                                 rank_b, one_b,
                                                                                                 mif1_b, cov_b,
                                                                                                 sa_b, maf1_b))
        '''
        acc_train += acc_train_
        loss += loss_
    acc_train /= num
    loss /= num
    return acc_train, loss, acc_test_b, loss_test_b, acc_cnt_b, ham_b, rank_b, one_b, mif1_b, cov_b, sa_b, maf1_b


def batch(train_view, Y_train, test_view, Y_test, dim_view, model, optimizer, k_):
    train_a = []
    train_l = []
    test_a = []
    test_l = []
    t_start = time.time()
    test_cnt = []
    test_ham = []
    test_rank = []
    test_one = []
    for epoch in range(args.end_epochs):
        train_view, Y_train = shuffle(train_view, Y_train)
        test_view, Y_test = shuffle(test_view, Y_test)
        loss_f = nn.MultiLabelSoftMarginLoss()
        acc_train, loss_train, acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = train_epoch(
            train_view, Y_train, dim_view, model, optimizer, loss_f, test_view, Y_test)

        if epoch < 0:
            acc_test, loss_test, acc_cnt, ham, rank, one, mif1, cov, sa, maf1 = 0, 10000, 0, 100000, 100000, 100000, 0, 1000000, 0, 0
            acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = 0, 10000, 0, 100000, 100000, 100000, 0, 1000000, 0, 0
            acc_test3, loss_test3 = 0, 0
        else:
            acc_test, loss_test, acc_cnt, ham, rank, one, mif1, cov, sa, maf1 = add_test_whole(train_view, Y_train,
                                                                                               test_view,
                                                                                               Y_test, model,
                                                                                               loss_f, dim_view)

            # acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12 = add_test_pre(train_view,
            # Y_train,test_view,Y_test, model, loss_f, dim_view) acc_test3, loss_test3= add_test(train_view, Y_train,
            # test_view, Y_test, model, loss_f, dim_view)

        train_a.append(acc_train)
        train_l.append(loss_train)
        test_a.append(acc_test)
        test_l.append(loss_test)
        test_cnt.append(acc_cnt)
        test_ham.append(ham2)
        test_rank.append(rank)
        test_one.append(one)
        if epoch % 1 == 0:
            print('epoch:{}/{}'.format(epoch, args.end_epochs))
            print('test_acc2:{} test_loss2:{}:'.format(acc_test2, loss_test2))
            print(
                'cnt2:{} ham2:{} rank2:{} one2:{} mif12:{} cov2:{} sa2:{} maf12:{}'.format(acc_cnt2, ham2, rank2, one2,
                                                                                           mif12, cov2, sa2, maf12))
            ts = time.time() - t_start
            print('time:{} remain:{}'.format(ts, (args.end_epochs - epoch - 1) * ts / (epoch + 1)))
    plt.clf()
    plt.ylim(0, 1)
    ax1 = plt.plot(k_ * 2 + 0)
    draw(train_a, test_a, pic + '/acc' + str(k_) + '.png')
    plt.clf()
    ax2 = plt.plot(k_ * 2 + 1)
    draw(train_l, test_l, pic + '/loss' + str(k_) + '.png')
    '''
    ax1 = plt.plot(k_ * 6 + 1)
    draw(train_a, test_a, pic + '/acc' + str(k_) + '.png') 
    ax2 = plt.plot(k_ * 6 + 2)
    draw(train_l, test_l, pic + '/loss' + str(k_) + '.png')
    ax3 = plt.plot(k_ * 6 + 3)
    draw(train_l, test_cnt, pic + '/loss' + str(k_) + '.png')
    ax4 = plt.plot(k_ * 6 + 4)
    draw(train_l, test_ham, pic + '/loss' + str(k_) + '.png')
    ax5 = plt.plot(k_ * 6 + 5)
    draw(train_l, test_rank, pic + '/loss' + str(k_) + '.png')
    ax6 = plt.plot(k_ * 6 + 0)
    draw(train_l, test_one, pic + '/loss' + str(k_) + '.png')
    '''

    # print('best epoch:' + str(best_epoch) + '    ' + 'best acc:' + str(best_acc))
    return acc_train, loss_train, acc_test2, loss_test2, acc_cnt2, ham2, rank2, one2, mif12, cov2, sa2, maf12


def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和测试数据，分开放，X_train为训练数据，X_test为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（向下取整）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数 得到测试集的索引
        # print('fold_size:{} idx:{}'.format(fold_size, idx))
        X_part, y_part = X[idx, :], y[idx]  # 只对第一维切片即可
        if j == i:  # 第i折作test
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # 其他剩余折进行拼接 也仅第一维
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_test, y_test


def k_fold(k, data, target):
    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0
    acc_cnt_sum, ham_sum, rank_sum, one_sum = 0, 0, 0, 0
    cov_sum = 0
    mif1_sum = 0
    sa_sum = 0
    # train_view, Y_train, test_view, Y_test, dim_view
    train_a = []
    train_l = []
    test_a = []
    test_l = []
    acc_cnt_l = []
    ham_l, rank_l, one_l, mif1_l, cov_l = [], [], [], [], []
    sa_l = []
    maf1_l = []
    for i in range(k):
        train_view = []
        test_view = []
        dim_view = []
        y_train = None
        y_test = None

        for j in range(len(data)):
            dim_view.append(data[j].shape[1])
        model = RGCN(dim_view, args.nhidden, target.shape[1], args.layers, args.dropout, 2)
        # print('model:', model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        dim_view = []  # 统计视图维度
        for j in range(len(data)):
            print('i:{} k:{} j:{} len:{}'.format(i, k, j, len(data)))
            train_v, y_train, test_v, y_test = get_k_fold_data(k, i, data[j], target)  # 获取第i折交叉验证的训练和验证数据
            print('X_train:{} y_train:{} X_test:{} y_test:{}'.format(
                train_v.shape, y_train.shape, test_v.shape, y_test.shape))
            train_view.append(train_v)
            test_view.append(test_v)
            dim_view.append(test_v.shape[1])
        train_view, y_train = shuffle(train_view, y_train)
        test_view, y_test = shuffle(test_view, y_test)
        # print('train_view:{} y_train:{} test_view:{} y_test:{}'.format(len(train_view), y_train.shape,
        # len(test_view),y_test.shape))
        acc_train, loss_train, acc_test, loss_test, acc_cnt_b, ham_b, rank_b, \
            one_b, mif1_b, cov_b, sa_b, maf1_b = batch(
            train_view,
            y_train,
            test_view,
            y_test,
            dim_view,
            model,
            optimizer,
            i + 2)
        print('i:{} acc_train:{} loss_train:{} acc_test:{} loss_test:{}'.format(i, acc_train, loss_train, acc_test,
                                                                                loss_test))
        train_loss_sum += loss_train
        test_loss_sum += loss_test
        train_acc_sum += acc_train
        test_acc_sum += acc_test
        acc_cnt_sum += acc_cnt_b
        ham_sum += ham_b
        rank_sum += rank_b
        one_sum += one_b
        mif1_sum += mif1_b
        cov_sum += cov_b
        sa_sum += sa_b
        train_a.append(acc_train)
        train_l.append(loss_train)
        test_a.append(acc_test)
        test_l.append(loss_test)
        acc_cnt_l.append(acc_cnt_b)
        ham_l.append(ham_b)
        rank_l.append(rank_b)
        one_l.append(one_b)
        mif1_l.append(mif1_b)
        cov_l.append(cov_b)
        sa_l.append(sa_b)
        maf1_l.append(maf1_b)
    print_std(test_a, 'ap')
    print_std(ham_l, 'hamming loss')
    print_std(rank_l, 'ranking loss')
    print_std(one_l, 'one error')
    print_std(mif1_l, 'micro-F1')
    print_std(cov_l, 'coverage')
    print_std(sa_l, 'subset accuracy')
    print_std(maf1_l, 'macro_f1')
    sa_sum /= k
    train_acc_sum /= k
    test_acc_sum /= k
    train_loss_sum /= k
    test_loss_sum /= k
    acc_cnt_sum /= k
    ham_sum /= k
    rank_sum /= k
    one_sum /= k
    mif1_sum /= k
    cov_sum /= k
    print('train_acc:{} train_loss:{}'.format(train_acc_sum, train_loss_sum))
    print('test_acc:{} test_loss:{}:'.format(test_acc_sum, test_loss_sum))
    print('acc_cnt:{} ham:{} rank:{} one:{} mif1:{} cov:{}'.format(acc_cnt_sum, ham_sum, rank_sum, one_sum, mif1_sum,
                                                                   cov_sum))
    fig = plt.figure(1)  # 如果不传入参数默认画板1
    # 第2步创建画纸，并选择画纸1
    ax1 = plt.subplot(2, 1, 1)
    draw(train_a, test_a, pic + '/acc.png')
    ax2 = plt.subplot(2, 1, 2)
    draw(train_l, test_l, pic + '/loss.png')
    return train_acc_sum, train_loss_sum, test_acc_sum, test_loss_sum, acc_cnt_sum, ham_sum, rank_sum, one_sum


def get_shu(data, indce):
    res = torch.zeros(data.shape[0], data.shape[1])
    for i in range(data.shape[0]):
        res[i] = data[indce[i]]
    return res


def shuffle(x, y):
    # return x,y
    num_ins = x[0].shape[0]
    # print('num_ins:', num_ins)
    indces = torch.randperm(num_ins)
    a = []
    for i in range(len(x)):
        a.append(get_shu(x[i], indces))
    b = get_shu(y, indces)
    return a, b


def draw(x1, x2, path):
    leng = len(x1)
    x_list = np.zeros(leng)
    train_l = np.zeros(leng)
    test_l = np.zeros(leng)
    for i in range(leng):
        x_list[i] = i
        train_l[i] = x1[i]
        test_l[i] = x2[i]
    plt.xlabel('X')
    plt.ylabel('Y')
    # colors1 = '#00CED1'  # 点的颜色
    # colors2 = '#DC143C'
    # area = np.pi * 4 ** 2  # 点面积
    # plt.scatter(x_list, train_l, s=area, c=colors1, alpha=0.4, label='类别A')
    # plt.scatter(x_list, test_l, s=area, c=colors2, alpha=0.4, label='类别B')
    plt.plot(x_list, train_l, color='r', linewidth=1, alpha=0.6, label='train')
    plt.plot(x_list, test_l, color='b', linewidth=1, alpha=0.6, label='test')
    plt.legend()
    plt.savefig(path, dpi=300)


def data_clean(view, target):
    eps = 1e-5
    for i in range(len(view)):
        valid_col = []
        print('less:', (view[i] < 0).sum())
        # ict = torch.zeros(view[i].shape[1], target.shape[1])
        # for j in range(view[i].shape[0]):
        # for k in range(view[i].shape[1]):
        # dict[k][target[]]
        for j in range(view[i].shape[1]):
            if not (view[i][:, j] == 0).sum() == view[i][:, j].shape[0]:
                valid_col.append(j)
            # print('shape:{} {} {}'.format(view[i].shape, view[i][:,j].shape, view[i][0].shape))
            # print(view[i][:,j])
            view[i][:, j] = (view[i][:, j] - view[i][:, j].mean() + eps) / (torch.std((view[i][:, j])) + eps)
            view[i][:, j] -= view[i][:, j].min()
            # view[i][:,j] = view[i][:,j] / (view[i][:,j].max() + eps) + eps
            # print('aft:')
            # print(view[i][:,j])
            if i == 70:
                print('j:{} view:{} shape:{}'.format(j, view[i][:, j], view[i][:, j].shape))
        view[i] = view[i][:, valid_col]
        print('i:{} viewi:{}'.format(i, view[i].shape))
    # return view, target
    # print(view[5][3195])
    # x = input()
    for i in range(len(view)):
        cnt = 0
        cnt_dim = 0
        print(view[i][0, :30])
        print('i:{} pos:{} num:{} per:{}'.format(i, (view[i] == 0).sum(), view[i].shape[0] * view[i].shape[1],
                                                 (view[i] == 0).sum() / (view[i].shape[0] * view[i].shape[1])))
        for j in range(view[i].shape[0]):
            if (view[i][j] == 0).sum() == view[i][j].shape[0]:
                cnt += 1
        for j in range(view[i].shape[1]):
            if (view[i][:, j] == 0).sum() == view[i][:, j].shape[0]:
                cnt_dim += 1
                # print('i:{} j:{} sum:{} len:{}'.format(i, j, (view[i][j] == 0).sum(), view[i][j].shape[0]))
                # print(view[i][j])
        print('i:{} cnt:{} cnt_dim:{}'.format(i, cnt, cnt_dim))
        # for j in range(view[i].shape[1]):
        # cnt = 0
        # for k in range(view[i].shape[0]):
        # if view[i][k][j] == 0:
        # cnt += 1
        # print('i:{} j:{} sum:{} len:{}'.format(i, j, (view[i][:,j] == 0).sum(), view[i][:,j].shape[0]))
        # view[i][:,j] = (view[i][:,j] - view[i][:,j].mean()) / torch.std(torch.from_numpy(view[i][:,j]))
    return view, target


def train():
    print('data load begin')
    # 获得数据和标签
    view, target = data_loader(args.data_root + args.dataset)
    print('data load end')
    target = torch.tensor(target).permute(1, 0)
    print('positive:', target.sum() / (target.shape[0] * target.shape[1]))
    view, target = data_clean(view, target)
    dim_view = []  # 统计视图维度
    for i in range(len(view)):
        dim_view.append(view[i].shape[1])
        view[i] = torch.tensor(view[i])

    # model = RGCN(dim_view, 128, target.shape[1], 2, 0.2, 2)
    # print('model1:', model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # print('view:{} target:{}'.format(len(view), target.shape))

    t_start = time.time()
    best = 0
    print('view:', view[0][0:20], view[0].shape)
    num_k_fold = 1
    for epoch in range(num_k_fold):
        view, target = shuffle(view, target)
        print('view:', view[0][0:20], view[0].shape)
        print('num_k_fold:{}/{}'.format(epoch, num_k_fold))
        # 最顶层的调用，输入训练数据和标签
        train_acc_sum, train_loss_sum, test_acc_sum, test_loss_sum, \
            acc_cnt_sum, ham_sum, rank_sum, one_sum = k_fold(
            args.k_fold, view, target)

        best = max(best, test_acc_sum)
        ts = time.time() - t_start
        print('best:', best)
        print('time:{} remain:{}'.format(ts, (num_k_fold - epoch - 1) * ts / (epoch + 1)))


def test_code():
    # u, v = get_global_edge_new(5, 2)
    # print(u, v)
    '''
    test_shuffle
    n = 4
    len = 3
    x = []
    y = torch.randn(n, 2)
    for i in range(len):
        x.append(torch.randn(n, 4))
        print(x[i])
    print(y)
    a, b = shuffle(x, y)
    for i in range(len):
        print(a[i])
    print(b)
    '''
    x = [
        torch.tensor([[0, 0, 1], [0, 0, 1]]),
        torch.tensor([[0, 1, 1], [0, 1, 1]]),
        torch.tensor([[0, 2, 1], [0, 2, 1]]),
        torch.tensor([[3, 2, 5], [3, 2, 5]]),
        torch.tensor([[4, 2, 6], [4, 2, 6]]),
    ]
    y = torch.tensor([[0, 1],
                      [1, 1],
                      [1, 0],
                      [0, 1],
                      [1, 0]])
    dim_view = [3]
    construct_graph_k(x, dim_view)


if __name__ == '__main__':
    # torch.set_num_threads(3)
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default='emotions.mat', type=str, metavar='N', help='run_data')
    parser.add_argument('--data_root', default='data/', type=str, metavar='PATH',
                        help='root dir')
    parser.add_argument('--word2vec', default='data/voc_glove_word2vec.pkl', type=str, metavar='PATH',
                        help='root dir')
    parser.add_argument('--batch_size', default=16, type=int, help='number of batch size')
    parser.add_argument('--k', default=3, type=int, help='KNN')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--end_epochs', default=100, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--alpha', default=0.9, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nhidden', default=128, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--k_fold', default=5, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--dropout', default=0.2, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--layers', default=2, type=int,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--gamma', default=0, type=float,
                        metavar='H-P', help='n_hidden')
    parser.add_argument('--name', default='', type=str,
                        metavar='H-P', help='n_hidden')
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    args = parser.parse_args()
    pic = 'record'
    if not os.path.exists(pic):
        os.makedirs(pic)
    pic = pic + '/' + args.dataset + '_lr-' + str(args.lr) + '_epoch-' + str(args.end_epochs) + '_k-' + str(
        args.k) + '_nhidden-' + str(args.nhidden) + '_k-flod-' + str(args.k_fold) + '_batch-size-' + str(
        args.batch_size) + '_gamma-' + str(args.gamma) + '_' + args.name
    if not os.path.exists(pic):
        os.makedirs(pic)
    sys.stdout = open(pic + '/console.txt', 'w')
    print('train begin')
    train()
    # test_code()
    sys.stdout.close()
    #
