import torch
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from RGCN import RGCN
import dgl
import argparse
from util import evaluate, data_loader2, get_global_edge
import time
import math

"""train_view: 用于训练的多个视图数据，类型是列表，每个元素是一个矩阵（5011行，100列）。
Y_train: 训练数据的标签，类型是矩阵，形状为(ntrain, nlabel)，其中ntrain是训练数据的样本数，nlabel是标签的数量。
test_view: 用于测试的多个视图数据，类型是列表，每个元素是一个矩阵（5011行，100列）。
Y_test: 测试数据的标签，类型是矩阵，形状为(ntest, nlabel)，其中ntest是测试数据的样本数。
dim_view: 视图的维度，类型是列表，包含每个视图的维度信息。
model: 模型对象，用于训练和优化。
optimizer: 优化器对象，用于更新模型参数。"""


def batch(train_view, Y_train, test_view, Y_test, dim_view, model, optimizer):
    print('len of train_view:', len(train_view))
    for i in range(len(train_view)):
        print('train view i:{} shape:{}'.format(i, train_view[i].shape))
    print('Y_train_shape:', Y_test.shape)
    print('test_view:', len(test_view))
    for i in range(len(test_view)):
        print('i:{} shape:{}'.format(i, test_view[i].shape))
    print('Y_train_shape:', Y_train.shape)
    print('dim_view:', dim_view)
    ntrain, ntest, nlabel, nview = Y_train.shape[0], Y_test.shape[0], Y_train.shape[1], len(dim_view)
    g_list = []
    test_g_list = []
    feat_list = []
    test_feat_list = []
    # 对每个视图进行迭代
    for i in range(len(train_view)):
        print('----------------------------------------view:', i)
        view_feats = train_view[i]  # (3999,x)
        test_feat = test_view[i]
        feat_list.append(view_feats)
        test_feat_list.append(test_feat)

        ins_num, fea_num = view_feats.size()  # 5011 100
        test_ins_num, test_fea_num = test_feat.size()

        # train data
        data_dist = pairwise_distances(view_feats, metric='euclidean')  # (5011, 5011)
        topk_dist, topk_index = torch.from_numpy(data_dist).topk(dim=1, k=args.k, largest=False,
                                                                 sorted=True)  # (5011,5)
        print('topk_dist:{} {}'.format(topk_dist[0], topk_dist.shape))
        print('topk_index:{} {}'.format(topk_index[0], topk_index.shape))
        knn_idx = topk_index.reshape(-1, 1).type(torch.long)  # 5011x5=25055, 1
        ins_idx = torch.from_numpy(np.array(list(range(ins_num))).repeat(args.k).reshape(-1, 1)).type(
            torch.long)  # (25055,1) n的全排列顺序出现k次
        print('ins_idx:', ins_idx[0:20])
        # 生成变索引
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
        print('edges:', edges)
        g = dgl.graph(edges)
        print('g:', g)
        print('Y_train:', Y_train.shape)
        # Y_train = torch.tensor(Y_train)
        g.ndata['label'] = Y_train.float()  # [[0], [0], [0], [0], [1], [1], [1],
        # [1], [1], [1], [1]
        g_list.append(g)

        # test
        test_edges = test_edge_idx_ins[0].int(), test_edge_idx_ins[1].int()
        test_g = dgl.graph(test_edges)
        test_g.ndata['label'] = Y_test.float()
        print("test_g.ndata['label'].shape : ",test_g.ndata['label'].shape)
        test_g_list.append(test_g)
    """dgl.batch 函数的作用是将一组图数据（图列表）合并成一个大图。
    这在处理批量图数据时非常有用，可以提高计算效率并实现批处理的功能。合并后的大图 G 具有更高的内存利用效率，并且可以在其上进行更高效的图计算"""
    G = dgl.batch(g_list)
    print('G1:', G)
    print('ntrain:{} nview:{}'.format(ntrain, nview))
    u, v = get_global_edge(ntrain)

    u, v = torch.tensor(u, dtype=torch.int32), torch.tensor(v, dtype=torch.int32)
    # print('u:{} {}'.format(u.shape, u[0:20]))
    # print('v:{} {}'.format(v.shape, v[0:20]))
    # 在大图G中加入刚刚建立好的不同视图、同实例完全图的边
    G.add_edges(u, v)
    e_type = torch.zeros(G.number_of_edges())
    e_type[G.edge_ids(u, v).long()] = 1
    print('G2:', G)
    # test
    test_G = dgl.batch(test_g_list)
    test_u, test_v = get_global_edge(ntest)
    test_u, test_v = torch.tensor(test_u, dtype=torch.int32), torch.tensor(test_v, dtype=torch.int32)
    # 加入刚刚建立好的不同视图、同实例完全图的边
    test_G.add_edges(test_u, test_v)
    test_e_type = torch.zeros(test_G.number_of_edges())
    test_e_type[test_G.edge_ids(test_u, test_v).long()] = 1

    best_acc = 0
    best_epoch = 0
    print('feat_list_shape:', len(feat_list))
    for i in range(len(feat_list)):
        print('i:{} shape:{}', i, feat_list[i].shape)
    print('G3:', G)
    print('e_type_shape', e_type.shape)
    # print('Y_train0:', Y_train[0])
    # aaa = input()
    t_start = time.time()
    for epoch in range(args.end_epochs):
        print('epoch:{}/{}'.format(epoch, args.end_epochs))
        model.train()
        pred = model(feat_list, G, e_type)  # (5011, 20)

        print('pred[0]:', pred[0].shape)
        # print('Y_train0:', Y_train[0])
        loss1 = 0
        for i in range(5):
            l_ = i * 3999
            r_ = (i + 1) * 3999
            print('i:{} l:{} r:{}'.format(i, l_, r_))
            epoch_loss = torch.nn.BCEWithLogitsLoss(pred[0])
            loss1 += ((1 - torch.tensor(Y_train)) * pred[l_:r_]).sum()
        # loss2 = (-pred*torch.log(pred)).sum()
        loss = loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            acc, pred_index = evaluate(model, test_feat_list, test_G, test_e_type, torch.tensor(Y_test).argmax(1))
            if best_acc < acc:
                best_acc = acc
                best_epoch = epoch
                beat_pred_index = pred_index
            print('epoch: ' + str(epoch) + '  loss:' + str(loss.item()) + '    acc:' + str(acc))
            ts = time.time() - t_start
            print('time:{} remain:{}'.format(ts, (args.end_epochs - epoch) * ts / (epoch + 1)))

    print('best epoch:' + str(best_epoch) + '    ' + 'best acc:' + str(best_acc))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='Corel5k.mat', type=str, metavar='N', help='run_data')
# parser.add_argument('--data_root', default='/app/users/wangyiyuan/TRUST/D-VSM/', type=str, metavar='PATH',
#                   help='root dir')
parser.add_argument('--data_root', default='D:/LVCM/D-vsm-dataset/D-vsm-dataset/', type=str, metavar='PATH',
                  help='root dir')
parser.add_argument('--word2vec', default='data/voc_glove_word2vec.pkl', type=str, metavar='PATH',
                    help='root dir')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--k', default=5, type=int, help='KNN')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end_epochs', default=10, type=int, metavar='H-P',
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--alpha', default=0.9, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--nhidden', default=256, type=int,
                    metavar='H-P', help='n_hidden')

args = parser.parse_args()
# mat = sio.loadmat('D:/LVCM/D-vsm-dataset/D-vsm-dataset/Corel5k.mat')
# print(sorted(mat.keys()))
# print(mat)
# x = input()
train_view, Y_train, test_view, Y_test, dim_view = data_loader2(args.data_root + args.dataset)
print('train_view:', len(train_view))
for i in range(len(train_view)):
    print('i:{} shape:{}'.format(i, train_view[i].shape))
print('Y_test_shape:', Y_test.shape)
print('test_view:', len(test_view))
for i in range(len(test_view)):
    print('i:{} shape:{}'.format(i, test_view[i].shape))
print('Y_train_shape:', Y_train.shape)
print('dim_view:', dim_view)


# 指定model
model = RGCN(dim_view, 128, Y_train.shape[1], 2, .2, 2)
print("cuda available : " , torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('model:', model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
batch_size = 128
batch_num = math.floor(train_view[0].shape[0] / batch_size)
print('batch_num:', batch_num)
batch(train_view, Y_train, test_view, Y_test, dim_view, model, optimizer)


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
