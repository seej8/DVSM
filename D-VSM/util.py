import numpy as np
import torch
from torch_geometric.data import Dataset, Data
import torch.utils.data as data
import random
import pickle
import math
from build_graph import Graph_Generator
import scipy.io as scio
from sklearn.metrics import label_ranking_loss, label_ranking_average_precision_score, coverage_error


def print_std(list, name):
    x = torch.tensor(list)
    print('{}:{}±{}'.format(name, x.mean(), torch.std(x)))


def macro_f1(predict, target):
    cnt = 0
    mul = (predict * target).sum(0) * 2
    su = predict.sum(0) + target.sum(0)
    for i in range(predict.shape[1]):
        if mul[i] > 0:
            cnt += mul[i] / su[i]
    return cnt / predict.shape[1]
    # return 2 * ((predict * target).sum(0) / (predict.sum(0) + target.sum(0))).sum() / predict.shape[1]

    for j in range(predict.shape[1]):
        ans = 0
        div = 0
        for i in range(predict.shape[0]):
            ans += predict[i][j] * target[i][j] * 2
            div += predict[i][j] + target[i][j]
        if div > 0:
            cnt += ans / (div)
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt / predict.shape[1]


def subset_accuracy(predict, target):
    return ((predict == target).sum(1) == target.shape[1]).sum() / predict.shape[0]
    cnt = 0
    for i in range(predict.shape[0]):
        flag = True
        for j in range(predict.shape[1]):
            if predict[i][j] != target[i][j]:
                flag = False
                break
        if flag:
            cnt += 1
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt / predict.shape[0]


def ranking_loss(output, target):
    return label_ranking_loss(target, output)
    cnt = 0
    for i in range(output.shape[0]):
        ans = 0
        positvive, negative = [], []
        for j in range(output.shape[1]):
            if target[i][j] == 1:
                positvive.append(j)
            else:
                negative.append(j)
        for j in range(len(positvive)):
            for k in range(len(negative)):
                # print('j:{} {} k:{} {}'.format(j, output[i][positvive[j]], k, output[i][negative[k]]))
                if output[i][positvive[j]] <= output[i][negative[k]]:
                    ans += 1
        if ans > 0:
            cnt += ans / (len(positvive) * len(negative))
    cnt /= output.shape[0]
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt


def avgp(output, target):
    return label_ranking_average_precision_score(target.detach(), output.detach())
    cnt = 0
    for i in range(output.shape[0]):
        ans = 0
        rank = torch.zeros(output.shape[1])
        rank_label = []
        for j in range(output.shape[1]):
            for k in range(output.shape[1]):
                if output[i][k] >= output[i][j]:
                    rank[j] += 1
            if target[i][j] == 1:
                rank_label.append(rank[j])
        rank_label.sort()
        for j in range(len(rank_label)):
            ans += (j + 1) / rank_label[j]
        if ans > 0:
            cnt += ans / len(rank_label)
    cnt /= output.shape[0]
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt


def coverage(output, target):
    return coverage_error(target, output) - 1
    cnt = 0
    for i in range(output.shape[0]):
        ans = 0
        rank = torch.zeros(output.shape[1])
        num = 0
        for j in range(output.shape[1]):
            for k in range(output.shape[1]):
                if output[i][k] >= output[i][j]:
                    rank[j] += 1
            if target[i][j] == 1:
                ans = max(ans, rank[j])
                num += 1
        cnt += ans - 1
    cnt /= output.shape[0]
    # cnt /= output.shape[1] #归一化
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt


def one_error(logits, labels):
    _, indices = torch.max(logits, dim=1)
    # print('logits:', logits, logits.shape)
    # print('indices:', indices, indices.shape)
    # print('label_i', labels)
    # print('labels:', labels[indices])
    cnt1 = 0
    for i in range(logits.shape[0]):
        cnt1 += (labels[i][indices[i]] == 0)
    # print('cnt1:', cnt1.item())
    return cnt1.item() / logits.shape[0]
    # return cnt1
    # correct = torch.sum(indices == labels)
    cnt = 0
    for i in range(logits.shape[0]):
        mx = 0
        loc = 0
        for j in range(logits.shape[1]):
            if logits[i][j] > mx:
                mx = logits[i][j]
                loc = j
        if labels[i][loc] == 0:
            cnt += 1
        # print('i:{} loc:{} indices:{}'.format(i, loc, indices[i]))
    cnt /= logits.shape[0]
    print('cnt1:{} cnt:{}'.format(cnt1, cnt))
    return cnt


def evaluate(model, feats, g, e_type, labels):
    model.eval()
    with torch.no_grad():
        logits = model(feats, g, e_type)  # (1010, 16)
        _, indices = torch.max(logits, dim=1)  # 1010
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), indices


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count > 0:
            precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1


def data_loader2(data_path):
    datasetFile = data_path
    data = scio.loadmat(datasetFile)
    train_view = []
    test_view = []
    dim_view = []
    for i in range(5):
        # 获取每一个视图下实例的总数，划分训练集和测试集
        ins_feat = data['data'][i][0].astype(float)
        total_ins = ins_feat.shape[0]
        train_ins_amount = int(total_ins * 0.8)
        test_ins_amount = int(total_ins * 0.2)
        train_feats = torch.from_numpy(ins_feat).type(torch.float32)
        # print(type(train_feats))
        train_feats = torch.nn.functional.normalize(train_feats[:train_ins_amount, :], p=2,
                                                    dim=1)
        # print(train_feats.shape)
        test_feats = torch.from_numpy(ins_feat).type(torch.float32)
        test_feats = torch.nn.functional.normalize(test_feats[train_ins_amount:, :], p=2,
                                                   dim=1)
        print(test_feats.shape)
        train_view.append(train_feats)
        test_view.append(test_feats)
        dim_view.append(train_feats.shape[1])
    print("train_view shape : ", train_view[0].shape)
    print("train_view len : ", len(train_view))
    print("test_view shape : ", test_view[0].shape)
    print("test_view len : ", len(test_view))
    Y_train = torch.transpose(torch.tensor(data['target'][:, :train_ins_amount]), 0, 1)
    print("Y_train : ", Y_train.shape)
    Y_test = torch.transpose(torch.tensor(data['target'][:, train_ins_amount:]), 0, 1)
    print("Y_test : ", Y_test.shape)

    return train_view, Y_train, test_view, Y_test, dim_view


train_view, y_train, test_view, y_test, dim_view = data_loader2("D:/LVCM/D-vsm-dataset/D-vsm-dataset/Corel5k.mat")


def data_loader(data_path, k, word2vecFile):
    datasetFile = data_path
    data = scio.loadmat(datasetFile)
    train_view = []
    test_view = []
    dim_view = []
    for i in range(5):
        train_name = 'train_view' + str(i + 1)
        train_feats = torch.nn.functional.normalize(torch.from_numpy(data[train_name] / 1.0).type(torch.float32), p=2,
                                                    dim=1)
        test_name = 'test_view' + str(i + 1)
        test_feats = torch.nn.functional.normalize(torch.from_numpy(data[test_name] / 1.0).type(torch.float32), p=2,
                                                   dim=1)
        train_view.append(train_feats)
        test_view.append(test_feats)
        dim_view.append(train_feats.shape[1])

    Y_train = data['Y_train']
    Y_test = data['Y_test']

    label_feats = pickle.load(open(word2vecFile, 'rb'))
    label_feats = torch.from_numpy(label_feats)

    graph_generator = Graph_Generator()
    graph_ins_list, graph_label = graph_generator(train_view, label_feats, Y_train, k)

    return graph_ins_list, graph_label, Y_train, test_view, Y_test, dim_view


# def data_loader(data_path, k):
#     graph_generator = Graph_Generator()
#     dataset = scio.loadmat(data_path)
#
#     ins_feats = torch.nn.functional.normalize(torch.from_numpy(dataset['data']).type(torch.float32), p=2, dim=1)
#     partial_target = torch.from_numpy(dataset['partial_target'])  #(16,1122)
#     target = torch.from_numpy(dataset['target'])
#     label_feats = torch.eye(partial_target.shape[0])

#     graph_ins, graph_label = graph_generator(ins_feats, label_feats, partial_target, k)
#     return graph_ins, graph_label,target

def weighted_loss(prediction, a):  # (1122,16)

    # loss = (1-torch.sum(prediction.pow(2), dim=1))+a*(1-torch.sum(prediction, dim=1))
    # loss = (1 - torch.sum(prediction.pow(2), dim=1))

    loss = 0
    for i in range(prediction.shape[0]):
        cand_label = prediction[i] > 0
        cand_value = prediction[i][cand_label]
        cand_softmax = torch.nn.functional.softmax(cand_value)
        if i == 0:
            print('cand_softmax0:', cand_softmax.data)
        if i == 10:
            print('--------------------------------------------------------cand_softmax10:', cand_softmax.data)
        loss += (1 - torch.sum(cand_softmax.pow(2), dim=0))

    return loss


class GraphData(Dataset):
    def __init__(self, data_path, k):
        super(GraphData, self).__init__()
        self.graph_generator = Graph_Generator()
        dataset = scio.loadmat(data_path)
        self.k = k

        self.ins_feats = torch.from_numpy(dataset['data']).type(torch.float32)
        self.partial_target = torch.from_numpy(dataset['partial_target'])
        self.target = torch.from_numpy(dataset['target'])

        self.label_feats = torch.eye(16)

    def __len__(self):
        'Denotes the total number of samples'
        return 1

    def __getitem__(self, index):
        'Generates one sample of data'

        graph_ins, graph_label = self.graph_generator(self.ins_feats, self.label_feats, self.partial_target, self.k)
        return graph_ins, graph_label


def generate_cross_graph(view_cross_edges, global_cross_edges):
    view_cross_edges = view_cross_edges.type(torch.long).transpose(1, 0)  # (2, 2504) #(2,5008)
    global_cross_edges = global_cross_edges.type(torch.long).transpose(1, 0)

    if torch.cuda.is_available():
        view_graph_cross = Data(x=None, edge_index=view_cross_edges.cuda(),
                                edge_attr=None,
                                y=None)
    else:
        global_cross_edges = Data(x=None, edge_index=global_cross_edges.type(torch.long), edge_attr=None,
                                  y=None)

    return view_graph_cross, global_cross_edges


class dataloader(data.Dataset):
    def __init__(self, data_path):
        dataset = scio.loadmat(data_path)
        # print(type(dataset['data']))

        target = dataset['target']
        # print(type(data[0]))
        # print(data[5][0].shape)
        # print(target.shape)
        self.ins_feats = torch.from_numpy(dataset['data'][5][0])
        # self.partial_target = torch.from_numpy(dataset['partial_target'])
        self.target = torch.from_numpy(dataset['target'])

        # self.label_feats = torch.eye(self.partial_target.shape[0])
        self.label_feats = torch.eye(self.target.shape[0])

        # print('[dataset] the number of instance=%d' % (self.partial_target.shape[1]))

    def __getitem__(self, index):
        # return (self.ins_feats, self.label_feats, self.partial_target), self.target
        return (self.ins_feats, self.label_feats), self.target

    def __len__(self):
        return self.ins_feats.shape[0]


def get_global_edge_test(nins, nview):
    u = []
    v = []
    for i in range(nins):
        for j in range(nview):
            for k in range(j + 1, nview, 1):
                u.append(i * nins + j)
                v.append(i * nins + k)
    u1 = u.copy()
    v1 = v.copy()
    u.extend(v1)
    v.extend(u1)
    return u, v


def get_global_edge(node_num):  # 不同视图的同一个节点之间建立完全图
    u, v = [], []
    lis = [[] for i in range(5)]
    for i in range(len(lis)):
        lis[i] = [j + i * node_num for j in range(node_num)]

    for e1, e2, e3, e4, e5 in zip(lis[0], lis[1], lis[2], lis[3], lis[4]):
        u.append(e1)
        v.append(e2)
        u.append(e1)
        v.append(e3)
        u.append(e1)
        v.append(e4)
        u.append(e1)
        v.append(e5)

        u.append(e2)
        v.append(e3)
        u.append(e2)
        v.append(e4)
        u.append(e2)
        v.append(e5)

        u.append(e3)
        v.append(e4)
        u.append(e3)
        v.append(e5)

        u.append(e4)
        v.append(e5)
    u1 = u.copy()
    v1 = v.copy()
    u.extend(v1)
    v.extend(u1)
    return u, v
