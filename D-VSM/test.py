import torch
import scipy.io as scio
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import random
import itertools

# import h5py


# print(b)
datasetFile = "D:/LVCM/D-vsm-dataset/D-vsm-dataset/Corel5k.mat"
# data = h5py.File(datasetFile)
data = scio.loadmat(datasetFile)

train_view = []
test_view = []


'''for i in range(5):
    train_name = 'train_view' + str(i + 1)
    train_feats = torch.nn.functional.normalize(torch.from_numpy(data[train_name] / 1.0).type(torch.float32), p=2,
                                                dim=1)
    print("train feats : ",train_feats)
    test_name = 'test_view' + str(i + 1)
    test_feats = torch.nn.functional.normalize(torch.from_numpy(data[test_name] / 1.0).type(torch.float32), p=2, dim=1)
    print("test feats : ",test_feats)
    train_view.append(train_feats)
    test_view.append(test_feats)
Y_train = data['Y_train']
print(Y_train)
Y_test = data['Y_test']
print(Y_test)
data = torch.from_numpy(data)
partial_target = torch.from_numpy(partial_target)
target = torch.from_numpy(target)

test_index = np.array(random.sample(range(1122), 112))
train_index = np.delete(np.arange(1122), test_index)

train_data = data[train_index]

frac = round(1122 * 0.1)

print(target[0][10])
label_feats = torch.eye(partial_target.shape[0])

data_norm = torch.nn.functional.normalize(data, p=2, dim=1)
label_feats_norm = torch.nn.functional.normalize(label_feats, p=2, dim=1)

print(data_norm)

ins_list = []
label_list = []
for i, label in enumerate(partial_target.permute(1, 0)):
    label = label.numpy()
    index = np.argwhere(label > 0)
    ins = np.array(i).repeat(len(index)).reshape(-1, 1)
    ins_list.append(ins.tolist())
    label_list.append(index.tolist())

a = list(itertools.chain.from_iterable(ins_list))
b = list(itertools.chain.from_iterable(label_list))

ins_idx_C = torch.tensor(a).type(torch.long)
label_idx_C = torch.tensor(b).type(torch.long)
edge_index_C = torch.cat((ins_idx_C, label_idx_C), dim=1)

print(edge_index_C)
'''

print('ahahah')
# data = dataset['data']
# partial_target = dataset['partial_target']
# target = dataset['target'].T

# a = target.argmax(1).reshape(-1,1)


print('hahaha')
#

