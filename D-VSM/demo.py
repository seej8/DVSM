import torch
from torch._prims import gt

from util import  weighted_loss,data_loader, generate_cross_graph
from GNBlock_model import _Model
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='Pascal07_5view.mat', type=str, metavar='N', help='run_data')
parser.add_argument('--data_root', default='N:/Dataset/multi-view/', type=str, metavar='PATH',
					help='root dir')
parser.add_argument('--word2vec', default='N:/Dataset/voc/voc_glove_word2vec.pkl', type=str, metavar='PATH',
					help='root dir')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--k', default=5, type=int, help='KNN')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('--end_epochs', default=500, type=int, metavar='H-P',
					help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float,
					metavar='H-P', help='initial learning rate')
parser.add_argument('--alpha', default=0.9, type=float,
					metavar='H-P', help='initial learning rate')


args = parser.parse_args()

graph_ins_list, graph_label, Y_train, test_view, Y_test, dim_view= data_loader(args.data_root+args.dataset, args.k, args.word2vec)


model = _Model(dim_view)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)

if torch.cuda.is_available():
	model = model.cuda()

for epoch in range(args.start_epoch, args.end_epochs):
	optimizer.zero_grad()
	view_cross_edges = graph_label["kwargs1"]  # [25022,2]
	global_cross_edges = graph_label["kwargs2"] # [100220,2]
	# counts = torch.tensor(graph_ins["kwargs"])

	view_cross_edges = view_cross_edges.type(torch.long).transpose(1, 0)  # (2, 2504) #(2,5008)
	global_cross_edges = global_cross_edges.type(torch.long).transpose(1, 0)

	prediction = model(graph_ins_list, view_cross_edges, global_cross_edges) #view_graph_cross:(2,25055) global_cross_edges: (2, 100220)
	# print('prediction:', prediction)

	pred_init = torch.zeros(counts[0],counts[1])

	for j in range(prediction.shape[0]):
		pred_init[edge_index_cross[0][j], edge_index_cross[1][j]] = prediction[j]

	# compute loss
	loss = weighted_loss(pred_init,args.alpha)
	print('loss:', torch.true_divide(loss, 1122))
	loss.backward()

	optimizer.step()

	# train accu
	pred_value, pred_max_index = torch.max(pred_init, dim=1)

	_, gt_index = torch.max(gt, dim=0)

	accu = torch.true_divide(torch.sum(pred_max_index == gt_index), 1122)
	print('============================================================================accu:', accu)


	#test accu
	model.eval()

	# Training mode
	model.train()
	torch.set_grad_enabled(True)



