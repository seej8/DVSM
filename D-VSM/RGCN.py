import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import RelGraphConv, GATConv
import constr_loss


class RGCN(nn.Module):
    def __init__(self, in_feat, n_hidden, n_class, n_layers, dropout, e_type):
        # model = RGCN(dim_view, 128, target.shape[0], 2, .2, 2)
        super(RGCN, self).__init__()

        self.in_feat = in_feat
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.n_layers = n_layers
        self.dropout = dropout
        self.etype = e_type
        self.encoders = nn.ModuleList()
        self.drop = nn.Dropout(p=self.dropout)
        self.lea = 0.3

        for i in range(len(in_feat)):
            if in_feat[i] <= 128:
                encoder = nn.Sequential(
                    nn.Linear(in_feat[i], n_hidden),
                    nn.LeakyReLU(self.lea),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(n_hidden, n_hidden)
                )
            elif in_feat[i] <= 512:
                encoder = nn.Sequential(
                    nn.Linear(in_feat[i], 256),
                    nn.LeakyReLU(self.lea),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(256, n_hidden)
                )
            elif in_feat[i] <= 1024:
                encoder = nn.Sequential(
                    nn.Linear(in_feat[i], 512),
                    nn.LeakyReLU(self.lea),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(self.lea),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(256, n_hidden)
                )
            else:
                encoder = nn.Sequential(
                    nn.Linear(in_feat[i], 1024),
                    nn.LeakyReLU(self.lea),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(self.lea),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(self.lea),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(256, n_hidden)
                )
            self.encoders.append(encoder)
        # 定义construct encoder
        self.constr_encoder = nn.Sequential(
            nn.Linear(n_class, 1024),
            nn.LeakyReLU(self.lea),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, n_hidden)

        )
        self.encoders.append(self.constr_encoder)

        '''
        self.common = nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(n_hidden, n_hidden)
                )
        '''
        self.RGCNs = nn.ModuleList()
        for i in range(n_layers):
            RGCNLayer = RelGraphConv(n_hidden, n_hidden, e_type)
            self.RGCNs.append(RGCNLayer)
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, 256),
            nn.LeakyReLU(self.lea),
            nn.Dropout(p=self.dropout),
            # nn.Linear(256, 512),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(512, 1024),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.dropout),
            nn.Linear(256, n_class)
        )
        # constr_decoder
        self.constr_decoder = nn.Sequential(
            nn.Linear(n_hidden, 256),
            nn.LeakyReLU(self.lea),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, n_class)
        )
        self.constr_loss = constr_loss.construct_loss()

    def forward(self, feat_list, g, etypes):
        with g.local_scope():
            feat = None
            ndim = len(feat_list)
            for i in range(len(feat_list)):
                flag = self.encoders[i](feat_list[i])
                if i == 0:
                    feat = flag
                else:
                    feat = torch.cat((feat, flag), dim=0)
            feat = self.drop(feat)
            # rel = nn.LeakyReLU(0.2)
            # feat = self.common(feat)
            for layer in self.RGCNs:
                feat = layer(g, feat, etypes.to(torch.int64))
                # feat=self.drop(feat)
                # feat = rel(feat)

            feat = self.drop(feat)
            feat = self.decoder(feat)
            print("feat : " ,feat.shape)
            # 编码解码处理，加loss
            constr_feat = self.constr_encoder(feat)
            constr_feat = self.constr_decoder(constr_feat)
            con_loss = self.constr_loss(constr_feat,feat)
            print("constr_feat : ", constr_feat.shape)

            res = torch.zeros(constr_feat.shape[0] // ndim, self.n_class)
            for i in range(res.shape[0]):
                for j in range(self.n_class):
                    for k in range(ndim):
                        res[i][j] += constr_feat[k * res.shape[0]+i][j]
                    res[i][j] /= ndim
            print("res : ",res.shape)
            print(res)
            return res, con_loss
