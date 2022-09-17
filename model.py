import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class Generator(nn.Module):

    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout):
        super().__init__()
        self.gc1 = GCNConv(
            1,
            1,
        )
        self.gc2 = GCNConv(
            1,
            1,
        )

        self.dropout = dropout
        self.decode_layer = nn.Linear(hidden_dim2, feat_dim)

    def encode(self, x, adj):
        x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def decode(self, x):
        x = self.decode_layer(x)
        return x

    def forward(self, x, adj):

        adj = torch.stack(torch.nonzero(adj, as_tuple=True), dim=0)
        gcn_list = []
        for i in range(x.shape[0]):
            x_ = x[i, :, :].T
            x_ = self.encode(x_, adj)
            gcn_list.append(x_)
        x = torch.stack(gcn_list, dim=1)
        x.transpose_(1, 2).transpose_(0, 2)
        out = self.decode(x)
        return out


class Conv(nn.Module):

    def __init__(self, c1, c2, k):
        super(Conv, self).__init__()
        self.c = nn.Conv1d(c1, c2, k, 1, int(k // 2))
        self.a = nn.ReLU()

    def forward(self, x):
        return self.a(self.c(x))


class GRU_Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, hn = self.gru(x)
        return hn[-1]


# 初始化GRU模型 95%,对应main_test1


class Discriminator(nn.Module):

    def __init__(self, Config):
        super().__init__()
        # cs = [1, 128, 64, 32, 64]
        # ls = []
        # for i in range(1, 4):
        #     ls.append(Conv(cs[i - 1], cs[i], 5))
        # self.m = nn.Sequential(*ls)
        self.g = GRU_Model(Config.input_dim, Config.hidden_dim,
                           Config.layer_dim, Config.output_dim)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.m(x)
        # print(f"x from cnn {x.shape = }")
        print(f"{x.shape = }")
        x = x.transpose(1, 2)
        x = self.g(x)
        # out = self.sigmoid(x)
        x = F.softmax(x)
        return x
