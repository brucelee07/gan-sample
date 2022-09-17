import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,
                                                  out_features))  # W
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)  # 初始化

    def forward(self, input, adj):
        # input就是x，对应的是特征矩阵
        # adj对应的是A，就是邻接矩阵

        print(f"{ input.shape = }")
        print(f"{ self.weight.shape = }")
        support = torch.bmm(input, self.weight)  # XW
        output = torch.spmm(adj, support)  # A(XW)  spmm: 稀疏矩阵的乘法
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'
