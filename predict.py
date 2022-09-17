import argparse
import time
import IEEE14
import numpy as np
import scipy.sparse as sp
import torch
from scipy import sparse
from torch import optim
# import model
from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
import pandas as pd

adj, features = IEEE14.getdata()

data = pd.read_excel('3节点攻击0.5度.xlsx', header=None)  # 默认读取第一个sheet
data = data.dropna(axis=1)
my_array = np.array(data).T
my_tensor = torch.tensor(my_array, dtype=torch.float32)
my_tensor = my_tensor / 100
features1 = my_tensor

features1 = features1[:, 1728:]

n_nodes, feat_dim = features1.shape
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
# edge coords
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train
adj_norm = preprocess_graph(adj)

# model = GCNModelVAE(feat_dim, 32, 16, 0, feat_dim=feat_dim)

# model1=model.load_state_dict(torch.load('m1.pt'))
# model.load_state_dict(torch.load('m2.pt'))

model = torch.load('m1.pt')
# model=model.to(device)#或者下面这个
model.eval()

recovered, mu, logvar, f = model(x=features1, adj=adj_norm)  # recovered 重构的模型
f_df = pd.DataFrame(100 * f.detach().numpy().T)
out = f_df.tail(1728)
out = out.values
print(out)
print("----------------------------------")

def zuocha(out):
    data = pd.read_excel('3节点攻击0.5度.xlsx')  # 默认读取第一个sheet
    # print(data)
    data = data.tail(1728)
    data = data.dropna(axis=1)
    data = data.values
    s = out - data
    print("作差结果")
    print(s)

    s = pd.DataFrame(s)
    s.to_csv('frame1.csv', header=None, index=None)

zuocha(out)
