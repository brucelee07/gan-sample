import numpy as np
import pandas as pd
import torch
from scipy import sparse

from config import Data_Fold


def getdata():
    df_adj = pd.read_excel(Data_Fold / 'IEEE_54.xlsx', header=None)
    sp_adj = sparse.csr_matrix(df_adj)

    data = pd.read_excel(Data_Fold / '正常数据TS.xlsx',
                         header=None)  # 默认读取第一个sheet
    data = data.dropna(axis=1)
    feat = np.array(data).T
    feat = torch.tensor(feat, dtype=torch.float32)
    feat = feat / 100

    return sp_adj, feat


if __name__ == "__main__":
    adj, feat = getdata()
