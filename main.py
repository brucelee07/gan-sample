import random
import warnings

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import Dataset
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)

from model import Generator, Discriminator
from utils import normalize, sparse_mx_to_torch_sparse_tensor
from gcn_data import getdata
from config import Data_Fold, Config

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seed = 2022

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


class Data_Set(Dataset):

    def __init__(self, x_data):
        self.data = x_data

    def __len__(self):
        length = self.data.shape[0] - Config.window_length - 1
        return length

    def __getitem__(self, index):
        X = self.data[index:index + Config.window_length, :]
        return X


def load_data():

    sc = Normalizer()

    traindata = np.loadtxt(Data_Fold / 'training.csv',
                           dtype=float,
                           delimiter=',')
    # NOTE 数据不均衡 class 1 只占 0.1
    x_train = sc.fit_transform(traindata[:, 1:]).astype("float32")
    # x_label = traindata[:, 0].astype('int32')

    testdata = np.loadtxt(Data_Fold / 'test.csv', dtype=float, delimiter=',')
    x_test = sc.fit_transform(testdata[:, 1:]).astype("float32")
    # y_label = testdata[:, 0].astype('int32')

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)

    train_dataset = Data_Set(x_train)
    test_dataset = Data_Set(x_test)

    return train_dataset, test_dataset,


def gae_for():

    adj, _ = getdata()
    adj = adj + np.eye(adj.shape[0])
    adj = torch.from_numpy(adj)

    train_dataset, test_data_set = load_data()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=Config.batch_size,
                                               shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                              batch_size=Config.batch_size,
                                              shuffle=False,
                                              drop_last=True)

    gen = Generator(
        Config.input_dim,
        Config.hidden1,
        Config.hidden2,
        Config.dropout,
    ).to(DEVICE)

    dis = Discriminator(Config).to(DEVICE)

    loss_entropy = torch.nn.CrossEntropyLoss().to(DEVICE)
    loss_mse = torch.nn.MSELoss().to(DEVICE)

    optimizer_gen = optim.Adam(gen.parameters(),
                               lr=Config.lr,
                               betas=(0.9, 0.999))
    optimizer_dis = optim.Adam(dis.parameters(),
                               lr=Config.lr,
                               betas=(0.9, 0.999))

    gen.train()
    dis.train()

    loss_gs = []
    loss_ds = []
    print("-" * 30 + "开始训练" + "-" * 30)
    for epoch in range(Config.epochs):

        train_preds = []
        train_trues = []

        G_loss = 0.0
        D_loss = 0.0

        count = len(train_loader)

        for data in train_loader:
            # 生成器生成伪造数据
            data = data.to(DEVICE)
            noise = torch.rand_like(data).to(DEVICE)
            noise = data * .5 + noise * .5

            adj = adj.to(DEVICE)

            fake = gen(x=noise, adj=adj)
            fake = fake.transpose(0, 1).unsqueeze(1)

            fake_output = dis(fake.detach())  # 生成器损失函数

            real_output = dis(data)

            dis_real_loss = loss_entropy(
                real_output, torch.ones_like(real_output.shape[0])) * .5
            dis_fake_loss = loss_entropy(
                fake_output, torch.zeros_like(fake_output.shape[0]))

            dis_loss = dis_real_loss + dis_fake_loss
            optimizer_dis.zero_grad()
            dis_loss.backward()
            optimizer_dis.step()

            fake_gen = dis(fake)
            gen_fake_loss = loss_entropy(
                fake_gen, torch.ones_like(fake_gen.shape[0])) * 0.2
            gen_mse_loss = loss_mse(fake, data) * 3
            gen_loss = gen_fake_loss + gen_mse_loss
            optimizer_gen.zero_grad()
            gen_loss.backward()
            optimizer_gen.step()

            #
            G_loss += gen_loss
            D_loss += dis_loss

            train_outputs = torch.argmax(fake_gen, dim=1)
            train_preds.extend(train_outputs.detach().cpu().numpy())

        sklearn_accuracy = accuracy_score(train_trues, train_preds)
        f1 = f1_score(np.zeros_like(train_preds), train_preds, zero_division=1)
        loss_g = G_loss / count
        loss_d = D_loss / count
        loss_gs.append(loss_g)
        loss_ds.append(loss_d)

        print("Epoch:{} G_loss:{:.5f} D_loss:{:.5f} accuracy:{:.5f} f1:{:.5f}".
              format(epoch, loss_g, loss_d, sklearn_accuracy, f1))

    return
    print("-" * 30 + "开始测试" + "-" * 30)
    test_preds = []
    test_trues = []
    gen.eval()
    dis.eval()

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            data_1 = data.squeeze().transpose(0, 1)
            recovered, mu, logvar, f = gen(x=data_1, adj=adj_norm)
            input = f.transpose(0, 1).unsqueeze(1)

            output = dis(torch.cat((data, input), 0))
            score_tmp = output
            pred = output.data.max(1, keepdim=True)[1]
            test_preds.extend(pred.squeeze().detach().cpu().numpy())
            test_trues.extend((torch.cat((target, target),
                                         0)).detach().cpu().numpy())
            score_list.extend(score_tmp.detach().cpu().numpy())
            label_list.extend((torch.cat((target, target), 0)).cpu().numpy())
    sklearn_accuracy = accuracy_score(test_trues, test_preds)
    sklearn_precision = precision_score(test_trues,
                                        test_preds,
                                        average='macro')
    sklearn_recall = recall_score(test_trues, test_preds, average='macro')
    sklearn_f1 = f1_score(test_trues, test_preds, average='macro')
    print(classification_report(test_trues, test_preds))
    print(
        "[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}"
        .format(sklearn_accuracy, sklearn_precision, sklearn_recall,
                sklearn_f1))

    # ##ROC曲线绘制###
    score_list = np.array(score_list)
    fpr, tpr, thresholds_keras = roc_curve(test_trues, score_list[:, 1])
    v1 = auc(fpr, tpr)
    print("AUC : ", v1)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(v1))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig("ROC.png")
    plt.show()

    # ###PR曲线####
    num_class = 2
    score_array = np.array(score_list)
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)
    print("label_onehot:", label_onehot.shape)

    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(num_class):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = average_precision_score(
            label_onehot[:, i], score_array[:, i])
        print(precision_dict[i].shape, recall_dict[i].shape,
              average_precision_dict[i])

    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(
        label_onehot.ravel(), score_array.ravel())
    average_precision_dict["micro"] = average_precision_score(label_onehot,
                                                              score_array,
                                                              average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.4f}'.
          format(average_precision_dict["micro"]))
    plt.figure()
    plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.4f}'
        .format(average_precision_dict["micro"]))
    plt.savefig("pr.jpg")
    plt.show()

    # 绘制 gen_loss, dis_loss
    plt.plot(loss_gs, linewidth='1', label="gen_loss", color='#FF4500')
    plt.plot(loss_ds, linewidth='1', label="disc_loss", color='#0000CD')
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("生成对抗网络损失函数图片.png")
    plt.show()


if __name__ == '__main__':
    rr = gae_for()
