import os
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import mean
from torch import nn
import torch.utils.data as data
from tensorboardX import SummaryWriter


def tran_tensor(a, b, c, d, e, f):  # 将每一行数据转化为张量

    x_train_main = torch.Tensor(np.array(a))
    x_train_aux = torch.Tensor(np.array(b))
    y_train = torch.Tensor(np.array(c))

    x_test_main = torch.Tensor(np.array(d))
    x_test_aux = torch.Tensor(np.array(e))
    y_test = torch.Tensor(np.array(f))

    return x_train_main, x_train_aux, y_train, x_test_main, x_test_aux, y_test


# 定义注意力机制网络
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=input_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Linear(input_dim, 1)  # 解码器输出为1维的线性层

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted_x = attention_weights * x
        output = self.decoder(weighted_x)
        return output


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim1, input_dim2):
        super(NeuralNetwork, self).__init__()
        self.attention = AttentionLayer(input_dim2)
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim1 + 1, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=8),
            nn.Tanh(),
            nn.Linear(in_features=8, out_features=4),
            nn.Tanh(),
            nn.Linear(in_features=4, out_features=1),
            nn.Sigmoid()
        )
        # # 使用He初始化
        # for layer in self.model[:18]:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_normal_(layer.weight)
        #
        # # 使用Xavier初始化
        # if isinstance(self.model[-2:-1], nn.Linear):
        #     nn.init.xavier_normal_(self.model[-1].weight)

    def forward(self, input1, input2):
        attention_output = self.attention(input2)
        combined_input = torch.cat([input1, attention_output], dim=1)
        output = self.model(combined_input)
        return output


def test(model_t, criterion, test_main_x, test_aux_x, test_y):

    outputs = model_t(test_main_x, test_aux_x)
    loss_t = criterion(outputs, test_y)

    print(loss_t)


def train(model_t, criterion, optimizers, epochs, para_name, train_x_main, train_x_aux, train_y_):
    # 转换成dataset
    torch_dataset = data.TensorDataset(train_x_main, train_x_aux, train_y_)

    # 把dataset放入DataLoader
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=800,  # 每批提取的数量
        shuffle=True,  # 打乱数据
        num_workers=0  # 读取数据线程
    )

    average_loss = []
    loss_s = []
    for epoch in range(epochs):
        for step, (batch_xm, batch_xa, batch_y) in enumerate(loader):

            outputs = model_t(batch_xm, batch_xa)
            loss = criterion(outputs, batch_y)
            loss_s.append(float(loss.detach().numpy()))

            # 清零梯度，计算loss，反向传播
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
        average_loss.append(mean(loss_s))
        loss_s.clear()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, average_loss[epoch]))
    plt.figure()
    plt.plot(average_loss)
    loss_path = os.path.join('loss_save', str(para_name) + '_loss.png')
    plt.savefig(loss_path)
    # plt.show()
    # plt.close()


def model_train(inputs, next_label):
    input_norm = inputs
    next_norm = next_label

    h_list = {'H1': ['Uv1', 'Uv2', 'Us1', 'Us2', 'H0', 'T01', 'T12'],
              'H2': ['Uv2', 'Uv3', 'Us2', 'Us3', 'H1', 'T12', 'T23'],
              'H3': ['Uv3', 'Uv4', 'Us3', 'Us4', 'H2', 'T23', 'T34'],
              'H4': ['Uv4', 'Uv5', 'Us4', 'Us5', 'H3', 'T34', 'T45'],
              'H5': ['Uv5', 'Us5', 'H4', 'T45', 'T56']
              }

    t_list = {'T01': ['Uv1', 'Us1', 'H0', 'H1', 'T12'],
              'T12': ['Uv1', 'Uv2', 'Us1', 'Us2', 'H0', 'H1', 'T01', 'T12', 'T23'],
              'T23': ['Uv2', 'Uv3', 'Us2', 'Us3', 'H1', 'H2', 'T12', 'T23', 'T34'],
              'T34': ['Uv3', 'Uv4', 'Us3', 'Us4', 'H2', 'H3', 'T23', 'T34', 'T45'],
              'T45': ['Uv4', 'Uv5', 'Us4', 'Us5', 'H3', 'H4', 'T34', 'T45', 'T56'],
              'T56': ['Uv5', 'Us5', 'H4', 'H5', 'T45']
              }
    # 变量列表
    label_list = {**h_list, **t_list}

    for key, value in label_list.items():

        main_param = input_norm[value]
        aux_param = input_norm.drop(columns=value)
        label = next_norm[[key]]

        data_split = random.sample(range(0, 2600), 200)
        data_split.sort()
        x_train_main = main_param.drop(data_split)
        x_train_aux = aux_param.drop(data_split)
        y_train = label.drop(data_split)

        x_test_main = main_param.loc[data_split]
        x_test_aux = aux_param.loc[data_split]
        y_test = label.loc[data_split]

        x_train_main_tensor, x_train_aux_tensor, y_train_tensor, x_test_main_tensor, x_test_aux_tensor, y_test_tensor = \
            tran_tensor(x_train_main, x_train_aux, y_train, x_test_main, x_test_aux, y_test)

        model_net = NeuralNetwork(len(value), 22 - len(value))
        with SummaryWriter(logdir="network_visualization") as w:
            w.add_graph(model_net, [x_train_main_tensor, x_train_aux_tensor])
        # 定义优化器
        optimizer = torch.optim.Adam(model_net.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率
        # 定义目标损失函数

        loss_func = torch.nn.MSELoss(reduction='sum')  # 这里采用均方差函数

        train(model_net, loss_func, optimizer, 1000, key, x_train_main_tensor, x_train_aux_tensor, y_train_tensor)
        test(model_net, loss_func, x_test_main_tensor, x_test_aux_tensor, y_test_tensor)

        model_path = os.path.join('model_save', key + '.pth')
        torch.save(model_net.state_dict(), model_path)


if __name__ == '__main__':
    # 读数据
    input_datas = pd.read_excel('train_data\\cold_rolling_reduce.xlsx', sheet_name='example')
    next_state = pd.read_excel('train_data\\cold_rolling_reduce.xlsx', sheet_name='label')
    # 创建文件夹保存结果
    model_save_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(model_save_path + '\\model_save'):
        os.makedirs(model_save_path + '\\model_save')
    if not os.path.exists(model_save_path + '\\loss_save'):
        os.makedirs(model_save_path + '\\loss_save')
    # 归一化
    data_max = input_datas.apply(lambda x: np.max(x), axis=0)
    data_min = input_datas.apply(lambda x: np.min(x), axis=0)
    data_max.to_excel(excel_writer='train_data\\input_max.xlsx', sheet_name='sheet_1')
    data_min.to_excel(excel_writer='train_data\\input_min.xlsx', sheet_name='sheet_1')
    input_data_norm = input_datas.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0)
    next_data_norm = next_state.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0)
    input_data_norm.to_excel(excel_writer='train_data\\input_norm.xlsx', sheet_name='sheet_1')
    model_train(input_data_norm, next_data_norm)

