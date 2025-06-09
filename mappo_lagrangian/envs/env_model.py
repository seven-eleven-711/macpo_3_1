import torch.nn.functional as F
from torch import nn
import torch


# 定义注意力机制网络
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            # nn.Tanh(),
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
        # attention_weights = F.softmax(encoded, dim=1)
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

    def forward(self, input1, input2):
        attention_output = self.attention(input2)
        combined_input = torch.cat([input1, attention_output])
        output = self.model(combined_input)
        return output

