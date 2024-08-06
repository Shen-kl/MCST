import torch
import torch.nn as nn


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.beta + self.alpha * x


class MLP(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(2 * dim, eps=1e-6)
        # self.norm = Affine(2 * dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rest_parameter()

    def rest_parameter(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        cell_list = nn.ModuleList([])
        for i in range(layer_num):
            cell_list.append(
                MLP(hidden_dim, dropout_rate)
            )
        self.mlp = cell_list
        self.rest_parameter()

    def rest_parameter(self):
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x_linear = self.linear1(x)
        for layer in self.mlp:
            x_output = layer(x_linear)
            x_linear = x_linear + x_output
        x = self.linear2(x_output)
        return x