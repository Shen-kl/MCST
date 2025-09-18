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
    """一个标准的两层 MLP：Linear -> Activation -> Dropout -> Norm"""
    def __init__(self, dim, hidden_dim=None, dropout=0.1, activation=nn.LeakyReLU):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = activation(0.1, inplace=False)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)  # 残差 + 归一化
        return x


class ResMLP(nn.Module):
    """输入 -> Linear1 -> 多层 Residual MLP -> Linear2 -> 输出"""
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=4, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.mlp_blocks = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, dropout_rate) for _ in range(layer_num)
        ])
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        for block in self.mlp_blocks:
            x = block(x)  # 每层 MLP 内部已经包含残差
        x = self.linear2(x)
        return x
