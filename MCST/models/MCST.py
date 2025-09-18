import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import config
from models.AttentionMechanism import AdditiveAttention
from models.MLP import ResMLP
from typing import Tuple
import torch.nn.functional as F

class Updater(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropoutrate,
                 time_series_len,
                 device
                 ):
        super(Updater, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropoutrate = dropoutrate

        self.embed = nn.Linear(in_features, int(hidden_features))
        self.embed_detection = nn.Linear(int(in_features) * time_series_len, int(hidden_features * in_features))

        self.linear1 = nn.Linear(int(hidden_features), int(out_features))

        self.linear2 = nn.Linear(int(hidden_features), 1)
        self.linear3 = nn.Linear(int(hidden_features), 1)

        self.softplus = nn.Softplus()

        rank = max(1, hidden_features // 4)
        self.U_proj = nn.Linear(self.in_features + self.out_features, hidden_features * rank)
        self.V_proj = nn.Linear(self.in_features + self.out_features, rank * hidden_features)

        self.mlp_sigma = ResMLP(int(hidden_features * 1), int(hidden_features),
                            int(hidden_features * 1), 8, dropoutrate)

        self.mlp_detection = ResMLP(int(hidden_features * 1 ), int(hidden_features),
                            int(hidden_features * 1), 4, dropoutrate)

        self.mlp_detection_sigma = ResMLP(int(hidden_features * 1), int(hidden_features),
                            int(hidden_features * 1), 4, dropoutrate)

        self.layer_norm = nn.LayerNorm(int(hidden_features), eps=1e-6).to(device)

        self.linear4 = nn.Linear(int(hidden_features), int(hidden_features))
        self.linear5 = nn.Linear(int(hidden_features), out_features)
        self.leakyrelu = nn.LeakyReLU(0.01, inplace=False)

        # 可学习的 W 缩放因子，初始较小，防止 W 一开始过大
        self.register_parameter("w_scale", nn.Parameter(torch.tensor(0.1)))

        self.rest_parameters()

    def rest_parameters(self):
        nn.init.kaiming_normal_(self.embed.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.embed_detection.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear4.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_normal_(self.linear5.weight)

    def forward(self, x: torch.Tensor, x_sigma_log: torch.Tensor, detection: torch.Tensor):

        # location_sample_input = torch.cat([location_sample_reshape, location_sigma_repeat], dim=2)
        residuals = (x[:, -1, 0::2] - detection[:, -1, :]).unsqueeze(dim=1)
        residuals_embedding = self.leakyrelu(self.embed(residuals))
        detection_embedding = self.leakyrelu(self.embed_detection(detection.reshape([detection.shape[0],-1]))).reshape([detection.shape[0], self.in_features, -1])


        # 观测值
        detection_sigma_log = torch.log(self.softplus(self.linear3(detection_embedding + 
                                                                   self.mlp_detection_sigma(detection_embedding)))+1e-5).\
            squeeze(dim=2).unsqueeze(dim=1)


        x_sigma_log_clamped = torch.clamp(x_sigma_log, min=-10.0, max=10.0)
        det_sigma_log_clamped = torch.clamp(detection_sigma_log, min=-10.0, max=10.0)

        s1 = torch.exp(0.5 * det_sigma_log_clamped)  # (B,1,hidden)
        s2 = torch.exp(0.5 * x_sigma_log_clamped)    # (B,1,hidden)

        # 构造用于计算增益的输入
        mlp_input =  torch.cat([s1, s2], dim=-1)

        rank = max(1, self.hidden_features // 4)
        U = self.U_proj(mlp_input).reshape(-1, self.hidden_features, rank)
        V = self.V_proj(mlp_input).reshape(-1, rank, self.hidden_features)

        # 生成 增益 W
        W = torch.bmm(U, V)  # [B, H, H]

        # Frobenius norm 归一化 + 可学习缩放
        W = W / (W.norm(2, [1, 2], True) + 1e-6)  # 保持稳定
        W = self.w_scale * W

        x_trans = torch.einsum("bij,bjk->bik", residuals_embedding, W)

        # layer norm + nonlinearity
        x_trans = self.layer_norm(x_trans.squeeze(1)).unsqueeze(1)  # (B,1,hidden)
        x_trans = self.leakyrelu(self.linear4(x_trans.squeeze(1))).unsqueeze(1)  # (B,1,hidden)

        # final mapping
        x_pred_delta = torch.tanh(self.linear5(x_trans.squeeze(1))).unsqueeze(1)  # (B,1,out_features)

        # residual connection to previous x
        # ensure x[:,-1,:] shape matches out_features
        last_x = x[:, -1, :].unsqueeze(1)  # (B,1,out_features)
        x_predict = x_pred_delta + last_x

        x_sigma_log = torch.log(self.softplus(self.linear1(residuals_embedding + self.mlp_sigma(residuals_embedding)))+1e-5)

        return x_predict, x_sigma_log, detection_sigma_log


class dual_stage_attention(nn.Module):
    def __init__(self, hidden_features):
        super(dual_stage_attention, self).__init__()
        self.hidden_features = hidden_features
        self.linear1 = nn.Linear(hidden_features, int(hidden_features * 2))
        self.norm1 = nn.LayerNorm(int(hidden_features * 2))  # 正则化 防止过拟合
        self.leakyrelu = nn.LeakyReLU(0.5, inplace=False)

        self.attention_time = AdditiveAttention(int(hidden_features), int(hidden_features), int(hidden_features * 2),
                                           dropout=0.1)
        self.attention_sample_points = AdditiveAttention(int(hidden_features * 2), int(hidden_features * 2),
                                                         int(hidden_features * 2), dropout= 0.1)

        self.rest_parameter()

    def rest_parameter(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.5)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, lstm_output: torch.Tensor, batch_size: int, sample_num: int):
        attention_time_output = self.attention_time(lstm_output, lstm_output,
                                             lstm_output[:, -1, :].unsqueeze(dim=1))

        attention_time_output = attention_time_output.permute(1, 0, 2).view(batch_size, sample_num, -1).transpose(1,0)
        attention_time_output_linear = self.leakyrelu(self.norm1(self.linear1(attention_time_output.transpose(0,1))))
        attention_sample_points_output = self.attention_sample_points(attention_time_output_linear,attention_time_output_linear,
                                                           attention_time_output_linear[:,0,:].unsqueeze(dim=1))

        return attention_sample_points_output

class ManuverCompensationUnit(nn.Module):
    def __init__(self, fft_point):
        super(ManuverCompensationUnit, self).__init__()
        self.fft_point = fft_point

        self.gate = ResMLP(self.fft_point * 2, self.fft_point,
                            self.fft_point, 2, 0.1)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(fft_point)
    def forward(self, innovationError: torch.Tensor):

        innovationError_reshape = innovationError.permute(0, 2, 3, 1).reshape(innovationError.shape[0],-1,innovationError.shape[1])
        innovationError_frequency_spectrum = torch.fft.fft(innovationError_reshape, n=self.fft_point, dim=-1)
        freq_features = torch.cat([innovationError_frequency_spectrum.real,
                                   innovationError_frequency_spectrum.imag], dim=-1)
        gate = self.sigmoid((self.gate(freq_features))) # 缩放到[0,1]之间

        innovationError_frequency_spectrum_save = innovationError_frequency_spectrum * gate
        manuver_compensation = torch.fft.ifft(innovationError_frequency_spectrum_save, n=self.fft_point, dim=-1).real
        manuver_compensation_downsample = manuver_compensation[:, :, :innovationError_reshape.shape[2]]
        manuver_compensation_downsample = manuver_compensation_downsample.reshape(innovationError.shape[0],
                                                                                  innovationError.shape[2],
                                                                                  innovationError.shape[3],
                                                                                  -1)
        return manuver_compensation_downsample.permute(0, 3, 1, 2)


class MCU(nn.Module):
    def __init__(self, MCU_layer, MCU_hidden_dim):
        super(MCU, self).__init__()
        cell_list = nn.ModuleList([])
        for i in range(MCU_layer):
            cell_list.append(
                ManuverCompensationUnit(MCU_hidden_dim)
            )

        self.manuverCompensationLayer = cell_list

    def forward(self,normalized_detection_MCU: torch.Tensor, normalized_update_history_MCU: torch.Tensor):
        # # 残差 保留高频分量 高频分量意味着机动残差
        innovationError = (normalized_detection_MCU\
                          - normalized_update_history_MCU[:,:,0::2]).unsqueeze(dim=3)
        x = innovationError
        manuver_compensation_input_list = []
        manuver_compensation_output_list = []
        manuver_compensation_input_list.append(innovationError)
        layer_index = 0
        for layer in self.manuverCompensationLayer:
            output = layer(x)
            manuver_compensation_output_list.append(output)
            x = x - output

        manuver_compensation = torch.stack(manuver_compensation_output_list, dim=0).sum(0)

        return manuver_compensation

class Predictor(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropoutrate,
                 num_layers,
                 sample_num,
                 MCU_layer,
                 MCU_hidden_features,
                 device
                 ):
        super(Predictor, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropoutrate = dropoutrate
        self.num_layers = num_layers
        self.sample_num = sample_num
        self.device = device

        self.x_embed = nn.Linear(int(in_features),
                                 int(hidden_features))

        self.manuverWeight_embed = nn.Linear(int(in_features/2),
                                 int(hidden_features))

        self.linear1 = nn.Linear(sample_num, 1)
        self.linear2 = nn.Linear(hidden_features * self.num_layers * 2, out_features)
        self.linear3 = nn.Linear(hidden_features * self.num_layers * 2, out_features)

        self.softplus = nn.Softplus()

        self.bilstm = nn.LSTM(input_size=hidden_features,
                             hidden_size=hidden_features,
                             num_layers=self.num_layers,
                             bidirectional=True,
                             batch_first=True)

        self.mlp_x = ResMLP(int(hidden_features * 4), int(hidden_features),
                                out_features, 2, 0.1)

        self.mlp_sigma = ResMLP(int(hidden_features * self.num_layers * 2), int(hidden_features),
                                int(hidden_features * self.num_layers * 2), 2, 0.1)

        self.MCU = MCU(MCU_layer, MCU_hidden_features)
        self.dual_stage_attention = dual_stage_attention(hidden_features * self.num_layers)
        self.linear4 = nn.Linear(hidden_features * 2, hidden_features)

        self.norm1 = nn.LayerNorm(int(hidden_features * 4))  # 正则化 防止过拟合
        self.norm2 = nn.LayerNorm(hidden_features * self.num_layers * 2)  # 正则化 防止过拟合

        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)

        self.rest_parameters()

    def rest_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.kaiming_normal_(self.x_embed.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.manuverWeight_embed.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear4.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, update_sigma: torch.Tensor, xUpdatePrevious_norm: torch.Tensor, normalized_detection_MCU: torch.Tensor,
                normalized_update_history_MCU: torch.Tensor, hidden_states: Tuple[torch.Tensor, torch.Tensor]):
        batch, seq_len, _ = xUpdatePrevious_norm.size()
        #采样
        xUpdatePrevious_norm = self.unscented_transform(xUpdatePrevious_norm, update_sigma)
        #MCU 补偿
        manuver_compensation = self.MCU(normalized_detection_MCU, normalized_update_history_MCU)
        #LSTM
        xPrevious_embedding = self.leakyrelu(self.x_embed(xUpdatePrevious_norm.permute(0,3,1,2).reshape(-1,xUpdatePrevious_norm.shape[1],xUpdatePrevious_norm.shape[2])))
        manuver_weight_embed = self.leakyrelu(self.manuverWeight_embed(manuver_compensation.permute(0,3,1,2).reshape(-1,manuver_compensation.shape[1],manuver_compensation.shape[2])))
        manuver_weight_embed = manuver_weight_embed[:,-seq_len: ,:].unsqueeze(dim=3).repeat(1, 1, 1, self.sample_num).permute(0,3,1,2).reshape(-1,seq_len,manuver_weight_embed.shape[2]) #

        lstm_input = torch.cat([xPrevious_embedding, manuver_weight_embed], dim=2)
        lstm_input = F.layer_norm(lstm_input, lstm_input.shape[2:])
        lstm_input = F.dropout(lstm_input, p=0.1)
        lstm_input_tensor = self.leakyrelu(self.linear4(lstm_input))

        output, (h_predict, c_predict) = self.bilstm(lstm_input_tensor, hidden_states)

        # dual-stage attention
        attention_output = self.dual_stage_attention(output, batch, self.sample_num)

        # 计算输出
        pre = self.mlp_x(self.norm1(attention_output))  # (B, T, C)
        x_predict = self.bounded_identity_grad(pre)

        # x_predict = torch.tanh(self.mlp_x(self.norm1(attention_output)).permute(0,2,1)).permute(0,2,1)
        x_sigma = self.softplus(self.leakyrelu(self.linear3(self.norm2(self.mlp_sigma(attention_output)))))
        x_sigma_log = torch.log(x_sigma)

        return x_predict, x_sigma_log, (h_predict, c_predict)

    def init_hidden(self, input_size):
        return torch.zeros(self.num_layers * 2, input_size, self.hidden_features, requires_grad=True, device=self.device)

    def init_cell(self, input_size):
        return torch.zeros(self.num_layers * 2, input_size, self.hidden_features, requires_grad=True, device=self.device)

    def unscented_transform(self, xUpdatePrevious_norm, update_sigma):
        xUpdatePrevious_norm = xUpdatePrevious_norm.unsqueeze(dim=3).repeat(1, 1, 1, self.sample_num)
        update_sigma = update_sigma.unsqueeze(dim=3).repeat(1, 1, 1,self.sample_num)
        xUpdatePrevious_norm = xUpdatePrevious_norm + torch.sqrt(torch.exp(update_sigma)) * \
                               torch.cat([torch.zeros([xUpdatePrevious_norm.shape[0],xUpdatePrevious_norm.shape[1],
                                                       xUpdatePrevious_norm.shape[2],1]).to(xUpdatePrevious_norm.device),
                                          torch.randn(xUpdatePrevious_norm.shape[0],xUpdatePrevious_norm.shape[1],
                                                               xUpdatePrevious_norm.shape[2],xUpdatePrevious_norm.shape[3]-1).
                                         to(xUpdatePrevious_norm.device)],dim=3)
        return xUpdatePrevious_norm
    def bounded_identity_grad(self, x):
        y = torch.tanh(x)
        return x + (y - x).detach()

# 引入残差序列 求子相关 FFT后得到功率谱密度 作为机动检测 （机动后目标残差不再是高斯白噪声 功率谱密度不再是常数）
class ManeuverCompensationStrongTracker(nn.Module):
    def __init__(self,
                 predictor_in_features: int,
                 predictor_hidden_features: int,
                 predictor_out_features: int,
                 predictor_dropoutrate,
                 predictor_num_layers,
                 predictor_sample_num,
                 predictor_MCU_layer,
                 predictor_MCU_hidden_features,
                 updater_in_features,
                 updater_hidden_features,
                 updater_out_features,
                 updater_dropoutrate,
                 time_series_len,
                 device
                 ):
        super(ManeuverCompensationStrongTracker, self).__init__()
        self.predictor = Predictor(predictor_in_features,
                                     predictor_hidden_features,
                                     predictor_out_features,
                                     predictor_dropoutrate,
                                     predictor_num_layers,
                                     predictor_sample_num,
                                     predictor_MCU_layer,
                                     predictor_MCU_hidden_features,
                                     device)
        self.updater = Updater(updater_in_features,
                               updater_hidden_features,
                               updater_out_features,
                               updater_dropoutrate,
                               time_series_len,
                               device)

    def forward(self, update_sigma, normalized_detection, normalized_update_history, normalized_detection_MCU, normalized_update_history_MCU,
                hidden_states_encoder, detection_flag=0):
        # 预测
        x_predict, x_predict_sigma_log, hidden_states_encoder \
            = self.predictor(update_sigma, normalized_update_history, normalized_detection_MCU,
                             normalized_update_history_MCU, hidden_states_encoder)

        if detection_flag == 0:
            # 滤波  检测到点迹
            update_input = torch.cat([normalized_update_history[:, 1:, :], x_predict], dim=1)
            detection_input = normalized_detection[:, 1:, :]
            output_update, output_update_sigma, detection_sigma_log,  \
                = self.updater(update_input, x_predict_sigma_log, detection_input)
        else:
            # 滤波  未检测到点迹
            update_input = torch.cat([normalized_update_history[:, 1:, :], x_predict], dim=1)
            detection_input = torch.cat([normalized_detection[:, 1:-1, :], x_predict[:,:,0::2]], dim=1)
            output_update, output_update_sigma, detection_sigma_log \
                = self.updater(update_input, x_predict_sigma_log, detection_input)


        return x_predict, x_predict_sigma_log, hidden_states_encoder,\
               output_update, output_update_sigma, detection_sigma_log