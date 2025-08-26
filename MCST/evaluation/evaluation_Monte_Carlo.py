import torch
import torch.nn as nn
import time

from sympy import false

from data.load_data import *
from utils.tracksManage import TARTRACKS
import numpy as np
import config
import matplotlib.pyplot as plt
from data.load_data import DataLoadFromMatlab_oneTarget_3D_for_paper
from utils.MinMaxScaler import MyMinMaxScaler
from models.MCST import ManeuverCompensationStrongTracker
from statistics import mean
from typing import Callable

Monte_Carlo = 1
def error_caculation(predict_output: np.double, update_output: np.double, state_labels: np.double):
    # 函数功能： 计算误差结果 输出位置 速度上的误差

    # 定义位置和速度索引
    pos_idx = [0, 2, 4]
    vel_idx = [1, 3, 5]
    label_pos_idx = [0, 3, 6]
    label_vel_idx = [1, 4, 7]

    start_idx = args.predictor_time_series_len

    def _calc_rmse(output: np.double, pos_or_vel_idx: list, label_idx: list):
        diffs = [(output[0, :, 0, i] - state_labels[0, start_idx:, 0, j]) ** 2
                 for i, j in zip(pos_or_vel_idx, label_idx)]
        return np.sqrt(sum(diffs))

    predict_error = _calc_rmse(predict_output, pos_idx, label_pos_idx)
    predict_vel_err = _calc_rmse(predict_output, vel_idx, label_vel_idx)
    update_error = _calc_rmse(update_output, pos_idx, label_pos_idx)
    update_vel_err = _calc_rmse(update_output, vel_idx, label_vel_idx)

    return predict_error, predict_vel_err, update_error, update_vel_err

def normalize(tarTracks: TARTRACKS, detections, minMaxScaler: Callable, minMaxScaler_MCU: Callable, frame_index, args, epoch_index):
    # 函数功能： 归一化
    update_history = torch.cat(tarTracks.x_update_history, dim=1).clone()
    input_sigma = torch.cat(tarTracks.x_sigma, dim=1).detach()

    def _pad_sequence(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """对长度不足的序列补零"""
        pad_len = target_len - tensor.shape[1]
        if pad_len > 0:
            pad_shape = [tensor.shape[0], pad_len] + list(tensor.shape[2:])
            padding = torch.zeros(pad_shape, device=tensor.device)
            tensor = torch.cat([padding, tensor], dim=1)
        return tensor

    # 对LSTM 输入归一化
    begin_index = max(frame_index - args.predictor_time_series_len, 0)
    input_sigma = input_sigma[:, begin_index:, :]
    (normalized_detections, normalized_update_history,
     min_vals, max_vals) = \
        minMaxScaler(detections[:, begin_index: frame_index + 1, :, :],
                     update_history[:, begin_index:, :], args.T, (args.max_velocity, args.max_acceleration), mode="-1_1", frame_index=frame_index, epoch_index=epoch_index)

    # 对MCU 输入归一化
    begin_MCU_index = max(frame_index - args.predictor_MCU_len, 0)
    (normalized_detections_MCU, normalized_update_history_MCU,
     min_vals, max_vals) = \
        minMaxScaler_MCU(detections[:, begin_MCU_index: frame_index, :, :],
                         update_history[:, begin_MCU_index:, :], args.T, (args.max_velocity, args.max_acceleration), mode="-1_1", frame_index=frame_index, epoch_index=epoch_index)

    # 对不足长度序列进行补零
    normalized_detections = _pad_sequence(normalized_detections, args.predictor_time_series_len)
    normalized_update_history = _pad_sequence(normalized_update_history, args.predictor_time_series_len)
    input_sigma = _pad_sequence(input_sigma, args.predictor_time_series_len)

    normalized_detections_MCU = _pad_sequence(normalized_detections_MCU, args.predictor_MCU_len)
    normalized_update_history_MCU = _pad_sequence(normalized_update_history_MCU, args.predictor_MCU_len)

    # 移除冗余维度
    normalized_detections = normalized_detections.squeeze(dim=2)
    normalized_detections_MCU = normalized_detections_MCU.squeeze(dim=2)


    return normalized_update_history, normalized_update_history_MCU, normalized_detections, normalized_detections_MCU,\
        input_sigma

# 使用六维状态向量
def evaluation(model, args):
    state_labels, detections = DataLoadFromMatlab_oneTarget_3D_for_paper()

    minMaxScaler = MyMinMaxScaler(use_max_velocity=False, train_mode=False)
    minMaxScaler_MCU = MyMinMaxScaler(use_max_velocity=False, train_mode=False)  # 归一化方法

    detections = detections.to(args.device)  # (batch, n_frames_Ob, ob_num_max, 3)
    state_labels = state_labels.to(args.device)  # (batch, n_frames_state_labels, tg_num_max, 9)

    detections = detections.to(torch.float32)
    state_labels = state_labels.to(torch.float32)

    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict'])

    # 模型设置
    model.eval()

    predict_output = []  # 存放预测结果
    update_output = []  # 存放滤波结果

    # 初始化类数组
    tarTracks = TARTRACKS()  # 假设1个目标
    tarTracks.track_init(args, detections, args.eval_batch_size)

    h_predict = model.predictor.init_hidden(args.predictor_sampling_num * args.eval_batch_size)

    c_predict = model.predictor.init_cell(args.predictor_sampling_num * args.eval_batch_size)

    label_tmp = []
    for frame_index in range(args.predictor_time_series_len, detections.shape[1]):  #开始当前sequence
        # 对张量进行最小-最大归一化
        normalized_update_history, normalized_update_history_MCU, normalized_detections, normalized_detections_MCU, \
        input_sigma = normalize(tarTracks, detections, minMaxScaler, minMaxScaler_MCU, frame_index, args, 0)

        predict_output_this_frame = []
        update_output_this_frame = []

        # 预测
        output_normalized_predict, output_predict_sigma, (h_predict, c_predict), \
        output_normalized_update, output_update_sigma, \
        output_detection_predict_sigma \
            = model(input_sigma, normalized_detections,
                    normalized_update_history.to(args.device), normalized_detections_MCU,
                    normalized_update_history_MCU.to(args.device),
                    (h_predict, c_predict),np.random.rand(1) >= 0.7)
        # 更新 predict update
        predict_output_data = minMaxScaler.deMinMaxScaler(
            output_normalized_predict.unsqueeze(dim=2)).squeeze(
            dim=2)
        update_output_data = minMaxScaler.deMinMaxScaler(output_normalized_update.unsqueeze(dim=2)).squeeze(
            dim=2)

        # 更新历史
        tarTracks.x_sigma.append(output_update_sigma)
        tarTracks.x_update_history.append(update_output_data)

        # 存储结果
        predict_output_this_frame.append(predict_output_data.unsqueeze(dim=1))  # (batch, 1, 1, state_dim)
        predict_output_this_frame_cat = torch.cat(predict_output_this_frame, dim=2)
        predict_output.append(predict_output_this_frame_cat)  # predict_output_this_frame (batch, 1, n_trg, state_dim)

        update_output_this_frame.append(update_output_data.unsqueeze(dim=1))  # (batch, 1, 1, state_dim)
        update_output_this_frame_cat = torch.cat(update_output_this_frame, dim=2)
        update_output.append(update_output_this_frame_cat)  # update_output_this_frame (batch, 1, n_trg, state_dim)

    predict_output_cat = torch.cat(predict_output, dim=1)  # (batch, frame_total, n_trg, state_dim)

    update_output_cat = torch.cat(update_output, dim=1)  # (batch, frame_total, n_trg, state_dim)

    update_output_cat_cpu = update_output_cat.cpu().detach().numpy()
    predict_output_cat_cpu = predict_output_cat.cpu().detach().numpy()

    state_labels_cpu =state_labels.cpu().numpy()

    #   计算误差
    predict_error, predict_vel_err, update_err, update_vel_err =\
        error_caculation(predict_output_cat_cpu, update_output_cat_cpu, state_labels_cpu)
    return predict_error, predict_vel_err, update_err, update_vel_err


if __name__ == '__main__':
    args = config.Args().get_parser()
    # setup_seed(args.seed)
    # 模型
    track_model = ManeuverCompensationStrongTracker(args.predictor_in_features,
                                                     args.predictor_hidden_features,
                                                     args.predictor_out_features,
                                                     args.dropout_prob,
                                                     args.predictor_lstm_num_layers,
                                                     args.predictor_sampling_num,
                                                     args.predictor_MCU_layer,
                                                     args.predictor_MCU_hidden_features,
                                                     args.updater_in_features,
                                                     args.updater_hidden_features,
                                                     args.updater_out_features,
                                                     args.updater_dropoutrate,
                                                     args.predictor_time_series_len,
                                                     args.device).to(args.device)

    predict_loc_err_Monte_Carlo = []
    predict_vel_err_Monte_Carlo = []
    update_loc_err_Monte_Carlo = []
    update_vel_err_Monte_Carlo = []
    for index in range(Monte_Carlo):
        print('Monte Carlo index: %d\n' %(index))
        predict_error, predict_vel_err, update_err, update_vel_err = evaluation(track_model, args)
        predict_loc_err_Monte_Carlo.append(predict_error)
        predict_vel_err_Monte_Carlo.append(predict_vel_err)
        update_loc_err_Monte_Carlo.append(update_err)
        update_vel_err_Monte_Carlo.append(update_vel_err)

    predict_error_Monte_Carlo_Mean = sum(predict_loc_err_Monte_Carlo)/len(predict_loc_err_Monte_Carlo)
    predict_vel_err_Monte_Carlo_Mean = sum(predict_vel_err_Monte_Carlo)/len(predict_vel_err_Monte_Carlo)
    update_err_Monte_Carlo_Mean = sum(update_loc_err_Monte_Carlo)/len(update_loc_err_Monte_Carlo)
    update_vel_err_Monte_Carlo_Mean = sum(update_vel_err_Monte_Carlo)/len(update_vel_err_Monte_Carlo)


    predict_error_rmse = np.mean(predict_error_Monte_Carlo_Mean ** 2)**0.5
    predict_vel_err_rmse = np.mean(predict_vel_err_Monte_Carlo_Mean ** 2) ** 0.5
    update_err_rmse = np.mean(update_err_Monte_Carlo_Mean ** 2) ** 0.5
    update_vel_err_rmse = np.mean(update_vel_err_Monte_Carlo_Mean ** 2) ** 0.5

    print(f'predict_error_rmse:{predict_error_rmse} m predict_vel_err_rmse:{predict_vel_err_rmse} m/s '
          f'update_err_rmse:{update_err_rmse} m update_vel_err_rmse:{update_vel_err_rmse}')
