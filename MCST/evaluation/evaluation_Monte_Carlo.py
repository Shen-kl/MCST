import torch
import torch.nn as nn
import time
from data.load_data import *
from utils.tracksManage import TARTRACKS
import numpy as np
import config
import matplotlib.pyplot as plt
from data.load_data import DataLoadFromMatlab_oneTarget_3D_for_paper
import mpldatacursor
from utils.MinMaxScaler import MyMinMaxScaler
from models.MCST import ManeuverCompensationStrongTracker
from statistics import mean
from typing import Callable

Monte_Carlo = 1
def error_caculation(predict_output_cat_cpu, update_output_cat_cpu, state_labels_cpu):
    # 函数功能： 计算误差结果 输出位置 速度上的误差
    predict_error = ((predict_output_cat_cpu[0, :, 0, 0] -
                      state_labels_cpu[0, args.predictor_time_series_len:, 0, 0]) ** 2
                     + (predict_output_cat_cpu[0, :, 0, 2] -
                        state_labels_cpu[0, args.predictor_time_series_len:, 0, 3]) ** 2
                     + (predict_output_cat_cpu[0, :, 0, 4] -
                        state_labels_cpu[0, args.predictor_time_series_len:, 0, 6]) ** 2
                     ) ** 0.5

    predict_vel_err = ((predict_output_cat_cpu[0, :, 0, 1] -
                               state_labels_cpu[0, args.predictor_time_series_len:, 0, 1]) ** 2
                              + (predict_output_cat_cpu[0, :, 0, 3] -
                                 state_labels_cpu[0, args.predictor_time_series_len:, 0, 4]) ** 2
                              + (predict_output_cat_cpu[0, :, 0, 5] -
                                 state_labels_cpu[0, args.predictor_time_series_len:, 0, 7]) ** 2
                              ) ** 0.5
    update_err = ((update_output_cat_cpu[0, :, 0, 0] -
                   state_labels_cpu[0, args.predictor_time_series_len:, 0, 0]) ** 2
                  + (update_output_cat_cpu[0, :, 0, 2] -
                     state_labels_cpu[0, args.predictor_time_series_len:, 0, 3]) ** 2
                  + (update_output_cat_cpu[0, :, 0, 4] -
                     state_labels_cpu[0, args.predictor_time_series_len:, 0, 6]) ** 2
                  ) ** 0.5
    update_vel_err = ((update_output_cat_cpu[0, :, 0, 1] -
                       state_labels_cpu[0, args.predictor_time_series_len:, 0, 1]) ** 2
                      + (update_output_cat_cpu[0, :, 0, 3] -
                         state_labels_cpu[0, args.predictor_time_series_len:, 0, 4]) ** 2
                      + (update_output_cat_cpu[0, :, 0, 5] -
                         state_labels_cpu[0, args.predictor_time_series_len:, 0, 7]) ** 2
                      ) ** 0.5

    return predict_error, predict_vel_err, update_err, update_vel_err

def normalize(tarTracks: TARTRACKS, detections, minMaxScaler: Callable, minMaxScaler_MCU: Callable, frame_index):
    # 函数功能： 归一化
    update_history = torch.cat(tarTracks.x_update_history, dim=1).clone()
    input_sigma = torch.cat(tarTracks.x_sigma, dim=1).detach()
    # 对LSTM 输入归一化
    begin_index = (frame_index - args.predictor_time_series_len) if (frame_index - args.predictor_time_series_len) > 0 else 0
    input_sigma = input_sigma[:, begin_index:, :]
    (normalized_detections, normalized_update_history,
     min_vals, max_vals) = \
        minMaxScaler(detections[:, begin_index: frame_index + 1, :, :],
                     update_history[:, begin_index:, :], args.T, args.max_velocity, mode="-1_1")

    # 对MCU 输入归一化
    begin_MCU_index = (frame_index - args.predictor_MCU_len) if (frame_index - args.predictor_MCU_len) > 0 else 0
    (normalized_detections_MCU, normalized_update_history_MCU,
     min_vals, max_vals) = \
        minMaxScaler_MCU(detections[:, begin_MCU_index: frame_index, :, :],
                         update_history[:, begin_MCU_index:, :], args.T, args.max_velocity, mode="-1_1")
    # 对长度不足的序列补零
    if frame_index - args.predictor_time_series_len < 0:
        normalized_detections = torch.cat([torch.zeros([normalized_detections.shape[0],
                                                            args.predictor_time_series_len - normalized_detections.shape[1],
                                                            normalized_detections.shape[2],
                                                            normalized_detections.shape[3]]).to(
            normalized_detections.device),
                                               normalized_detections], dim=1)
        normalized_update_history = torch.cat([torch.zeros([normalized_update_history.shape[0],
                                                                args.predictor_time_series_len -
                                                                normalized_update_history.shape[1],
                                                                normalized_update_history.shape[2]]).to(
            normalized_update_history.device),
                                                   normalized_update_history], dim=1)
        input_sigma = torch.cat([torch.zeros([input_sigma.shape[0],
                                                        args.predictor_time_series_len -
                                                        input_sigma.shape[1],
                                                        input_sigma.shape[2]]).to(
        input_sigma.device),
        input_sigma], dim=1)
    # 对长度不足的序列补零
    if frame_index - args.predictor_MCU_len < 0:
        normalized_detections_MCU = torch.cat([torch.zeros([normalized_detections_MCU.shape[0],
                                                            args.predictor_MCU_len - normalized_detections_MCU.shape[1],
                                                            normalized_detections_MCU.shape[2],
                                                            normalized_detections_MCU.shape[3]]).to(
            normalized_detections_MCU.device),
                                               normalized_detections_MCU], dim=1)
        normalized_update_history_MCU = torch.cat([torch.zeros([normalized_update_history_MCU.shape[0],
                                                                args.predictor_MCU_len -
                                                                normalized_update_history_MCU.shape[1],
                                                                normalized_update_history_MCU.shape[2]]).to(
            normalized_update_history_MCU.device),
                                                   normalized_update_history_MCU], dim=1)



    normalized_detections = normalized_detections.squeeze(dim=2)
    normalized_detections_MCU = normalized_detections_MCU.squeeze(dim=2)

    return normalized_update_history, normalized_update_history_MCU, normalized_detections, normalized_detections_MCU,\
        input_sigma

# 使用六维状态向量
def evaluation(model, args):
    state_labels, detections = DataLoadFromMatlab_oneTarget_3D_for_paper()

    minMaxScaler = MyMinMaxScaler()
    minMaxScaler_MCU = MyMinMaxScaler()  # 归一化方法

    detections = detections.to(args.device)  # (batch, n_frames_Ob, ob_num_max, 3)
    state_labels = state_labels.to(args.device)  # (batch, n_frames_state_labels, tg_num_max, 9)

    detections = detections.to(torch.float32)
    state_labels = state_labels.to(torch.float32)

    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])  # 2024_03_26_20_42_ 2024_04_01_20_02_

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
        input_sigma = normalize(tarTracks, detections, minMaxScaler, minMaxScaler_MCU, frame_index)

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
