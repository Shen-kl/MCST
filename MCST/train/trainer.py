import torch
import torch.nn as nn
from tqdm import tqdm
from config import Args
from torch.utils.data import DataLoader
from torch import optim
from utils.tracksManage import TARTRACKS
from utils.MinMaxScaler import MyMinMaxScaler
from utils.LossCompute import LossCompute_NLL
import time
import config

def train_and_evaluate(model,
          train_loader: DataLoader,
          test_loader: DataLoader,
          optimizer: optim,
          lr_schedule,
          logger,
          args: Args
          ):

    minMaxScaler = MyMinMaxScaler()  #归一化方法
    minMaxScaler_MCU = MyMinMaxScaler()  # 归一化方法
    mse = nn.MSELoss(reduction='mean') #损失函数
    nll = LossCompute_NLL() #损失函数
    min_evaluation = 2e8

    for index in range(args.train_epochs):
        print("memory difference")
        # tr.print_diff()

        model.train()
        train_track_model_loss_sum = []
        test_track_model_loss_sum = []
        train_track_model_update_location_evalutaion_sum = []
        train_track_model_update_velocity_evalutaion_sum = []
        train_track_model_predict_location_evalutaion_sum = []
        train_track_model_predict_velocity_evalutaion_sum = []
        with tqdm(total=len(train_loader), desc=f'{index}/{args.train_epochs}') as pbar:
            for (detections, state_labels) in train_loader:
                detections = detections.to(args.device)  # (batch, n_frames_Ob, ob_num_max, 2)
                state_labels = state_labels.to(args.device)  # (batch, n_frames_state_labels, tg_num_max, 4)

                # 初始化类数组
                tarTracks = TARTRACKS()  # 假设1个目标
                tarTracks.track_init(args, detections, args.train_batch_size)

                h_predict = model.predictor.init_hidden(args.predictor_sampling_num * args.train_batch_size)

                c_predict = model.predictor.init_cell(args.predictor_sampling_num * args.train_batch_size)

                total_loss = []
                total_predict_loss = []
                total_update_loss = []
                total_predict_location_evaluation = []
                total_predict_velocity_evaluation = []
                total_update_location_evaluation = []
                total_update_velocity_evaluation = []
                loss = 0
                for frame_index in range(args.predictor_time_series_len, detections.shape[1]):
                    config.frame_index = frame_index
                    # 对张量进行最小-最大归一化
                    predict_history = torch.cat(tarTracks.x_predict_history, dim=1).clone()
                    update_history = torch.cat(tarTracks.x_update_history, dim=1).clone()
                    (normalized_state_labels, normalized_detections, normalized_update_history,
                     min_vals, max_vals) = \
                        minMaxScaler(state_labels[:, frame_index - args.predictor_time_series_len: frame_index + 1, :,
                                     [0, 1, 3, 4, 6, 7]],
                                     detections[:, frame_index - args.predictor_time_series_len: frame_index + 1, :, :],
                                     update_history[:, -args.predictor_time_series_len:, :], args.T, args.max_velocity,
                                     mode="-1_1")

                    tmp = (frame_index - args.predictor_MCU_len) if (frame_index - args.predictor_MCU_len) > 0 else 0
                    (_, normalized_detections_MCU, normalized_update_history_MCU,
                     min_vals, max_vals) = \
                        minMaxScaler_MCU(state_labels[:, tmp: frame_index, :,
                                         [0, 1, 3, 4, 6, 7]],
                                         detections[:, tmp: frame_index, :, :],
                                         update_history, args.T, args.max_velocity, mode="-1_1")

                    if frame_index - args.predictor_MCU_len < 0:
                        normalized_detections_MCU = torch.cat([torch.zeros([normalized_detections_MCU.shape[0],
                                                                            args.predictor_MCU_len -
                                                                            normalized_detections_MCU.shape[1],
                                                                            normalized_detections_MCU.shape[2],
                                                                            normalized_detections_MCU.shape[3]]).to(
                            normalized_detections_MCU.device), normalized_detections_MCU], dim=1)
                        normalized_update_history_MCU = torch.cat([torch.zeros([normalized_update_history_MCU.shape[0],
                                                                                args.predictor_MCU_len -
                                                                                normalized_update_history_MCU.shape[1],
                                                                                normalized_update_history_MCU.shape[
                                                                                    2]]).to(
                            normalized_update_history_MCU.device), normalized_update_history_MCU], dim=1)

                    input_sigma = torch.cat(tarTracks.x_sigma, dim=1).detach()

                    normalized_detections = normalized_detections.squeeze(dim=2)
                    normalized_detections_MCU = normalized_detections_MCU.squeeze(dim=2)
                    # 预测
                    output_normalized_predict, output_predict_sigma, (h_predict, c_predict), \
                    output_normalized_update, output_update_sigma, \
                    output_detection_predict_sigma \
                        = model(input_sigma, normalized_detections,
                                      normalized_update_history.to(args.device), normalized_detections_MCU, normalized_update_history_MCU.to(args.device),
                                (h_predict, c_predict))

                    # 更新 predict update
                    predict_output_data = minMaxScaler.deMinMaxScaler(
                        output_normalized_predict.unsqueeze(dim=2)).squeeze(
                        dim=2)
                    update_output_data = minMaxScaler.deMinMaxScaler(output_normalized_update.unsqueeze(dim=2)).squeeze(
                        dim=2)

                    # 更新历史
                    if len(tarTracks.x_update_history) == args.predictor_MCU_len:
                        tarTracks.x_predict_history.pop(0)
                        tarTracks.x_predict_history.append(predict_output_data)
                        tarTracks.x_sigma.pop(0)
                        tarTracks.x_sigma.append(output_update_sigma)
                        tarTracks.x_update_history.pop(0)
                        tarTracks.x_update_history.append(update_output_data)
                    else:
                        tarTracks.x_predict_history.pop(0)
                        tarTracks.x_predict_history.append(predict_output_data)
                        tarTracks.x_sigma.pop(0)
                        tarTracks.x_sigma.append(output_update_sigma)
                        tarTracks.x_update_history.append(update_output_data)

                    # 计算loss
                    state_labels_copy0 = normalized_state_labels[:, -1, 0, :].unsqueeze(dim=1)
                    detections_copy0 = normalized_detections[:, -1, :].unsqueeze(dim=1)

                    predict_loss = nll(output_normalized_predict, output_predict_sigma,
                                       state_labels_copy0) + \
                                   mse(output_normalized_predict, state_labels_copy0)

                    update_loss = nll(detections_copy0, output_detection_predict_sigma,
                                      state_labels_copy0[:, :, 0::2]) \
                                  + nll(output_normalized_update, output_update_sigma,
                                        state_labels_copy0) + \
                                  mse(output_normalized_update, state_labels_copy0)

                    loss = loss + predict_loss + update_loss

                    single_loss = predict_loss.data + update_loss.data
                    # 所有损失
                    total_loss.append(single_loss.item())
                    total_predict_loss.append(predict_loss.item())
                    total_update_loss.append(update_loss.item())

                    state_labels_copy1 = state_labels[:, frame_index, 0, [0, 1, 3, 4, 6, 7]].unsqueeze(dim=1).data

                    predict_location_evaluation = mse(predict_output_data[:,:,0::2].data, state_labels_copy1[:,:,0::2])
                    predict_velocity_evaluation = mse(predict_output_data[:,:,1::2].data, state_labels_copy1[:,:,1::2])

                    update_location_evaluation = mse(update_output_data[:,:,0::2].data, state_labels_copy1[:,:,0::2])
                    update_velocity_evaluation = mse(update_output_data[:,:,1::2].data, state_labels_copy1[:,:,1::2])

                    total_predict_location_evaluation.append(float(predict_location_evaluation.item()))
                    total_predict_velocity_evaluation.append(float(predict_velocity_evaluation.item()))

                    total_update_location_evaluation.append(float(update_location_evaluation.item()))
                    total_update_velocity_evaluation.append(float(update_velocity_evaluation.item()))

                    train_track_model_loss_sum.append(single_loss.item())

                train_track_model_update_location_evalutaion_sum.append(
                    float(update_location_evaluation.item()))
                train_track_model_update_velocity_evalutaion_sum.append(
                    float(update_velocity_evaluation.item()))

                train_track_model_predict_location_evalutaion_sum.append(
                    float(predict_location_evaluation.item()))
                train_track_model_predict_velocity_evalutaion_sum.append(
                    float(predict_velocity_evaluation.item()))

                # 梯度归零
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 更新
                optimizer.step()

                torch.cuda.empty_cache()

                pbar.set_postfix({
                    'train_predict_loss': float(sum(total_predict_loss) / len(total_predict_loss)),
                    'train_update_loss': float(sum(total_update_loss) / len(total_update_loss)),
                    'train_predict_location_evaluation': float(
                        sum(total_predict_location_evaluation) / len(total_predict_location_evaluation)),
                    'train_predict_velocity_evaluation': float(
                        sum(total_predict_velocity_evaluation) / len(total_predict_velocity_evaluation)),
                    'train_update_location_evaluation': float(
                        sum(total_update_location_evaluation) / len(total_update_location_evaluation)),
                    'train_update_velocity_evaluation': float(
                        sum(total_update_velocity_evaluation) / len(total_update_velocity_evaluation)),
                })
                pbar.update(1)

                del tarTracks.x_predict_history
                del tarTracks.x_sigma
                del tarTracks.x_update_history
                del state_labels
                del detections
                del tarTracks

        # evaluation
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f'{index}/{args.train_epochs}') as pbar:
                for (detections, state_labels) in test_loader:
                    detections = detections.to(args.device)  # (batch, n_frames_Ob, ob_num_max, 2)
                    state_labels = state_labels.to(args.device)  # (batch, n_frames_state_labels, tg_num_max, 4)

                    # 初始化类数组
                    tarTracks = TARTRACKS()  # 假设1个目标
                    tarTracks.track_init(args, detections, args.train_batch_size)

                    h_predict = model.predictor.init_hidden(args.predictor_sampling_num * args.train_batch_size)

                    c_predict = model.predictor.init_cell(args.predictor_sampling_num * args.train_batch_size)

                    total_loss = []
                    total_predict_loss = []
                    total_update_loss = []
                    total_predict_location_evaluation = []
                    total_predict_velocity_evaluation = []
                    total_update_location_evaluation = []
                    total_update_velocity_evaluation = []

                    loss = 0
                    for frame_index in range(args.predictor_time_series_len, detections.shape[1]):
                        # 对张量进行最小-最大归一化
                        predict_history = torch.cat(tarTracks.x_predict_history, dim=1).clone()
                        update_history = torch.cat(tarTracks.x_update_history, dim=1).clone()
                        (normalized_state_labels, normalized_detections, normalized_predict_history, normalized_update_history,
                         min_vals, max_vals) = \
                            minMaxScaler(
                                state_labels[:, frame_index - args.predictor_time_series_len: frame_index + 1, :,
                                [0, 1, 3, 4, 6, 7]],
                                detections[:, frame_index - args.predictor_time_series_len: frame_index + 1, :, :],
                                predict_history[:, -args.predictor_time_series_len:, :],
                                update_history[:, -args.predictor_time_series_len:, :], args.T, args.max_velocity,
                                mode="-1_1")

                        (_, normalized_detections_MCU, _, normalized_update_history_MCU,
                         min_vals, max_vals) = \
                            minMaxScaler_MCU(state_labels[:, frame_index - args.predictor_MCU_len: frame_index, :,
                                             [0, 1, 3, 4, 6, 7]],
                                             detections[:, frame_index - args.predictor_MCU_len: frame_index, :, :],
                                             predict_history,
                                             update_history, args.T, args.max_velocity, mode="-1_1")

                        if frame_index - args.predictor_MCU_len < 0:
                            normalized_detections_MCU = torch.cat([torch.zeros([normalized_detections_MCU.shape[0],
                                                                                args.predictor_MCU_len -
                                                                                normalized_detections_MCU.shape[1],
                                                                                normalized_detections_MCU.shape[2],
                                                                                normalized_detections_MCU.shape[3]]).to(
                                normalized_detections_MCU.device), normalized_detections_MCU], dim=1)
                            normalized_update_history_MCU = torch.cat(
                                [torch.zeros([normalized_update_history_MCU.shape[0],
                                              args.predictor_MCU_len - normalized_update_history_MCU.shape[1],
                                              normalized_update_history_MCU.shape[2]]).to(
                                    normalized_update_history_MCU.device), normalized_update_history_MCU], dim=1)
                            
                        input_sigma = torch.cat(tarTracks.x_sigma, dim=1).detach()

                        normalized_detections = normalized_detections.squeeze(dim=2)
                        normalized_detections_MCU = normalized_detections_MCU.squeeze(dim=2)
                        # 预测
                        output_normalized_predict, output_predict_sigma, (h_predict, c_predict), \
                        output_normalized_update, output_update_sigma, \
                        output_detection_predict_sigma \
                            = model(input_sigma, normalized_detections,
                                    normalized_update_history.to(args.device), normalized_detections_MCU,
                                    normalized_update_history_MCU.to(args.device),
                                    (h_predict, c_predict))

                        # 更新 predict update
                        predict_output_data = minMaxScaler.deMinMaxScaler(
                            output_normalized_predict.unsqueeze(dim=2)).squeeze(
                            dim=2)
                        update_output_data = minMaxScaler.deMinMaxScaler(
                            output_normalized_update.unsqueeze(dim=2)).squeeze(
                            dim=2)

                        # 更新历史
                        if len(tarTracks.x_update_history) == args.predictor_MCU_len:
                            tarTracks.x_predict_history.pop(0)
                            tarTracks.x_predict_history.append(predict_output_data)
                            tarTracks.x_sigma.pop(0)
                            tarTracks.x_sigma.append(output_update_sigma)
                            tarTracks.x_update_history.pop(0)
                            tarTracks.x_update_history.append(update_output_data)
                        else:
                            tarTracks.x_predict_history.pop(0)
                            tarTracks.x_predict_history.append(predict_output_data)
                            tarTracks.x_sigma.pop(0)
                            tarTracks.x_sigma.append(output_update_sigma)
                            tarTracks.x_update_history.append(update_output_data)

                        # 计算loss
                        state_labels_copy0 = normalized_state_labels[:, -1, 0, :].unsqueeze(dim=1)
                        detections_copy0 = normalized_detections[:, -1, :].unsqueeze(dim=1)

                        predict_loss = nll(output_normalized_predict, output_predict_sigma,
                                           state_labels_copy0) + \
                                       mse(output_normalized_predict, state_labels_copy0)

                        update_loss = nll(detections_copy0, output_detection_predict_sigma,
                                          state_labels_copy0[:, :, 0::2]) \
                                      + nll(output_normalized_update, output_update_sigma,
                                            state_labels_copy0) + \
                                      mse(output_normalized_update, state_labels_copy0)

                        loss = loss + predict_loss + update_loss

                        single_loss = predict_loss.data + update_loss.data
                        # 所有损失
                        total_loss.append(single_loss.item())
                        total_predict_loss.append(predict_loss.item())
                        total_update_loss.append(update_loss.item())

                        state_labels_copy1 = state_labels[:, frame_index, 0, [0, 1, 3, 4, 6, 7]].unsqueeze(dim=1).data

                        predict_location_evaluation = mse(predict_output_data[:, :, 0::2].data,
                                                          state_labels_copy1[:, :, 0::2])
                        predict_velocity_evaluation = mse(predict_output_data[:, :, 1::2].data,
                                                          state_labels_copy1[:, :, 1::2])

                        update_location_evaluation = mse(update_output_data[:, :, 0::2].data,
                                                         state_labels_copy1[:, :, 0::2])
                        update_velocity_evaluation = mse(update_output_data[:, :, 1::2].data,
                                                         state_labels_copy1[:, :, 1::2])

                        total_predict_location_evaluation.append(float(predict_location_evaluation.item()))
                        total_predict_velocity_evaluation.append(float(predict_velocity_evaluation.item()))

                        total_update_location_evaluation.append(float(update_location_evaluation.item()))
                        total_update_velocity_evaluation.append(float(update_velocity_evaluation.item()))

                        test_track_model_loss_sum.append(single_loss.item())

                    torch.cuda.empty_cache()

                    pbar.set_postfix({
                        'test_predict_loss': float(sum(total_predict_loss) / len(total_predict_loss)),
                        'test_update_loss': float(sum(total_update_loss) / len(total_update_loss)),
                        'test_predict_location_evaluation': float(
                            sum(total_predict_location_evaluation) / len(total_predict_location_evaluation)),
                        'test_predict_velocity_evaluation': float(
                            sum(total_predict_velocity_evaluation) / len(total_predict_velocity_evaluation)),
                        'test_update_location_evaluation': float(
                            sum(total_update_location_evaluation) / len(total_update_location_evaluation)),
                        'test_update_velocity_evaluation': float(
                            sum(total_update_velocity_evaluation) / len(total_update_velocity_evaluation)),
                    })
                    pbar.update(1)

                    del tarTracks.x_predict_history
                    del tarTracks.x_sigma
                    del tarTracks.x_update_history
                    del state_labels
                    del detections
                    del tarTracks

        lr_schedule.step(sum(train_track_model_loss_sum) / len(train_track_model_loss_sum))

        now = time.localtime()
        nowt = time.strftime("%Y_%m_%d_%H_%M_", now)
        train_evaluation = sum(train_track_model_update_location_evalutaion_sum) / len(train_track_model_update_location_evalutaion_sum) \
                          + sum(train_track_model_update_velocity_evalutaion_sum) / len(train_track_model_update_velocity_evalutaion_sum)

        if min_evaluation > train_evaluation:
            min_evaluation = train_evaluation
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': sum(test_track_model_loss_sum) / len(test_track_model_loss_sum),
                'lr_schedule': lr_schedule.state_dict()
            }, './state_dict/ManeuverCompensationStrongTracker3D/' + str(nowt) + ".pth")

