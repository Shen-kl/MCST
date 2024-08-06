import argparse
import os
import torch
from train.trainer import train_and_evaluate
from data.load_data import DataLoad_OneManeuveringTarget_3D
from torch.utils.data import DataLoader
import config
import numpy as np
from models.MCST import ManeuverCompensationStrongTracker
from loguru import logger
from evaluation.evaluation_Monte_Carlo import evaluation

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = config.Args().get_parser()
    setup_seed(args.seed)

    logger.add(args.log_dir)

    train_data = DataLoad_OneManeuveringTarget_3D(
        data_path=args.data_dir,
        is_train=1,
        n_frames=args.frame_max,
    )
    test_data = DataLoad_OneManeuveringTarget_3D(
        data_path=args.data_dir,
        is_train=0,
        n_frames=args.frame_max,
    )

    # 按照batch_size划分
    train_load = DataLoader(train_data, args.train_batch_size, shuffle=True)
    test_load = DataLoader(test_data, args.train_batch_size, shuffle=True)

    # 模型
    track_model = ManeuverCompensationStrongTracker(args.predictor_in_features,
                                                     args.predictor_hidden_features,
                                                     args.predictor_out_features,
                                                     args.predictor_lstm_num_layers,
                                                     args.predictor_MCU_num_layers,
                                                     args.predictor_sampling_num,
                                                     args.predictor_MCU_layer,
                                                     args.predictor_MCU_hidden_features,
                                                     args.updater_in_features,
                                                     args.updater_hidden_features,
                                                     args.updater_out_features,
                                                     args.updater_dropoutrate,
                                                     args.predictor_time_series_len,
                                                     args.device).to(args.device)
    track_model_optimizer = torch.optim.Adam(track_model.parameters(), args.lr)
    track_model_lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(track_model_optimizer,
                                                                         factor=args.optimizer_factor,
                                                                         patience=args.optimizer_patience)

    train = 1
    if train == 1:
        train_and_evaluate(track_model, train_load, test_load, track_model_optimizer, track_model_lr_schedule, logger, args)
    else:
        evaluation(track_model, args)