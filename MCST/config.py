import argparse
import torch

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(description='MCST')
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./checkpoints/',
                            help='the output dir for model checkpoints')
        parser.add_argument('--data_dir', default='G:/Dataset/OneManeuveringTargets3Dv5/',
                            help='data dir for uer')
        parser.add_argument('--data_dir_test', default='G:/Dataset/OneManeuveringTargets3Dv5_forTest/',
                            help='evaluation data dir for uer')
        parser.add_argument('--log_dir', default='./log/demo_log.log',
                            help='log dir for uer')

        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), #
                            help='cpu or gpu')
        parser.add_argument('--train_batch_size', default=8, type=int)
        parser.add_argument('--train_epochs', default=5000, type=int,
                            help='Max training epoch')
        parser.add_argument('--test_epochs', default=10, type=int,
                            help='Max training epoch')
        parser.add_argument('--eval_batch_size', default=1, type=int)
        parser.add_argument('--optimizer_factor', default=0.8, type=int)
        parser.add_argument('--optimizer_patience', default=5, type=int)
        parser.add_argument('--lr', default=1e-5, type=float,
                            help='learning rate')
        # train args
        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')
        parser.add_argument('--T', default=0.4, type=float,
                            help='data rate') # 0.4  3
        parser.add_argument('--predictor_in_features', default=6, type=float)
        parser.add_argument('--predictor_hidden_features', default=64, type=float)
        parser.add_argument('--predictor_out_features', default=6, type=float,)
        parser.add_argument('--predictor_lstm_num_layers', default=2, type=float,)
        parser.add_argument('--predictor_MCU_num_layers', default=2, type=float)
        parser.add_argument('--predictor_sampling_num', default=5, type=int)
        parser.add_argument('--predictor_time_series_len', default=5, type=float)
        parser.add_argument('--predictor_MCU_len', default=16, type=float,)
        parser.add_argument('--predictor_MCU_layer', default=2, type=float, )
        parser.add_argument('--predictor_MCU_hidden_features', default=256, type=float, )

        parser.add_argument('--updater_in_features', default=3, type=float)
        parser.add_argument('--updater_hidden_features', default=128, type=float)
        parser.add_argument('--updater_out_features', default=6, type=float)
        parser.add_argument('--updater_dropoutrate', default=0.1, type=float)
        parser.add_argument('--frame_max', default=100, type=int,
                            help='the maximum number of frames in a track')
        parser.add_argument('--max_velocity', default=340*5, type=float)  #  340*5  340

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()

frame_index = 0