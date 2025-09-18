import torch
from models.MCST import ManeuverCompensationStrongTracker
import config
from typing import Tuple, Optional, Union

if __name__ == '__main__':
    args = config.Args().get_parser()

    # 加载模型
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

    track_model.eval()

    track_model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict'])

    # 提取子模块
    predictor = track_model.predictor
    updater = track_model.updater

    # 准备dummy inputs
    # predictor
    update_sigma = torch.randn(1, args.predictor_time_series_len, args.predictor_in_features)
    xUpdatePrevious_norm = torch.randn(1, args.predictor_time_series_len, args.predictor_in_features)
    normalized_detection_MCU = torch.randn(1, args.predictor_MCU_len, args.updater_in_features)
    normalized_update_history_MCU = torch.randn(1, args.predictor_MCU_len, args.predictor_in_features)

    hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
        torch.zeros(args.predictor_lstm_num_layers * 2, args.predictor_sampling_num * 1, args.predictor_hidden_features),
        torch.zeros(args.predictor_lstm_num_layers * 2, args.predictor_sampling_num * 1, args.predictor_hidden_features)
    )

    # updater
    x = torch.randn(1, 1, args.updater_out_features)
    x_sigma_log = torch.randn(1, 1, args.updater_out_features)
    detection = torch.randn(1, args.predictor_time_series_len, args.updater_in_features)

    predictor_script = torch.jit.script(predictor, (update_sigma, xUpdatePrevious_norm, normalized_detection_MCU,
         normalized_update_history_MCU, hidden_states))
    predictor_script.eval()
    out = predictor_script(update_sigma, xUpdatePrevious_norm, normalized_detection_MCU,
         normalized_update_history_MCU, hidden_states)

    predictor_script.save("predictor.pt")

    updater_script = torch.jit.script(updater, (x, x_sigma_log, detection))
    updater_script.eval()
    out = updater_script(x, x_sigma_log, detection)
    updater_script.save("updater.pt")

