import torch
import torch.nn as nn

class TARTRACKS(object):
    def __init__(self):
        self.x_sigma = []
        self.x_predict_history = []
        self.x_update_history = []

    def track_init(self, args, Ob, batch_size):
        for i in range(args.predictor_time_series_len):
            self.x_sigma.append(
                (torch.zeros(batch_size, 1,
                             args.predictor_in_features,
                             requires_grad=True)).to(args.device))
        
        velcoity1 = ((Ob[:, 4, 0, 0] - Ob[:, 0, 0, 0]) / (4 * args.T)).to(args.device)
        velcoity2 = ((Ob[:, 4, 0, 1] - Ob[:, 0, 0, 1]) / (4 * args.T)).to(args.device)
        velcoity3 = ((Ob[:, 4, 0, 2] - Ob[:, 0, 0, 2]) / (4 * args.T)).to(args.device)

        for index in range(args.predictor_time_series_len):
            tmp = torch.cat(
                (Ob[:, index, 0, 0].unsqueeze(dim=1).unsqueeze(dim=2),
                 velcoity1.unsqueeze(dim=1).unsqueeze(dim=2),
                 Ob[:, index, 0, 1].unsqueeze(dim=1).unsqueeze(dim=2),
                 velcoity2.unsqueeze(dim=1).unsqueeze(dim=2),
                 Ob[:, index, 0, 2].unsqueeze(dim=1).unsqueeze(dim=2),
                 velcoity3.unsqueeze(dim=1).unsqueeze(dim=2)), dim=2)
            self.x_predict_history.append(tmp.to(args.device))
            self.x_update_history.append(tmp.to(args.device))


