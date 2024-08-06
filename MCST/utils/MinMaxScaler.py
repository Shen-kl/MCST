import torch
import torch.nn as nn

class MyMinMaxScaler(nn.Module):
    #  速度 归一化范围由设定最大小值决定 ，与输入值大小无关
    def __init__(self):
        super(MyMinMaxScaler, self).__init__()

    def forward(self, *args, mode: str):
        # 最小最大归一化
        # labels (batch, len, dim) detection (batch, len, dim)
        # 沿着时间维度的方向计算最小值和最大值
        self.mode = mode
        max_velocity = args[-1]
        if len(args) == 5:
            labels = args[0]
            detection = args[1]
            estimation = args[2]
            T = args[3]

            min_vals_detection, _ = torch.min(detection[:, :-1, :, :], dim=1, keepdim=True)
            max_vals_detection, _ = torch.max(detection[:, :-1, :, :], dim=1, keepdim=True)

            dim_size = len(labels.shape)

            if dim_size == 3:
                min_vals = torch.zeros([estimation.shape[0], 1, estimation.shape[-1]]).to(min_vals_detection.device)
                max_vals = torch.zeros([estimation.shape[0], 1, estimation.shape[-1]]).to(min_vals_detection.device)

                min_vals[:, :, 0::2] = min_vals_detection
                max_vals[:, :, 0::2] = max_vals_detection

                # 考虑未来 增大归一化范围
                max_vals = max_vals + torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                max_vel_tensor = torch.ones_like(max_vals[:, :, 1::2]) * max_velocity
                max_vals[:, :, 1::2] = max_vel_tensor

                min_vals = min_vals - torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                min_vel_tensor = -torch.ones_like(min_vals[:, :, 1::2]) * max_velocity
                min_vals[:, :, 1::2] = min_vel_tensor

            elif dim_size == 4:
                min_vals = torch.zeros([estimation.shape[0], 1, 1, estimation.shape[-1]]).to(min_vals_detection.device)
                max_vals = torch.zeros([estimation.shape[0], 1, 1, estimation.shape[-1]]).to(min_vals_detection.device)

                min_vals[:, :, :, 0::2] = min_vals_detection
                max_vals[:, :, :, 0::2] = max_vals_detection

                # 考虑未来 增大归一化范围
                max_vals = max_vals + torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                max_vel_tensor = torch.ones_like(max_vals[:, :, :, 1::2]) * max_velocity
                max_vals[:, :, :, 1::2] = max_vel_tensor

                min_vals = min_vals - torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                min_vel_tensor = -torch.ones_like(min_vals[:, :, :, 1::2]) * max_velocity
                min_vals[:, :, :, 1::2] = min_vel_tensor

            else:
                print("MyMinMaxScaler error!")

            self.min_vals = min_vals
            self.max_vals = max_vals

            # 分母向量
            denominator = max_vals - min_vals
            one_tensor = torch.ones_like(denominator)
            # 将分母中为 0 的数 变为 1 避免 nan
            denominator = torch.where(denominator == 0, one_tensor, denominator)

            # 最小-最大缩放，将x的范围缩放到[0, 1]
            scaled_estimation = (estimation.unsqueeze(dim=2) - min_vals) / denominator
            scaled_detection = torch.zeros_like(detection)

            if dim_size == 3:
                scaled_detection = (detection - min_vals[:, :, 0::2]) / denominator[:, :, 0::2]
                if labels.shape[-1] == 3:
                    scaled_label = (labels - min_vals[:, :, 0::2]) / denominator[:, :, 0::2]
                else:
                    scaled_label = (labels - min_vals) / denominator
            elif dim_size == 4:
                scaled_detection = (detection - min_vals[:, :, :, 0::2]) / denominator[:, :, :, 0::2]
                if labels.shape[-1] == 3:
                    scaled_label = (labels - min_vals[:, :, :, 0::2]) / denominator[:, :, :, 0::2]
                else:
                    scaled_label = (labels - min_vals) / denominator
            else:
                print("MyMinMaxScaler error!")

            # 将x的范围缩放到[-1, 1]
            if mode == "-1_1":
                scaled_label = (scaled_label - 0.5) / 0.5
                scaled_detection = (scaled_detection - 0.5) / 0.5
                scaled_estimation = (scaled_estimation - 0.5) / 0.5

            scaled_estimation = scaled_estimation.squeeze(dim=2)

            return (scaled_label, scaled_detection, scaled_estimation, min_vals, max_vals)
        elif len(args) == 4:
            detection = args[0]
            estimation = args[1]
            T = args[2]

            min_vals_detection, _ = torch.min(detection[:, :-1, :, :], dim=1, keepdim=True)
            max_vals_detection, _ = torch.max(detection[:, :-1, :, :], dim=1, keepdim=True)

            dim_size = len(detection.shape)

            if dim_size == 3:
                min_vals = torch.zeros([estimation.shape[0], 1, estimation.shape[-1]]).to(min_vals_detection.device)
                max_vals = torch.zeros([estimation.shape[0], 1, estimation.shape[-1]]).to(min_vals_detection.device)

                min_vals[:, :, 0::2] = min_vals_detection
                max_vals[:, :, 0::2] = max_vals_detection

                # 考虑未来 增大归一化范围
                max_vals = max_vals + torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                max_vel_tensor = torch.ones_like(max_vals[:, :, 1::2]) * max_velocity
                max_vals[:, :, 1::2] = max_vel_tensor

                min_vals = min_vals - torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                min_vel_tensor = -torch.ones_like(min_vals[:, :, 1::2]) * max_velocity
                min_vals[:, :, 1::2] = min_vel_tensor

            elif dim_size == 4:
                min_vals = torch.zeros([estimation.shape[0], 1, 1, estimation.shape[-1]]).to(min_vals_detection.device)
                max_vals = torch.zeros([estimation.shape[0], 1, 1, estimation.shape[-1]]).to(min_vals_detection.device)

                min_vals[:, :, :, 0::2] = min_vals_detection
                max_vals[:, :, :, 0::2] = max_vals_detection

                # 考虑未来 增大归一化范围
                max_vals = max_vals + torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                max_vel_tensor = torch.ones_like(max_vals[:, :, :, 1::2]) * max_velocity
                max_vals[:, :, :, 1::2] = max_vel_tensor

                min_vals = min_vals - torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                min_vel_tensor = -torch.ones_like(min_vals[:, :, :, 1::2]) * max_velocity
                min_vals[:, :, :, 1::2] = min_vel_tensor

            else:
                print("MyMinMaxScaler error!")

            self.min_vals = min_vals
            self.max_vals = max_vals

            # 分母向量
            denominator = max_vals - min_vals
            one_tensor = torch.ones_like(denominator)
            # 将分母中为 0 的数 变为 1 避免 nan
            denominator = torch.where(denominator == 0, one_tensor, denominator)

            # 最小-最大缩放，将x的范围缩放到[0, 1]
            scaled_estimation = (estimation.unsqueeze(dim=2) - min_vals) / denominator

            scaled_detection = torch.zeros_like(detection)
            if dim_size == 3:
                scaled_detection = (detection - min_vals[:, :, 0::2]) / denominator[:, :, 0::2]
            elif dim_size == 4:
                scaled_detection = (detection - min_vals[:, :, :, 0::2]) / denominator[:, :, :, 0::2]
            else:
                print("MyMinMaxScaler error!")

            # 将x的范围缩放到[-1, 1]
            if mode == "-1_1":
                scaled_detection = (scaled_detection - 0.5) / 0.5
                scaled_estimation = (scaled_estimation - 0.5) / 0.5

            scaled_estimation = scaled_estimation.squeeze(dim=2)

            return (scaled_detection, scaled_estimation, min_vals, max_vals)
        else:
            print("MyMinMaxScaler error!")

    def deMinMaxScaler(self, x):
        if self.mode == "0_1":
            x_output = ((x * 1) + 0) * (
                    self.max_vals - self.min_vals) + self.min_vals
            return x_output
        elif self.mode == "-1_1":
            x_output = ((x * 0.5) + 0.5) * (
                    self.max_vals - self.min_vals) + self.min_vals
            return x_output
        else:
            print("MyMinMaxScaler error!")

class MyMinMaxScaler_UAV(nn.Module):
    # 只输入六维信息  没有加速度 速度 归一化范围由建模数据集时的设定最大小值决定 ，与输入值大小无关
    # UAV 真值 只有位置信息
    def __init__(self):
        super(MyMinMaxScaler_UAV, self).__init__()

    def forward(self, *args, mode: str):
        # 最小最大归一化
        # labels (batch, len, dim) detection (batch, len, dim)
        # 沿着时间维度的方向计算最小值和最大值
        # 速度 加速度 定死范围
        self.mode = mode
        max_velocity = args[-1]
        if len(args) == 5:
            labels = args[0]
            detection = args[1]
            estimation = args[2]
            T = args[3]

            min_vals_detection, _ = torch.min(detection[:, :-1, :, :], dim=1, keepdim=True)
            max_vals_detection, _ = torch.max(detection[:, :-1, :, :], dim=1, keepdim=True)

            dim_size = len(labels.shape)

            if dim_size == 3:
                min_vals = torch.zeros([estimation.shape[0], 1, estimation.shape[-1]]).to(min_vals_detection.device)
                max_vals = torch.zeros([estimation.shape[0], 1, estimation.shape[-1]]).to(min_vals_detection.device)

                min_vals[:, :, 0::2] = min_vals_detection
                max_vals[:, :, 0::2] = max_vals_detection

                # 考虑未来 增大归一化范围
                max_vals = max_vals + torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                max_vel_tensor = torch.ones_like(max_vals[:, :, 1::2]) * max_velocity
                max_vals[:, :, 1::2] = max_vel_tensor

                min_vals = min_vals - torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                min_vel_tensor = -torch.ones_like(min_vals[:, :, 1::2]) * max_velocity
                min_vals[:, :, 1::2] = min_vel_tensor

            elif dim_size == 4:
                min_vals = torch.zeros([estimation.shape[0], 1, 1, estimation.shape[-1]]).to(min_vals_detection.device)
                max_vals = torch.zeros([estimation.shape[0], 1, 1, estimation.shape[-1]]).to(min_vals_detection.device)

                min_vals[:, :, :, 0::2] = min_vals_detection
                max_vals[:, :, :, 0::2] = max_vals_detection

                # 考虑未来 增大归一化范围
                max_vals = max_vals + torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                max_vel_tensor = torch.ones_like(max_vals[:, :, :, 1::2]) * max_velocity
                max_vals[:, :, :, 1::2] = max_vel_tensor

                min_vals = min_vals - torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                min_vel_tensor = -torch.ones_like(min_vals[:, :, :, 1::2]) * max_velocity
                min_vals[:, :, :, 1::2] = min_vel_tensor

            else:
                print("MyMinMaxScaler error!")

            self.min_vals = min_vals
            self.max_vals = max_vals

            # 分母向量
            denominator = max_vals - min_vals
            one_tensor = torch.ones_like(denominator)
            # 将分母中为 0 的数 变为 1 避免 nan
            denominator = torch.where(denominator == 0, one_tensor, denominator)

            # 最小-最大缩放，将x的范围缩放到[0, 1]

            scaled_estimation = (estimation.unsqueeze(dim=2) - min_vals) / denominator

            scaled_detection = torch.zeros_like(scaled_estimation)

            if dim_size == 3:
                scaled_detection = (detection - min_vals[:, :, 0::2]) / denominator[:, :, 0::2]
                scaled_label = (labels - min_vals[:, :, 0::2]) / denominator[:, :, 0::2]
            elif dim_size == 4:
                scaled_detection = (detection - min_vals[:, :, :, 0::2]) / denominator[:, :, :, 0::2]
                scaled_label = (labels - min_vals[:, :, :, 0::2]) / denominator[:, :, :, 0::2]
            else:
                print("MyMinMaxScaler error!")

            # 将x的范围缩放到[-1, 1]
            if mode == "-1_1":
                scaled_label = (scaled_label - 0.5) / 0.5
                scaled_detection = (scaled_detection - 0.5) / 0.5
                scaled_estimation = (scaled_estimation - 0.5) / 0.5

            scaled_estimation = scaled_estimation.squeeze(dim=2)

            return (scaled_label, scaled_detection, scaled_estimation, min_vals, max_vals)
        elif len(args) == 4:
            detection = args[0]
            estimation = args[1]
            T = args[2]

            min_vals_detection, _ = torch.min(detection[:, :-1, :, :], dim=1, keepdim=True)
            max_vals_detection, _ = torch.max(detection[:, :-1, :, :], dim=1, keepdim=True)

            dim_size = len(detection.shape)

            if dim_size == 3:
                min_vals = torch.zeros([estimation.shape[0], 1, estimation.shape[-1]]).to(min_vals_detection.device)
                max_vals = torch.zeros([estimation.shape[0], 1, estimation.shape[-1]]).to(min_vals_detection.device)

                min_vals[:, :, 0::2] = min_vals_detection
                max_vals[:, :, 0::2] = max_vals_detection

                # 考虑未来 增大归一化范围
                max_vals = max_vals + torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                max_vel_tensor = torch.ones_like(max_vals[:, :, 1::2]) * max_velocity
                max_vals[:, :, 1::2] = max_vel_tensor

                min_vals = min_vals - torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                min_vel_tensor = -torch.ones_like(min_vals[:, :, 1::2]) * max_velocity
                min_vals[:, :, 1::2] = min_vel_tensor

            elif dim_size == 4:
                min_vals = torch.zeros([estimation.shape[0], 1, 1, estimation.shape[-1]]).to(min_vals_detection.device)
                max_vals = torch.zeros([estimation.shape[0], 1, 1, estimation.shape[-1]]).to(min_vals_detection.device)

                min_vals[:, :, :, 0::2] = min_vals_detection
                max_vals[:, :, :, 0::2] = max_vals_detection

                # 考虑未来 增大归一化范围
                max_vals = max_vals + torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                max_vel_tensor = torch.ones_like(max_vals[:, :, :, 1::2]) * max_velocity
                max_vals[:, :, :, 1::2] = max_vel_tensor

                min_vals = min_vals - torch.tensor([T * max_velocity, 0,
                                                    T * max_velocity, 0,
                                                    T * max_velocity, 0]).to(max_vals.device)
                min_vel_tensor = -torch.ones_like(min_vals[:, :, :, 1::2]) * max_velocity
                min_vals[:, :, :, 1::2] = min_vel_tensor

            else:
                print("MyMinMaxScaler error!")

            self.min_vals = min_vals
            self.max_vals = max_vals

            # 分母向量
            denominator = max_vals - min_vals
            one_tensor = torch.ones_like(denominator)
            # 将分母中为 0 的数 变为 1 避免 nan
            denominator = torch.where(denominator == 0, one_tensor, denominator)

            # 最小-最大缩放，将x的范围缩放到[0, 1]
            scaled_estimation = (estimation.unsqueeze(dim=2) - min_vals) / denominator

            scaled_detection = torch.zeros_like(detection)
            if dim_size == 3:
                scaled_detection = (detection - min_vals[:, :, 0::2]) / denominator[:, :, 0::2]
            elif dim_size == 4:
                scaled_detection = (detection - min_vals[:, :, :, 0::2]) / denominator[:, :, :, 0::2]
            else:
                print("MyMinMaxScaler error!")

            # 将x的范围缩放到[-1, 1]
            if mode == "-1_1":
                scaled_detection = (scaled_detection - 0.5) / 0.5
                scaled_estimation = (scaled_estimation - 0.5) / 0.5

            scaled_estimation = scaled_estimation.squeeze(dim=2)

            return (scaled_detection, scaled_estimation, min_vals, max_vals)
        else:
            print("MyMinMaxScaler error!")

    def deMinMaxScaler(self, x):
        if self.mode == "0_1":
            x_output = ((x * 1) + 0) * (
                    self.max_vals - self.min_vals) + self.min_vals
            return x_output
        elif self.mode == "-1_1":
            x_output = ((x * 0.5) + 0.5) * (
                    self.max_vals - self.min_vals) + self.min_vals
            return x_output
        else:
            print("MyMinMaxScaler error!")