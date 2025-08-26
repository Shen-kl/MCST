import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

class MyMinMaxScaler(nn.Module):
    """
    Custom Min-Max Scaler for velocity/position tensors with optional max-velocity augmentation.
    Supports scaling to [0,1] or [-1,1].
    """

    def __init__(self, use_max_velocity: bool = True, train_mode: bool = True):
        super().__init__()
        self.use_max_velocity = use_max_velocity
        self.train_mode = train_mode
        self.mode: Optional[str] = None
        self.min_vals: Optional[torch.Tensor] = None
        self.max_vals: Optional[torch.Tensor] = None

    def forward(
        self,
        *args,
        mode: str,
        frame_index: int,
        epoch_index: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward scaling function.

        Args:
            args: Variable input arguments
                - 5 args: labels, detection, estimation, T, (max_velocity, max_acceleration)
                - 4 args: detection, estimation, T, (max_velocity, max_acceleration)
            mode: "0_1" or "-1_1"
            frame_index: current frame index
            epoch_index: current epoch index
        """
        self.mode = mode
        *inputs, velocity_info = args
        max_velocity, max_acceleration = velocity_info

        alpha = self.alpha_schedule(epoch_index) if self.train_mode else 0.8
        frame_change_threshold = 10

        if len(inputs) == 4:
            labels, detection, estimation, T = inputs
            scaled_label, scaled_detection, scaled_estimation = self._scale_all(
                labels, detection, estimation, T, max_velocity, max_acceleration,
                alpha, frame_index, frame_change_threshold
            )
            return scaled_label, scaled_detection, scaled_estimation, self.min_vals, self.max_vals
        elif len(inputs) == 3:
            detection, estimation, T = inputs
            scaled_detection, scaled_estimation = self._scale_all(
                None, detection, estimation, T, max_velocity, max_acceleration,
                alpha, frame_index, frame_change_threshold
            )
            return scaled_detection, scaled_estimation, self.min_vals, self.max_vals
        else:
            raise ValueError("Invalid number of arguments passed to MyMinMaxScaler.")

    def _scale_all(
        self,
        labels: Optional[torch.Tensor],
        detection: torch.Tensor,
        estimation: torch.Tensor,
        T: float,
        max_velocity: float,
        max_acceleration: float,
        alpha: float,
        frame_index: int,
        frame_change_threshold: int
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Core scaling logic.
        """
        dim_size = len(estimation.shape)
        min_vals, max_vals = self._compute_min_max(
            detection, estimation, T, max_velocity, max_acceleration, alpha,
            frame_index, frame_change_threshold
        )

        self.min_vals = min_vals
        self.max_vals = max_vals

        # denominator with 0 check
        denominator = torch.where(max_vals - min_vals == 0,
                                  torch.ones_like(max_vals),
                                  max_vals - min_vals)

        estimation = estimation if dim_size == 4 else estimation.unsqueeze(dim=2)
        scaled_estimation = self._scale_tensor(estimation, min_vals, denominator)
        scaled_detection = self._scale_tensor(detection, min_vals, denominator)

        if labels is not None:
            scaled_label = self._scale_tensor(labels, min_vals, denominator)
        else:
            scaled_label = None

        if mode := self.mode == "-1_1":
            factor = 0.5
            offset = 0.5
            scaled_estimation = (scaled_estimation - offset) / factor
            scaled_detection = (scaled_detection - offset) / factor
            if scaled_label is not None:
                scaled_label = (scaled_label - offset) / factor

        if scaled_label is not None:
            return scaled_label, scaled_detection, scaled_estimation.squeeze(dim=2)
        else:
            return scaled_detection, scaled_estimation.squeeze(dim=2)

    def _compute_min_max(
        self,
        detection: torch.Tensor,
        estimation: torch.Tensor,
        T: float,
        max_velocity: float,
        max_acceleration: float,
        alpha: float,
        frame_index: int,
        frame_change_threshold: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute min and max values for scaling.
        """
        min_detection, _ = torch.min(detection[:, :-1, :, :], dim=1, keepdim=True)
        max_detection, _ = torch.max(detection[:, :-1, :, :], dim=1, keepdim=True)
        dim_size = len(estimation.shape)

        shape_map = [estimation.shape[0], 1, 1, estimation.shape[-1]]

        min_vals = torch.zeros(shape_map, device=detection.device)
        max_vals = torch.zeros(shape_map, device=detection.device)

        # assign positional min/max
        min_vals[:, :, :, 0::2] = min_detection
        max_vals[:, :, :, 0::2] = max_detection

        # velocity augmentation
        if self.use_max_velocity or frame_index <= frame_change_threshold:
            offset = torch.tensor([T * max_velocity, 0, T * max_velocity, 0, T * max_velocity, 0], device=detection.device)

            max_vals += offset.view(1, 1, 1, -1)
            max_vals[:, :, :, 1::2] = max_velocity
            min_vals -= offset.view(1, 1, 1, -1)
            min_vals[:, :, :, 1::2] = -max_velocity
        else:
            # dynamic velocity scaling
            velocity_est, _ = torch.max(torch.abs(estimation[:, :, 1::2].unsqueeze(dim=2)), dim=1, keepdim=True)
            vel_pred_upper = (1 - alpha) * max_velocity + alpha * (velocity_est + T * max_acceleration) * 1.5
            vel_pred_upper = vel_pred_upper.detach()
            max_vals[:, :, :, 0::2] += T * vel_pred_upper
            min_vals[:, :, :, 0::2] -= T * vel_pred_upper
            max_vals[:, :, :, 1::2] = vel_pred_upper
            min_vals[:, :, :, 1::2] = -vel_pred_upper

        return min_vals, max_vals

    def _scale_tensor(self, x: torch.Tensor, min_vals: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        """
        Scale tensor based on min_vals and denominator.
        """
        if x.shape[-1] == 6:
            return (x - min_vals) / denom
        else:
            return (x - min_vals[:, :, :, 0::2]) / denom[:, :, :, 0::2]

    def deMinMaxScaler(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse scaling."""
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("Scaler has not been fitted. Run forward() first.")

        if self.mode == "0_1":
            return x * (self.max_vals - self.min_vals) + self.min_vals
        elif self.mode == "-1_1":
            return (x * 0.5 + 0.5) * (self.max_vals - self.min_vals) + self.min_vals
        else:
            raise ValueError("MyMinMaxScaler: invalid mode")

    def alpha_schedule(self, step: int, warmup: int = 15, target: float = 0.8) -> float:
        """Linear alpha warmup schedule."""
        return target if step >= warmup else target * (step / warmup)
