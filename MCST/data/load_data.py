import numpy as np
import os
import scipy.io as io
import torch
import torch.utils.data as data
from path import Path
from torch.distributions import Normal

class DataLoad_OneManeuveringTarget_3D(data.Dataset):
    def __init__(self, data_path, is_train, n_frames):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(DataLoad_OneManeuveringTarget_3D, self).__init__()
        if is_train:
            self.data_root = Path(data_path + "train")
        else:
            self.data_root = Path(data_path + "test")
        self.dataSet_length = len(os.listdir(self.data_root))
        self.all_scence_path = [str(path) for path in os.listdir(self.data_root)]
        self.n_frames = n_frames

    def __getitem__(self, idx):
        scence_path = self.all_scence_path[idx]
        scence_path = os.path.join(self.data_root, scence_path)
        frame_list = os.listdir(scence_path)
        state_labels = []
        detections=[]

        frame_list.sort()
        for frame_id in range(len(frame_list)):
            detail_frame_path = os.path.join(scence_path, frame_list[frame_id])
            in_out = np.load(detail_frame_path, allow_pickle=True).tolist()
            param = torch.tensor(in_out, dtype=torch.float32)

            if True in (torch.isnan(param)):
                print('error')

            if frame_id < self.n_frames:
                detections.append(np.transpose(param).unsqueeze(dim=0))
            else:
                state_labels.append(np.transpose(param).unsqueeze(dim=0))

        state_labels = torch.cat(state_labels, dim=0)
        detections = torch.cat(detections, dim=0)

        return [detections, state_labels]

    def __len__(self):
        return self.dataSet_length


def DataLoadFromMatlab_oneTarget_3D_for_paper():
    state_labels = []
    detections =[]
    for frame in range(1, 201):
        path = "./dataset/scene007/"+"state_labels" + ('%02d' % frame) + ".mat"
        matr = io.loadmat(path)
        savedata_origin = matr.get('savedata')
        savedata_origin= savedata_origin.astype(float)
        state_labels.append(torch.tensor(np.transpose(savedata_origin)).unsqueeze(dim=0))
    state_labels = torch.cat(state_labels, dim=0)

    for frame in range(1, 201):
        path = "./dataset/scene007/"+"Ob" + ('%02d' % frame) + ".mat"
        matr = io.loadmat(path)
        savedata_ob = matr.get('savedata')
        savedata_ob = savedata_ob.astype(float)
        detections.append(torch.tensor(np.transpose(savedata_ob)).unsqueeze(dim=0))
    detections = torch.cat(detections, dim=0)

    state_labels =state_labels.unsqueeze(dim=0)
    detections = detections.unsqueeze(dim=0)

    return state_labels, detections


def enu2rae(enu_x, enu_y, enu_z):
    distance = (enu_x ** 2 + enu_y ** 2 + enu_z ** 2) ** 0.5
    azi = torch.atan2(enu_y, enu_x)
    ele = torch.atan2(enu_z, (enu_x ** 2 + enu_y ** 2) ** 0.5)
    return (distance, azi, ele)


def rae2enu(distance, azi, ele):
    enu_x = distance * torch.cos(azi) * torch.cos(ele)
    enu_y = distance * torch.sin(azi) * torch.cos(ele)
    enu_z = distance * torch.sin(ele)
    return (enu_x, enu_y, enu_z)