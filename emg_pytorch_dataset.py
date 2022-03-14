import glob
import os
import torch
from torch.utils.data import Dataset
import utils
import pandas as pd


class EmgDatasetMap(Dataset):
    def __init__(self, users_list=utils.FULL_USER_LIST,
                 data_dir=utils.FEATURES_DATAFRAMES_DIR, target_col='TRAJ_GT', transform=None, target_transform=None):
        user_trains = [f'emg_gestures-{user}-repeats_long' for user in users_list]
        user_gesture_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
        self.train_user_files = [f for f in user_gesture_files if any([a for a in user_trains if a in f])]
        length = 0
        self.base_indxs: dict = {}
        for f in self.train_user_files:
            self.base_indxs[length] = f
            df = pd.read_hdf(f)
            length += int(df.shape[0])
            del df
        self.len = length
        self.base_indxs_keys = list(self.base_indxs.keys())
        self.target_col = target_col
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get largest base id smaller than idx
        id = max([id for id in self.base_indxs_keys if id < idx])
        f = self.base_indxs[id]
        local_id = idx - id
        df = pd.read_hdf(f)
        data = df.iloc[local_id,:]
        target = data[self.target_col]
        data = data.drop([self.target_col], axis=1)
        del df
        return data, target

