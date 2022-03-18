import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import utils
import pandas as pd


class EmgDatasetMap(Dataset):

    def __init__(self, users_list=utils.FULL_USER_LIST,
                 data_dir=utils.FEATURES_DATAFRAMES_DIR, target_col='TRAJ_GT', trajectories=['sequential'],
                 transform=None, target_transform=None, max_cache_size=2, window_size=0):

        # list all filenames for given users and given trajectories
        user_trains = [f'emg_gestures-{user}-{traj}' for user in users_list for traj in trajectories]
        user_gesture_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
        self.train_user_files = [f for f in user_gesture_files if any([a for a in user_trains if a in f])]

        # index first raw of every file
        length = 0
        self.base_indices: dict = {}
        for f in self.train_user_files:
            self.base_indices[length] = f
            df = pd.read_hdf(f)
            length += int(df.shape[0]) - window_size
            del df
        self.len = length
        self.base_indices_keys = list(self.base_indices.keys())
        self.target_col = target_col
        self.transform = transform
        self.target_transform = target_transform

        self.max_cache_size = max_cache_size
        self.df_cache: dict = {}

        self.window_size = window_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        assert type(idx) == list, f'idx should be a list of indices, got {type(idx)} - {idx}'
        assert all([isinstance(idx_item, int) for idx_item in idx]), f'idx should be a list of integers'
        idx = sorted(idx)
        data_tensor = None
        target_tensor = None
        for ind in idx:
            # get largest base id smaller than idx
            largest_base = max([base for base in self.base_indices_keys if base <= ind])
            if largest_base in self.df_cache.keys():
                # dataframe already in cache
                df = self.df_cache[largest_base]
            else:
                # dataframe not in cache - load from file
                # get relevant file
                f = self.base_indices[largest_base]
                # load dataframe from file
                df = pd.read_hdf(f)
                # remove irrelevant fields
                df = df.drop(utils.COLS_TO_DROP, axis=1)
                # change action -1 to 10
                df.loc[df[self.target_col] < 0, self.target_col] = 10
                # insert to cache
                self.df_cache[largest_base] = df
                # keep cache size
                if len(self.df_cache) > self.max_cache_size:
                    self.df_cache.pop(list(self.df_cache.keys())[0])
            local_id = ind - largest_base
            data = df.iloc[local_id:local_id+self.window_size, :]
            target = data.iloc[-1][self.target_col]
            data = data.drop([self.target_col]) if data is pd.Series else data.drop([self.target_col], axis=1)
            data_tensor_tmp = torch.unsqueeze(torch.tensor(data.values.astype(float), dtype=torch.float32), dim=0)
            target_tensor_tmp = torch.tensor(target, dtype=torch.long)
            del data, target
            # aggregate data in tensor
            data_tensor = data_tensor_tmp if data_tensor is None \
                else torch.concat([data_tensor, data_tensor_tmp], axis=0)
            target_tensor = target_tensor_tmp if target_tensor is None \
                else torch.concat([target_tensor, target_tensor_tmp], axis=0)
        return data_tensor, target_tensor




