import glob
import os
import torch
from torch.utils.data import Dataset
import utils
import pandas as pd


class EmgDatasetMap(Dataset):

    def __init__(self, users_list=utils.FULL_USER_LIST, load_to_memory=False,
                 data_dir=utils.FEATURES_DATAFRAMES_DIR, target_col='TRAJ_GT', trajectories=['sequential'],
                 transform=None, target_transform=None, max_cache_size=2, window_size=1, stride=1,
                 filter_fn=None, shrink_to_one_raw=False, logger=None, file_index=None):

        # list all filenames for given users and given trajectories
        user_trains = [f'emg_gestures-{user}-{traj}' for user in users_list for traj in trajectories]
        user_gesture_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
        train_user_files = [f for f in user_gesture_files if any([a for a in user_trains if a in f])]
        del user_gesture_files, user_trains
        if file_index is not None:
            train_user_files = [train_user_files[file_index]]
        if logger:
            logger.info(f'{len(train_user_files)} files in dataset')
        # index first raw of every file
        length = 0
        self.base_indices = pd.Series(dtype=str)
        self.df_cache = pd.DataFrame()

        for f in train_user_files:
            self.base_indices.loc[length] = f
            df = pd.read_hdf(f)
            df = df if filter_fn is None else filter_fn(df)
            length += int((int(df.shape[0]) - (window_size - stride)) / stride)
            # Add filename column
            df['filename'] = f
            # remove irrelevant fields
            df = df.drop(utils.COLS_TO_DROP, axis=1)

            if load_to_memory:
                # change action -1 to 10
                df.loc[df[target_col] < 0, target_col] = 10
                self.df_cache = pd.concat([self.df_cache, df], ignore_index=True)
            elif self.df_cache.columns.empty:
                self.df_cache = pd.DataFrame(columns=df.columns)
            del df
        del train_user_files
        if logger:
            logger.info(f'{length} data items in dataset')
        self.len = length
        self.target_col = target_col
        self.transform = transform
        self.target_transform = target_transform
        self.max_cache_size = max_cache_size
        self.window_size = window_size
        self.stride = stride
        self.filter = filter_fn
        self.shrink_to_one_raw = shrink_to_one_raw
        self.logger = logger

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
            # get largest base id smaller than ind
            assert self.base_indices is not None, 'base indices not initialized'
            assert isinstance(self.base_indices, pd.Series), 'base_indices not a Series'
            assert not self.base_indices.empty, 'base_indices empty'
            largest_base = max([base for base in self.base_indices.index if base <= ind])
            # get relevant data from cache
            assert 'filename' in self.df_cache.columns, 'filename column does not exist'
            df = self.df_cache[self.df_cache['filename'] == self.base_indices.loc[largest_base]]
            if df.empty:
                # dataframe not in cache - load from file
                # get relevant file
                f = self.base_indices.loc[largest_base]
                df = self._read_dataframe_to_cache(f)
                # keep cache size
                if self.df_cache.index.size > self.max_cache_size:
                    first_filename = self.df_cache.iloc[0, 'filename']
                    self.df_cache.drop(self.df_cache[self.df_cache['filename'] == first_filename].index)
            # raw index in file
            local_id = ind - largest_base
            local_id *= self.stride
            # take window of raws
            data = df.iloc[local_id:local_id + self.window_size, :].drop('filename', axis=1)
            # take target as last raw
            target = data.iloc[-1][self.target_col]
            data = data.drop([self.target_col]) if data is pd.Series else data.drop([self.target_col], axis=1)
            data_tensor_tmp = torch.unsqueeze(torch.tensor(data.values.astype(float), dtype=torch.float32), dim=0)
            if self.shrink_to_one_raw:
                data_tensor_tmp = data_tensor_tmp.reshape(1, -1, 3, 8).mean(axis=2)
            target_tensor_tmp = torch.tensor(target, dtype=torch.long)
            del data, target, df
            # aggregate data in tensor
            data_tensor = data_tensor_tmp if data_tensor is None \
                else torch.concat([data_tensor, data_tensor_tmp], axis=0)
            target_tensor = target_tensor_tmp if target_tensor is None \
                else torch.concat([target_tensor, target_tensor_tmp], axis=0)
        return data_tensor, target_tensor

    def _read_dataframe_to_cache(self, f: str):
        assert os.path.exists(f), f'File {f} not found'
        # load dataframe from file
        df = pd.read_hdf(f)
        # filter dataframe if filter was given
        df = self.filter(df) if self.filter is not None else df
        # remove irrelevant fields
        df = df.drop(utils.COLS_TO_DROP, axis=1)
        # change action -1 to 10
        df.loc[df[self.target_col] < 0, self.target_col] = 10
        # add filename column
        df['filename'] = f
        # insert to cache
        self.df_cache = pd.concat([self.df_cache, df], ignore_index=True)
        return df
