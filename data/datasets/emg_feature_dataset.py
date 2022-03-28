import glob
import os
import torch
from torch.utils.data import Dataset
import utils
import pandas as pd


def _create_channel_for_feature(data, feature_data_tensor, feature_name):
    feature_columns = [col_name for col_name in data.index if feature_name in col_name]
    assert len(feature_columns) > 0, f'No columns for feature {feature_name}'
    feature_data = data[feature_columns]
    feature_data_tensor_tmp = torch.unsqueeze(torch.tensor(feature_data.values.astype(float),
                                                           dtype=torch.float32), dim=0)
    feature_data_tensor = feature_data_tensor_tmp if feature_data_tensor is None \
        else torch.concat([feature_data_tensor, feature_data_tensor_tmp], axis=0)
    del feature_columns, feature_data, feature_data_tensor_tmp
    return feature_data_tensor


class EmgFeatureDataset(Dataset):
    def __init__(self, users_list=utils.FULL_USER_LIST,
                 data_dir=utils.FEATURES_DATAFRAMES_DIR,
                 feature_names=utils.TD_FEATURES,
                 target_col='TRAJ_GT_LAST',
                 trajectories=['sequential'],
                 logger=None):
        assert logger is not None, 'Logger not given'
        logger.debug(f'EmgFeatureDataset for users {users_list} trajectories {trajectories}')

        # list all filenames for given users and given trajectories
        user_trains = [f'emg_gestures-{user}-{traj}' for user in users_list for traj in trajectories]
        user_gesture_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
        train_user_files = [f for f in user_gesture_files if any([a for a in user_trains if a in f])]
        logger.debug(f'train_user_files={train_user_files}')
        del user_gesture_files, user_trains

        self.df_cache = pd.DataFrame()
        for f in train_user_files:
            df = pd.read_hdf(f)
            df = df[df[target_col] > 0]
            # remove irrelevant fields
            # df = df.drop(utils.COLS_TO_DROP, axis=1)
            assert target_col in df.columns, f'target col {target_col} not in dataframe columns at file {f}'
            # change action -1 to 10
            df.loc[df[target_col] < 0, target_col] = 10
            self.df_cache = pd.concat([self.df_cache, df], ignore_index=True)
            del df
        del train_user_files
        self.len = self.df_cache.index.size
        self.target_col = target_col
        self.logger = logger
        self.feature_list = feature_names
        logger.info(f'Dataset initialized with {self.len} items')

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
            data_item = self.df_cache.iloc[ind, :].drop([self.target_col])

            feature_data_tensor = None
            for feature_name in self.feature_list:
                feature_data_tensor = _create_channel_for_feature(data_item, feature_data_tensor,
                                                                       feature_name)

            feature_data_tensor = torch.unsqueeze(feature_data_tensor, dim=0)
            data_tensor = feature_data_tensor if data_tensor is None \
                else torch.concat([data_tensor, feature_data_tensor], axis=0)

            target = self.df_cache[self.target_col].iloc[ind]
            target_tensor_tmp = torch.tensor(int(target), dtype=torch.long)
            target_tensor = target_tensor_tmp if target_tensor is None \
                else torch.concat([target_tensor, target_tensor_tmp], axis=0)

            del data_item, target, target_tensor_tmp

        return data_tensor, target_tensor


if __name__ == '__main__':
    logger = utils.config_logger('feature_main', log_folder='../../log/')
    datframes_dir = os.path.join('../../', utils.FEATURES_DATAFRAMES_DIR)
    dataset = EmgFeatureDataset(users_list=['03'], data_dir=datframes_dir, logger=logger)
    data, label = dataset[1000]
    logger.debug(f'data shape {data.shape}')
    logger.debug(f'data type {data.dtype}')
    logger.debug(f'label {label}')
