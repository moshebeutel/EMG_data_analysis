import logging

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse

import biolab_utilities
# import pyeeg
# from putemg_features import feature_mav, feature_zc, feature_ssc, feature_wl
import pandas as pd
import os
import numpy as np
import utils
from tqdm import tqdm

logger = utils.config_logger(os.path.basename(__file__)[:-3], level=logging.INFO)


# calculate features
# Mean Absolute Value (MAV), Zero Crossing (ZC), Slope Sign Changes (SSC) and Waveform Length (WL).
# the following feature calculation copied from https://github.com/biolab-put/putemg_features.git
def feature_mav(series, window, step):
    """Mean Absolute Value"""
    windows_strided, indexes = biolab_utilities.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.mean(np.abs(windows_strided), axis=1), index=series.index[indexes])


def feature_zc(series, window, step, threshold):
    """Zero Crossing"""
    windows_strided, indexes = biolab_utilities.moving_window_stride(series.values, window, step)
    zc = np.apply_along_axis(lambda x: np.sum(np.diff(x[(x < -threshold) | (x > threshold)] > 0)), axis=1,
                             arr=windows_strided)
    return pd.Series(data=zc, index=series.index[indexes])


def feature_ssc(series, window, step, threshold):
    """Slope Sign Change"""
    windows_strided, indexes = biolab_utilities.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.apply_along_axis(lambda x: np.sum((np.diff(x[:-1]) * np.diff(x[1:])) <= -threshold),
                                              axis=1, arr=windows_strided), index=series.index[indexes])


def feature_wl(series, window, step):
    """Waveform Length"""
    windows_strided, indexes = biolab_utilities.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.diff(windows_strided), axis=1), index=series.index[indexes])


def feature_last(series, window, step):
    """Last value of the window - resampling"""
    windows_strided, indexes = biolab_utilities.moving_window_stride(series.values, window, step)
    return pd.Series(data=windows_strided[::, -1], index=series.index[indexes])


def feature_butterworth(series, window, step):
    """Butterworth Filter"""
    windows_strided, indexes = biolab_utilities.moving_window_stride(series.values, window, step)


def calculate_feature_dataframe(record: pd.DataFrame) -> pd.DataFrame:
    calculated_features_df = pd.DataFrame()
    pbar = tqdm(record.columns[0:24])
    for col in pbar:
        pbar.set_description("Processing %s" % col)
        calculated_features_df[f'{col}_WL'] = feature_wl(record.loc[:, col], window=256, step=128)
        calculated_features_df[f'{col}_SSC'] = feature_ssc(record.loc[:, col], window=256, step=128, threshold=0.2)
        calculated_features_df[f'{col}_ZC'] = feature_zc(record.loc[:, col], window=256, step=128, threshold=0.2)
        calculated_features_df[f'{col}_MAV'] = feature_mav(record.loc[:, col], window=256, step=128)
    calculated_features_df['TRAJ_GT_LAST'] = feature_last(record.loc[:, 'TRAJ_GT'], window=256, step=128)
    return calculated_features_df


def create_feature_hdfs(users_list=utils.FULL_USER_LIST, trajectory='sequential'):
    users_pbar = tqdm(users_list)
    for user in users_pbar:
        users_pbar.set_description("Calculating features for user %s" % user)
        trial_train, trial_test = utils.get_traj_for_user(utils.HDF_FILES_DIR, trajectory, user)
        record_train = utils.read_trial(trial_train)
        record_test = utils.read_trial(trial_test)
        features_df_train: pd.DataFrame = calculate_feature_dataframe(record_train)
        features_df_train.to_hdf(os.path.join(utils.FEATURES_DATAFRAMES_DIR, 'features_' + trial_train),
                                 key='features_df_train', mode='w')
        features_df_test: pd.DataFrame = calculate_feature_dataframe(record_test)
        features_df_test.to_hdf(os.path.join(utils.FEATURES_DATAFRAMES_DIR, 'features_' + trial_test),
                                key='features_df_test', mode='w')
    del trial_train, trial_test, record_train, record_test, features_df_train, features_df_test


def between_days_same_user_for_all_users():
    users_pbar = tqdm(utils.FULL_USER_LIST)
    for user in users_pbar:
        users_pbar.set_description("Fitting using features for user %s" % user)
        between_days_same_user(user)


def between_days_same_user(user):
    trial_train, trial_test = utils.get_traj_for_user(utils.HDF_FILES_DIR, 'repeats_long', user)
    features_df_train = pd.read_hdf(os.path.join(utils.FEATURES_DATAFRAMES_DIR, 'features_' + trial_train))
    features_df_test = pd.read_hdf(os.path.join(utils.FEATURES_DATAFRAMES_DIR, 'features_' + trial_test))
    rf_model = RandomForestClassifier(n_estimators=30)
    lda_model = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    # train
    logger.debug(f'Processing features_{trial_train}')
    X_train, y_train = utils.prepare_X_y_from_dataframe(features_df_train, target='TRAJ_GT_LAST')
    logger.debug(f'rf estimators {rf_model.n_estimators}')
    logger.info(f'\nFitting features_{trial_train}')
    # rf_model.fit(X_train, y_train)
    lda_model.fit(X_train, y_train)
    logger.debug(f'Finished fitting features_{trial_train}')
    del X_train, y_train, features_df_train
    # test
    logger.debug(f'testing on features_{trial_test}')
    X_test, y_test = utils.prepare_X_y_from_dataframe(features_df_test, target='TRAJ_GT_LAST')
    # test model
    logger.debug(f'predicting features_{trial_test}')
    # y_pred = rf_model.predict(X_test)
    y_pred = lda_model.predict(X_test)
    # metrics
    logger.info('\n' + str(metrics.classification_report(y_pred, y_test)))
    del X_test, y_test, features_df_test


if __name__ == '__main__':
    # create_feature_hdfs(utils.FULL_USER_LIST)
    argparse = parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', default='03', type=str,
                        help='user ID to evaluate', choices=utils.FULL_USER_LIST)
    parser.add_argument('-a', '--all_users', default=False, type=bool, help='Evaluate on all users')

    args = parser.parse_args()

    if args.all_users:
        between_days_same_user_for_all_users()
    else:
        between_days_same_user(args.user)
