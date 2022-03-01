# imports
import glob
import logging
import sys
import os.path
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def config_logger():
    # config logger
    format = '%(created)f:%(levelname)s:%(name)s:%(module)s:%(message)s'
    formatter = logging.Formatter(format)
    logging.basicConfig(level=logging.DEBUG,
                        format=format,
                        filename='./log/myapp.log',
                        filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__ + '_logger')
    logger.addHandler(handler)
    return logger


logger = config_logger()

sns.set()

# constants
file_path = '../putemg-downloader/Data-HDF5'
trial_name = 'emg_gestures-03-repeats_long-2018-05-11-11-05-00-695'
filtered_file = '/filtered-03-repeats_long-2018-05-11-11-05-00-695'
users_train_list = ['03', '04', '05']
users_test_list = ['06', '07']
# users_list = ['03', '04', '05', '06', '07']

# prepare train user filenames
user_trains = [f'emg_gestures-{user}-repeats_long' for user in users_train_list]
user_gesture_files = glob.glob(os.path.join(file_path, "*.hdf5"))
train_user_files = [f for f in user_gesture_files if any([a for a in user_trains if a in f])]

rf_model = RandomForestClassifier(n_estimators=0, warm_start=True)


def prepare_X_y(data_file):
    cols_to_drop = ['TRAJ_1', 'type', 'subject', 'trajectory', 'date_time', 'TRAJ_GT_NO_FILTER', 'VIDEO_STAMP']
    df = pd.read_hdf(data_file)
    df.drop(cols_to_drop, axis=1, inplace=True)
    X = df.iloc[:, 0:24].to_numpy()
    y = df.TRAJ_GT
    del df
    return X, y


# train
for train_file in train_user_files:
    logger.info(f'Processing {train_file}')
    X_train, y_train = prepare_X_y(train_file)
    rf_model.n_estimators += 10
    logger.debug(f'rf estimators {rf_model.n_estimators}')
    logger.info(f'Fitting {train_file}')
    rf_model.fit(X_train, y_train)
    logger.info(f'Finished fitting {train_file}')
    del X_train, y_train
del train_user_files

# test
user_tests = [f'emg_gestures-{user}-repeats_long' for user in users_test_list]
logger.debug(f'test on {int(len(user_tests))}')
logger.debug(f'test on users: {user_tests}')
user_test_gesture_files = glob.glob(os.path.join(file_path, "*.hdf5"))
test_user_files = [f for f in user_test_gesture_files if any([a for a in user_tests if a in f])]

# test_user_files = [f for f in user_test_gesture_files if 'emg_gestures-07-repeats_long' in f]
for test_file in test_user_files:
    logging.info(f'testing on {test_file}')
    X_test, y_test = prepare_X_y(test_file)
    # test model
    logger.debug(f'predicting {test_file}')
    y_pred = rf_model.predict(X_test)
    # metrics
    logger.info(metrics.classification_report(y_pred, y_test))
    del X_test, y_test

del test_user_files
