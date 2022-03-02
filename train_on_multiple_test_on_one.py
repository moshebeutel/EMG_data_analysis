# imports
import glob
import logging
import os.path
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
logger = utils.config_logger(os.path.basename(__file__)[:-3])

# constants
file_path = '../putemg-downloader/Data-HDF5'
trial_name = 'emg_gestures-03-repeats_long-2018-05-11-11-05-00-695'
filtered_file = '/filtered-03-repeats_long-2018-05-11-11-05-00-695'
users_train_list = ['03', '04', '05', '07']
users_test_list = ['06']

# prepare train user filenames
user_trains = [f'emg_gestures-{user}-repeats_long' for user in users_train_list]
user_gesture_files = glob.glob(os.path.join(file_path, "*.hdf5"))
train_user_files = [f for f in user_gesture_files if any([a for a in user_trains if a in f])]

rf_model = RandomForestClassifier(n_estimators=0, warm_start=True)

# train
y_pred = np.array([])
y_train = np.array([])
for train_file in train_user_files:
    logger.info(f'Processing {train_file}')
    X_train, y_train_current = utils.prepare_X_y(train_file)
    rf_model.n_estimators += 10
    logger.debug(f'rf estimators {rf_model.n_estimators}')
    logger.info(f'Fitting {train_file}')
    rf_model.fit(X_train, y_train_current)
    logger.info(f'Finished fitting {train_file}')
    y_pred_current = rf_model.predict(X_train)
    logger.info(f'\n{train_file} Training Metrics:\n' + metrics.classification_report(y_pred_current, y_train_current))
    y_pred = np.concatenate((y_pred, y_pred_current), axis=None)
    y_train = np.concatenate((y_train, y_train_current), axis=None)
    del X_train, y_train_current, y_pred_current
del train_user_files
logger.info('\nEnd of Training Metrics:\n' + metrics.classification_report(y_pred, y_train))
del y_pred, y_train
# test
user_tests = [f'emg_gestures-{user}-repeats_long' for user in users_test_list]
logger.debug(f'test on {int(len(user_tests))}')
logger.debug(f'test on users: {user_tests}')
user_test_gesture_files = glob.glob(os.path.join(file_path, "*.hdf5"))
test_user_files = [f for f in user_test_gesture_files if any([a for a in user_tests if a in f])]
y_pred = np.array([])
y_test = np.array([])
for test_file in test_user_files:
    logging.info(f'testing on {test_file}')
    X_test, y_test_current = utils.prepare_X_y(test_file)
    # test model
    logger.debug(f'predicting {test_file}')
    y_pred_current = rf_model.predict(X_test)
    # metrics
    logger.info(f'\n{test_file} Test Metrics:\n' + metrics.classification_report(y_pred_current, y_test_current))
    y_pred = np.concatenate((y_pred, y_pred_current), axis=None)
    y_test = np.concatenate((y_test, y_test_current), axis=None)
    del X_test, y_test_current, y_pred_current
del test_user_files
logger.info('\nEnd of Test Metrics:\n' + metrics.classification_report(y_pred, y_test))
conf_mat = metrics.confusion_matrix(y_test, y_pred)
logger.info(f'\nConfusion Matrix\n{conf_mat}')
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.heatmap(conf_mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
