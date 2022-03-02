# imports
import logging
import os.path
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import utils

logger = utils.config_logger(os.path.basename(__file__)[:-3])

# constants
file_path = '../putemg-downloader/Data-HDF5'
train_name = 'emg_gestures-03-repeats_long-2018-05-11-11-05-00-695.hdf5'
test_name = 'emg_gestures-03-repeats_long-2018-06-14-12-32-53-659.hdf5'

# prepare train user filenames
user_gesture_train_file = os.path.join(file_path, train_name)
user_gesture_test_file = os.path.join(file_path, test_name)

rf_model = RandomForestClassifier(n_estimators=5)


# train
logger.info(f'Processing {user_gesture_train_file}')
X_train, y_train = utils.prepare_X_y(user_gesture_train_file)
logger.debug(f'rf estimators {rf_model.n_estimators}')
logger.info(f'Fitting {user_gesture_train_file}')
rf_model.fit(X_train, y_train)
logger.info(f'Finished fitting {user_gesture_train_file}')
del X_train, y_train

# test
logging.info(f'testing on {user_gesture_test_file}')
X_test, y_test = utils.prepare_X_y(user_gesture_test_file)
# test model
logger.debug(f'predicting {user_gesture_test_file}')
y_pred = rf_model.predict(X_test)
# metrics
logger.info(metrics.classification_report(y_pred, y_test))
del X_test, y_test
