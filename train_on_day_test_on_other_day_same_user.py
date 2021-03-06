# imports
import argparse
import logging
import os.path
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import utils


def main(args):
    logger = utils.config_logger(os.path.basename(__file__)[:-3], level=logging.INFO)

    train_name, test_name = utils.get_traj_for_user(args.file_path, args.traj, args.user)

    # prepare train user filenames
    user_gesture_train_file = os.path.join(args.file_path, train_name)
    user_gesture_test_file = os.path.join(args.file_path, test_name)

    model = RandomForestClassifier(n_estimators=30) if args.model == 'rf' \
        else LinearDiscriminantAnalysis(solver="svd", store_covariance=True)

    # train
    logger.info(f'Processing {user_gesture_train_file}')
    X_train, y_train = utils.prepare_X_y(user_gesture_train_file)
    X_val, y_val = X_train[-2000:], y_train[-2000:]
    X_train, y_train = X_train[:-2000], y_train[:-2000]
    # logger.debug(f'rf estimators {rf_model.n_estimators}')
    logger.info(f'Fitting {user_gesture_train_file}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    logger.debug(f'Finished fitting {user_gesture_train_file}')
    logger.info('\nValidation metrics\n' + str(metrics.classification_report(y_pred, y_val)))
    del X_train, y_train, y_pred

    # test
    logging.info(f'testing on {user_gesture_test_file}')
    X_test, y_test = utils.prepare_X_y(user_gesture_test_file)
    # test model
    logger.debug(f'predicting {user_gesture_test_file}')
    y_pred = model.predict(X_test)
    # metrics
    logger.info(metrics.classification_report(y_pred, y_test))
    del X_test, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['rf', 'lda'], default='rf')
    parser.add_argument('-fp', '--file_path', type=str, help='Path to experiments directory',
                        default='../putemg-downloader/Data-HDF5')
    parser.add_argument('-t', '--traj', type=str, help='Trajectory type', choices=utils.FULL_TRAJ_LIST,
                        default='repeats_long')
    parser.add_argument('-u', '--user', type=str, help='Two digit identifier', choices=utils.FULL_USER_LIST,
                        default='03')
    args = parser.parse_args()
    main(args)
