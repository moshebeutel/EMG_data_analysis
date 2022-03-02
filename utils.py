import logging
import time
import sys
import pandas as pd

cols_to_drop = ['TRAJ_1', 'type', 'subject', 'trajectory', 'date_time', 'TRAJ_GT_NO_FILTER', 'VIDEO_STAMP']


def config_logger(name='default'):
    # config logger
    log_format = '%(created)f:%(levelname)s:%(name)s:%(module)s:%(message)s'
    formatter = logging.Formatter(log_format)
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        filename=f'./log/{time.ctime()}_{name}.log',
                        filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    created_logger = logging.getLogger(name + '_logger')
    created_logger.addHandler(handler)
    return created_logger


def prepare_X_y(data_file: str):
    # read dataframe from file
    df = pd.read_hdf(data_file)
    # drop irrelevant columns
    df.drop(cols_to_drop, axis=1, inplace=True)
    # remove "idle" and "relax"
    df = df[df.TRAJ_GT > 0]
    X = df.iloc[:, 0:24].to_numpy()
    y = df.TRAJ_GT
    del df
    return X, y
