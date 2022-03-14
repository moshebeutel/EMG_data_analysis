# imports
import glob
import logging
import os.path
import torch
from sklearn import metrics
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from emg_pytorch_model import RawEmgConvnet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sns.set()
logger = utils.config_logger(os.path.basename(__file__)[:-3], level=logging.INFO)

# constants
NUM_OF_EPOCHS = 15
NUM_OF_USERS = 44
BATCH_SIZE = 11000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f'Device: {device}')
users_train_list = utils.FULL_USER_LIST
users_test_list = ['06']
users_train_list = [f for f in users_train_list if f not in users_test_list]
# assert int(len(users_train_list)) + int(len(users_test_list)) == num_of_users, 'Wrong Users Number'
logger.debug(f'User Train List:\n{users_train_list}')
logger.debug(f'User Test List:\n{users_test_list}')

# prepare train user filenames
user_trains = [f'emg_gestures-{user}-sequential' for user in users_train_list]
user_gesture_files = glob.glob(os.path.join(utils.HDF_FILES_DIR, "*.hdf5"))
train_user_files = [f for f in user_gesture_files if any([a for a in user_trains if a in f])]
num_of_classes = 10
model = RawEmgConvnet(number_of_class=num_of_classes, enhanced=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# train
y_pred = np.array([])
y_train = np.array([])
model.train()


def process_batch(X, y, batch_num):
    x_batch = X[batch_num * BATCH_SIZE:(batch_num + 1) * BATCH_SIZE, :].reshape((BATCH_SIZE, 1, 3, 8))
    x_batch = np.pad(x_batch, ((0, 0), (0, 0), (0, 0), (1, 1)), mode='wrap')
    x_batch = np.pad(x_batch, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0)
    input_batch = torch.from_numpy(x_batch).reshape((BATCH_SIZE, 1, 5, 10)).float().to(device)
    output_batch = model(input_batch)
    _, predicted_batch = torch.max(output_batch.data, 1)
    labels_batch = torch.from_numpy(np.array(y[batch_num * BATCH_SIZE:(batch_num + 1) * BATCH_SIZE]))\
        .long().reshape(BATCH_SIZE).to(device)
    loss_batch = criterion(output_batch, labels_batch)
    correct_batch = (predicted_batch == labels_batch).sum().item()
    del labels_batch, output_batch, input_batch, x_batch, predicted_batch
    return loss_batch, correct_batch


for epoch in range(NUM_OF_EPOCHS):
    running_loss = 0
    correct_train = 0
    total_train = 0
    counter = 0
    for train_file in train_user_files:
        # logger.info(f'Epoch: {epoch} Processing {train_file}')
        X_train, y_train_current = utils.prepare_X_y(train_file)
        num_samples_in_file = y_train_current.shape[0]
        y_pred_current = np.zeros((num_samples_in_file, num_of_classes))
        num_batches_in_file = int(num_samples_in_file / BATCH_SIZE)
        optimizer.zero_grad()
        for i in range(num_batches_in_file):  # list(range(max_ind))[::skip_rows]: #  range(X_train.shape[0]):
            counter += 1
            total_train += BATCH_SIZE
            logger.debug(f'Epoch: {epoch} Processing {train_file} batch num {i} batch size {BATCH_SIZE}'
                        f'  total_train {total_train} counter {counter}')

            loss, correct = process_batch(X_train, y_train_current, batch_num=i)

            correct_train += correct
            running_loss += float(loss)
            loss.backward()
            optimizer.step()
            del loss, correct
        del X_train, y_train_current, y_pred_current
    epoch_loss = running_loss / float(counter)
    train_acc = 100 * correct_train / float(total_train)
    logger.info(f'epoch {epoch}, train loss: {epoch_loss}, train_accuracy {train_acc}')

del train_user_files
# general metrics
logger.info('\nEnd of Training. Start testing')
# logger.info('\nEnd of Training Metrics:\n' + metrics.classification_report(y_pred, y_train))
del y_pred, y_train
# test
user_tests = [f'emg_gestures-{user}-sequential' for user in users_test_list]
logger.debug(f'test on {int(len(user_tests))}')
logger.debug(f'test on users: {user_tests}')
user_test_gesture_files = glob.glob(os.path.join(utils.HDF_FILES_DIR, "*.hdf5"))
test_user_files = [f for f in user_test_gesture_files if any([a for a in user_tests if a in f])]

y_pred = np.array([])
y_test = np.array([])
model.eval()
criterion = nn.CrossEntropyLoss()
test_loss = 0
correct_test = 0
total = 0
running_loss = 0
counter = 0
with torch.no_grad():
    for test_file in test_user_files:
        logging.info(f'Testing on {test_file}')
        X_test, y_test_current = utils.prepare_X_y(test_file)
        # test model
        logger.debug(f'Predicting {test_file}')
        num_samples_in_file = y_test_current.shape[0]
        y_pred_current = np.zeros((num_samples_in_file, num_of_classes))
        num_batches_in_file = int(num_samples_in_file / BATCH_SIZE)
        for i in range(num_batches_in_file):
            counter += 1
            total += BATCH_SIZE
            loss, correct = process_batch(X_test, y_test_current, batch_num=i)
            running_loss += float(loss)
            correct_test += correct

    los = running_loss / float(counter)
    acc = 100.0 * float(correct_test) / float(total)
    logger.info(f'test results loss: {los} accuracy: {acc}')
    del test_user_files
# logger.info('\nEnd of Test Metrics:\n' + metrics.classification_report(y_pred, y_test))
# conf_mat = metrics.confusion_matrix(y_test, y_pred)
# logger.info(f'\nConfusion Matrix\n{conf_mat}')
# sns.set(rc={'figure.figsize': (11.7, 8.27)})
# sns.heatmap(conf_mat.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()
