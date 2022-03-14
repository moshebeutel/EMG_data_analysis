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
logger = utils.config_logger(os.path.basename(__file__)[:-3])

# constants

num_of_users = 44

users_train_list = utils.FULL_USER_LIST # utils.get_users_list_from_dir(file_path)
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
model = RawEmgConvnet(number_of_class=num_of_classes, enhanced=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# train
y_pred = np.array([])
y_train = np.array([])
num_of_epochs = 5
model.train()
epoch_pbar = tqdm(range(num_of_epochs))
for epoch in epoch_pbar:
    running_loss = 0
    correct_train = 0
    total_train = 0
    counter = 0
    for train_file in train_user_files:
        # logger.info(f'Epoch: {epoch} Processing {train_file}')
        X_train, y_train_current = utils.prepare_X_y(train_file)
        y_pred_current = np.zeros((y_train_current.shape[0], num_of_classes))
        # logger.info(f'Fitting {train_file}')
        # model forward pass here
        optimizer.zero_grad()
        batch_loss = 0
        # TODO: written serially - convert to batch
        skip_rows = 40
        batch_size = 2750
        for i in list(range(batch_size * skip_rows))[::skip_rows]: #  range(X_train.shape[0]):
            counter += 1
            x = X_train[i,:].reshape((3,8))
            # x = np.pad(x, ((0,0),(1,1)), mode='wrap')
            # x = np.pad(x, ((1, 1), (0, 0)), mode='constant', constant_values=0)
            # input_tensor = torch.from_numpy(x).reshape((1, 1, 5, 10)).float()
            input_tensor = torch.from_numpy(x).reshape((1, 1, 3, 8)).float()
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            labels = torch.from_numpy(np.array(y_train_current[i])).long().reshape(1)
            correct_train += (predicted == labels).sum().item()
            loss = criterion(output, labels)
            running_loss += float(loss)
            loss.backward()
        total_train += batch_size
        optimizer.step()
        batch_loss /= float(batch_size)
        epoch_loss = running_loss / counter
        train_acc = 100 * float(correct_train) / float(total_train)
        del X_train, y_train_current, y_pred_current
    epoch_loss /= float(len(train_user_files))
    # logger.debug(f'epoch loss:{epoch_loss}')
    epoch_loss = running_loss / float(counter)
    train_acc = 100 * correct_train / total_train
    epoch_pbar.set_postfix({'epoch': epoch, 'train loss': epoch_loss, 'train_accuracy': train_acc})


del train_user_files
# general metrics
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
correct = 0
total = 0
running_loss = 0
counter = 0
with torch.no_grad():
    for test_file in test_user_files:
        logging.info(f'testing on {test_file}')
        X_test, y_test_current = utils.prepare_X_y(test_file)
        # test model
        logger.debug(f'predicting {test_file}')
        # TODO: written serially - convert to batch
        batch_size = 110000
        for i in range(batch_size):
            counter += 1
            x = X_test[i, :].reshape((3, 8))
            x = np.pad(x, ((0, 0), (1, 1)), mode='wrap')
            x = np.pad(x, ((1, 1), (0, 0)), mode='constant', constant_values=0)
            input_tensor = torch.from_numpy(x).reshape((1, 1, 5, 10)).float()
            output = model(input_tensor)
            labels = torch.from_numpy(np.array(y_test_current[i])).long().reshape(1)
            # y_pred_current[i] = model(torch.from_numpy(X_test[i, :]).reshape((1,1,3,8)).float())
            loss = criterion(output, labels)
            batch_loss += float(loss)
            running_loss += float(loss)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
        total += batch_size
    los = running_loss / float(counter)
    acc = 100.0 * float(correct) / float(total)
    logger.info(f'test results loss: {los} accuracy: {acc}')

    del test_user_files
    logger.debug(f'test_loss: {test_loss}')
# logger.info('\nEnd of Test Metrics:\n' + metrics.classification_report(y_pred, y_test))
# conf_mat = metrics.confusion_matrix(y_test, y_pred)
# logger.info(f'\nConfusion Matrix\n{conf_mat}')
# sns.set(rc={'figure.figsize': (11.7, 8.27)})
# sns.heatmap(conf_mat.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()
