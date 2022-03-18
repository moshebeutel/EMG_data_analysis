from emg_pytorch_dataset import EmgDatasetMap
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from emg_pytorch_model import RawEmgConvnet
import utils
import os
import logging
import glob
from tqdm import tqdm

logger = utils.config_logger(os.path.basename(__file__)[:-3], level=logging.DEBUG)

# constants
NUM_OF_EPOCHS = 3
NUM_OF_USERS = 44
BATCH_SIZE = 22000
WINDOW_SIZE = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
logger.debug(f'Device: {device}')
users_train_list = ['03']  # utils.FULL_USER_LIST
users_test_list = ['06']
users_train_list = [f for f in users_train_list if f not in users_test_list]
# assert int(len(users_train_list)) + int(len(users_test_list)) == num_of_users, 'Wrong Users Number'
logger.debug(f'User Train List:\n{users_train_list}')
logger.debug(f'User Test List:\n{users_test_list}')

# prepare train user filenames
user_trains = [f'emg_gestures-{user}-sequential' for user in users_train_list]
user_gesture_files = glob.glob(os.path.join(utils.HDF_FILES_DIR, "*.hdf5"))
train_user_files = [f for f in user_gesture_files if any([a for a in user_trains if a in f])]

dataset = EmgDatasetMap(users_list=['03', '04'], data_dir=utils.HDF_FILES_DIR, window_size=WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)

num_of_classes = 11
model = RawEmgConvnet(number_of_class=num_of_classes, enhanced=False).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000011288378916846883)

# train
train_loss_list, train_accuracy = [], []
y_pred = np.array([])
y_train = np.array([])
model.train()

train_losses, train_accs = [], []
epoch_pbar = tqdm(range(NUM_OF_EPOCHS))
for epoch in range(NUM_OF_EPOCHS):
    running_loss = 0
    correct_train = 0
    total_train = 0
    counter = 0
    for i, data in enumerate(dataloader, 0):
        counter += 1
        emg_data = data[0].to(device)
        labels = data[1].to(device)
        total_train += labels.size(0)
        optimizer.zero_grad()
        outputs = model(emg_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        correct_train += int(correct)
        running_loss += float(loss)
        logger.debug(f'Epoch {epoch} batch num {i} loss {float(loss)} accuracy {float(correct)/float(labels.size(0))}')
        del loss, outputs, emg_data, labels, predicted, correct
    epoch_loss = running_loss / float(counter)
    train_acc = 100 * correct_train / total_train
    train_loss_list.append(epoch_loss)
    train_accuracy.append(train_acc)
    # epoch_pbar.set_postfix({'epoch': epoch, 'train loss': epoch_loss, 'train_accuracy': train_acc, 'val loss': val_los,
    #                         'val_accuracy': val_acc})
    epoch_pbar.set_postfix({'epoch': epoch, 'train loss': epoch_loss, 'train_accuracy': train_acc})

utils.show_learning_curve(train_loss_list, [], train_accuracy,[], NUM_OF_EPOCHS, title='window 2d loss and accuracy')
